"""Tests for RF3 chiral feature tracking and zeroing."""

import pytest
import torch
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.utils.imports import require_rf3, RF3_AVAILABLE


if RF3_AVAILABLE:
    from sampleworks.models.rf3.wrapper import annotate_structure_for_rf3


def _make_sampler(device: torch.device) -> AF3EDMSampler:
    return AF3EDMSampler(EDMSamplerConfig(device=device, augmentation=False, align_to_input=False))


def _run_steps(wrapper, features, num_steps: int, seed: int = 42):
    """Run num_steps of denoising from seeded prior noise. Returns the final state."""
    device = wrapper.device
    sampler = _make_sampler(device)
    schedule = sampler.compute_schedule(num_steps=num_steps)

    torch.manual_seed(seed)
    state = wrapper.initialize_from_prior(batch_size=1, features=features)

    for i in range(num_steps):
        context = sampler.get_context_for_step(i, schedule)
        t = context.t.to(device)  # ty: ignore[unresolved-attribute]
        state = wrapper.step(state, t, features=features)

    return state


@pytest.mark.gpu
@pytest.mark.slow
class TestRF3ChiralFeatures:
    """Validate chiral feature tracking and zeroing in RF3Wrapper."""

    @require_rf3()
    def test_track_chiral_features_populates_stats(self, rf3_wrapper, structure_1vme):
        """track_chiral_features=True populates _chiral_grad_stats with one
        entry per denoising step, each containing 't' and 'l2_norm' keys."""
        annotated = annotate_structure_for_rf3(
            structure_1vme,
            track_chiral_features=True,
            disable_chiral_features=False,
        )
        features = rf3_wrapper.featurize(annotated)

        # Stats should be empty right after featurize
        assert rf3_wrapper._chiral_grad_stats == []

        num_steps = 5
        _run_steps(rf3_wrapper, features, num_steps)

        stats = rf3_wrapper._chiral_grad_stats
        assert len(stats) == num_steps, (
            f"Expected {num_steps} chiral stat entries, got {len(stats)}"
        )

        for entry in stats:
            assert "t" in entry, "Missing 't' key in chiral grad stats entry"
            assert "l2_norm" in entry, "Missing 'l2_norm' key in chiral grad stats entry"
            assert isinstance(entry["l2_norm"], float)
            # for a real protein L2 should be positive
            assert entry["l2_norm"] > 0.0

    @require_rf3()
    def test_disable_chiral_features_zeros_features(self, rf3_wrapper, structure_1vme):
        """disable_chiral_features=True sets chiral_centers to an empty (0,4)
        tensor and chiral_center_dihedral_angles to empty (0,) tensor in the
        conditioning features dict."""
        annotated = annotate_structure_for_rf3(
            structure_1vme,
            disable_chiral_features=True,
            track_chiral_features=False,
        )
        features = rf3_wrapper.featurize(annotated)
        cond = features.conditioning

        chiral_centers = cond.features["chiral_centers"]
        chiral_angles = cond.features["chiral_center_dihedral_angles"]

        assert chiral_centers.shape == (0, 4), (
            f"Expected chiral_centers shape (0, 4), got {chiral_centers.shape}"
        )
        assert chiral_angles.shape == (0,), (
            f"Expected chiral_center_dihedral_angles shape (0,), got {chiral_angles.shape}"
        )

    @require_rf3()
    def test_tracking_consistent_across_disabled_and_enabled(self, rf3_wrapper, structure_1vme):
        """Run both chiral configs from the same seeded noise and compare.

        Step 0 gets identical inputs (same R_L, same original chirals) so
        L2 norms must match exactly. Later steps diverge as the model's
        forward pass differs, but norms should stay within an OOM
        """
        num_steps = 5
        seed = 42

        def featurize_and_run(*, disable_chiral: bool):
            annotated = annotate_structure_for_rf3(
                structure_1vme,
                disable_chiral_features=disable_chiral,
                track_chiral_features=True,
            )
            features = rf3_wrapper.featurize(annotated)

            originals = (
                rf3_wrapper._original_chiral_centers.clone(),
                rf3_wrapper._original_chiral_dihedral_angles.clone(),
            )

            _run_steps(rf3_wrapper, features, num_steps, seed=seed)
            return rf3_wrapper._chiral_grad_stats.copy(), originals

        enabled_stats, enabled_originals = featurize_and_run(disable_chiral=False)
        disabled_stats, disabled_originals = featurize_and_run(disable_chiral=True)

        # Same original chiral features stored in both cases
        assert torch.equal(enabled_originals[0], disabled_originals[0])
        assert torch.equal(enabled_originals[1], disabled_originals[1])
        assert enabled_originals[0].shape[0] > 0, "1VME should have chiral centers"

        assert len(enabled_stats) == len(disabled_stats) == num_steps

        for e, d in zip(enabled_stats, disabled_stats):
            assert e["t"] == d["t"]

        assert enabled_stats[0]["l2_norm"] == pytest.approx(disabled_stats[0]["l2_norm"], rel=1e-4)

        for i, (e, d) in enumerate(zip(enabled_stats, disabled_stats)):
            assert e["l2_norm"] > 0.0
            assert d["l2_norm"] > 0.0

        for i, (e, d) in enumerate(zip(enabled_stats[1:], disabled_stats[1:]), start=1):
            ratio = e["l2_norm"] / d["l2_norm"]
            assert 0.1 < ratio < 10.0, (
                f"Step {i}: ratio={ratio:.2f} "
                f"(enabled={e['l2_norm']:.2f}, disabled={d['l2_norm']:.2f})"
            )

    @require_rf3()
    def test_no_tracking_when_disabled(self, rf3_wrapper, structure_1vme):
        """With track_chiral_features=False, _chiral_grad_stats stays empty."""
        annotated = annotate_structure_for_rf3(
            structure_1vme,
            track_chiral_features=False,
            disable_chiral_features=False,
        )
        features = rf3_wrapper.featurize(annotated)
        _run_steps(rf3_wrapper, features, num_steps=3)

        assert rf3_wrapper._chiral_grad_stats == [], (
            "Stats should be empty when track_chiral_features=False"
        )

    @require_rf3()
    def test_baseline_chiral_features_present(self, rf3_wrapper, structure_1vme):
        """1VME should have non-empty chiral features when
        disable_chiral_features=False."""
        annotated = annotate_structure_for_rf3(
            structure_1vme,
            disable_chiral_features=False,
            track_chiral_features=False,
        )
        features = rf3_wrapper.featurize(annotated)
        cond = features.conditioning

        chiral_centers = cond.features["chiral_centers"]
        assert chiral_centers.shape[0] > 0, "1VME should have chiral centers"
        assert chiral_centers.shape[1] == 4

        chiral_angles = cond.features["chiral_center_dihedral_angles"]
        assert chiral_angles.shape[0] > 0

    @require_rf3()
    def test_featurize_resets_tracking_state(self, rf3_wrapper, structure_1vme):
        """Each call to featurize() should reset _chiral_grad_stats to empty."""

        annotated = annotate_structure_for_rf3(
            structure_1vme,
            track_chiral_features=True,
        )
        features = rf3_wrapper.featurize(annotated)
        state = rf3_wrapper.initialize_from_prior(batch_size=1, features=features)
        t = torch.tensor([1.0], device=rf3_wrapper.device)
        rf3_wrapper.step(state, t, features=features)
        assert len(rf3_wrapper._chiral_grad_stats) > 0

        features = rf3_wrapper.featurize(annotated)
        assert rf3_wrapper._chiral_grad_stats == [], "featurize() should reset _chiral_grad_stats"
