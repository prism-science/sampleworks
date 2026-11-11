"""Standard ESMFold2 conditioning supports native denoising."""

import copy
import os
from collections.abc import Iterator
from typing import Any

import pytest


pytest.importorskip("esm.models.esmfold2.model", reason="ESMFold2 not installed")

import numpy as np
import torch
from esm.models.esmfold2.processor import _seed_context
from huggingface_hub import constants, hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.utils import reset_sessions
from sampleworks.models.esmfold2.structure_processing import (
    ESMFold2InputBuilderBlockLogs,
    structure_to_esmfold2_input,
)
from sampleworks.models.esmfold2.wrapper import ESMFold2Wrapper, ESMFold2WrapperConfig
from sampleworks.utils.atom_reconciler import AtomReconciler


# Loads the various structure dictionary fixtures into the namespace.
pytest_plugins = ["tests.models.esmfold2.structure_fixtures"]


@pytest.fixture(scope="module")
def esmfold2_cpu_wrapper() -> Iterator[ESMFold2Wrapper]:
    """Load an ESMFold2 wrapper from cached files for explicitly enabled tests.

    Set ``SAMPLEWORKS_TEST_ESMFOLD2_WEIGHTS=1`` to enable these tests. They require
    approximately **26.3 GB** of model weights and the 0.42 GB CCD. Both must already be
    in the Hugging Face cache.

    Download the required files from the repository root with offline mode disabled::

        pixi run -e esmfold2-dev hf download biohub/ESMFold2 --include '*.json' '*.safetensors'
        pixi run -e esmfold2-dev hf download biohub/ESMFold2 ccd.pkl

    This fixture uses ``local_files_only=True`` and enables Hugging Face offline
    mode throughout the tests to prevent downloads. Missing files cause a test
    failure. The original offline setting is restored during fixture cleanup.

    Yields
    ------
    ESMFold2Wrapper
        Standard checkpoint on CPU with its language model in FP32.
    """
    if os.environ.get("SAMPLEWORKS_TEST_ESMFOLD2_WEIGHTS") != "1":
        pytest.skip(
            "Set SAMPLEWORKS_TEST_ESMFOLD2_WEIGHTS=1 to run with predownloaded "
            "ESMFold2 weights (tens of GB) and CCD. These tests never download them."
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setattr(constants, "HF_HUB_OFFLINE", True)
        reset_sessions()
        try:
            try:
                checkpoint = snapshot_download("biohub/ESMFold2", local_files_only=True)
                hf_hub_download("biohub/ESMFold2", "ccd.pkl", local_files_only=True)
                wrapper = ESMFold2Wrapper.from_config(
                    ESMFold2WrapperConfig(
                        checkpoint_path=checkpoint, device="cpu", esmc_precision="fp32"
                    )
                )
            except (LocalEntryNotFoundError, FileNotFoundError) as exc:
                pytest.fail(
                    "ESMFold2 weight tests were enabled, but required cached files are missing. "
                    f"Predownload the complete biohub/ESMFold2 checkpoint and CCD. {exc}",
                    pytrace=False,
                )
            yield wrapper
        finally:
            reset_sessions()


@pytest.mark.slow
def test_featurize_matches_native_conditioning(
    esmfold2_cpu_wrapper: ESMFold2Wrapper, short_structure: dict[str, Any]
):
    """Featurization preserves atom identities and masks padding correctly.

    Pooled pair embeddings agree with native ESMFold2 inference.
    """
    # Prepare conditioning, saving the input and random state for comparison.
    original = copy.deepcopy(short_structure)
    rng_state = torch.random.get_rng_state().clone()
    conditioning = esmfold2_cpu_wrapper.featurize(
        short_structure,
        seed=0,
        num_loops=1,
    ).conditioning

    # Featurization leaves the caller's random state and experimental structure intact.
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert conditioning is not None
    assert short_structure["asym_unit"] == original["asym_unit"]
    assert short_structure["chain_info"] == original["chain_info"]
    assert conditioning.true_atom_array == original["asym_unit"]

    # Ensure atom count excludes ESMFold2's padding.
    assert conditioning.num_atoms == len(conditioning.model_atom_array)
    assert conditioning.num_atoms == conditioning.features["atom_attention_mask"].sum().item()
    assert conditioning.z_trunk.shape[:3] == (1, 18, 18)  # (batch, residue i, residue j, ...)
    assert conditioning.s_inputs.shape[:2] == (1, 18)  # (batch, residue i, ...)

    # Conditioning should contain no NaNs or infinities and start with gradients
    # disabled. Autograd may still need to save these tensors to calculate coordinate
    # gradients, and tensors with requires_grad=False allow this, but inference mode
    # tensors do not.
    for tensor in (
        conditioning.z_trunk,
        conditioning.s_inputs,
        conditioning.relative_position_encoding,
    ):
        assert torch.isfinite(tensor).all()
        assert not tensor.requires_grad
        assert not tensor.is_inference()

    # Every experimental atom must have a matching model atom, even if model order differs.
    reconciler = AtomReconciler.from_arrays(
        conditioning.model_atom_array, conditioning.true_atom_array
    )
    assert reconciler.n_common == len(conditioning.true_atom_array)
    np.testing.assert_array_equal(
        conditioning.model_atom_array.atom_name[reconciler.model_indices.numpy()],
        conditioning.true_atom_array.atom_name[reconciler.struct_indices.numpy()],
    )

    # Run native ESMFold2 independently with the same input, seed, and trunk loop
    # setting. We expect the same trunk output. We use one diffusion step to keep the
    # comparison cheap.
    builder = ESMFold2InputBuilderBlockLogs()

    features, _ = builder.prepare_input(structure_to_esmfold2_input(short_structure), seed=0)

    # Encoding must preserve native atom IDs while leaving padded rows entirely zero.
    atom_mask = features["atom_attention_mask"].bool()
    assert atom_mask.any() and (~atom_mask).any(), "Exercise both real atoms and padding"
    torch.testing.assert_close(
        conditioning.features["atom_attention_mask"], features["atom_attention_mask"]
    )
    for name in ("ref_element", "ref_atom_name_chars"):
        encoded = conditioning.features[name]
        assert encoded.is_floating_point()
        assert encoded.shape[:-1] == features[name].shape
        assert torch.all((encoded == 0) | (encoded == 1))
        assert torch.all(encoded[atom_mask].sum(dim=-1) == 1)
        torch.testing.assert_close(encoded[atom_mask].argmax(dim=-1), features[name][atom_mask])
        assert torch.count_nonzero(encoded[~atom_mask]) == 0

    # Real atoms still refer to the same tokens; padded atoms use the zero placeholder.
    torch.testing.assert_close(
        conditioning.features["atom_to_token"][atom_mask], features["atom_to_token"][atom_mask]
    )
    assert torch.count_nonzero(conditioning.features["atom_to_token"][~atom_mask]) == 0

    with _seed_context(0):
        native = esmfold2_cpu_wrapper.model(
            **features,
            num_loops=1,
            num_sampling_steps=1,
            include_embeddings=True,
        )

    # Native inference exposes pair embeddings averaged over the first residue axis.
    # Apply that same average to our cached embeddings and check numerical agreement.
    torch.testing.assert_close(
        conditioning.z_trunk.mean(dim=1),
        native["output_embedding_pair_pooled"],
        rtol=0,
        atol=0,
    )


@pytest.mark.slow
def test_featurize_calls_are_independent(
    esmfold2_cpu_wrapper: ESMFold2Wrapper,
    short_structure: dict[str, Any],
    other_short_structure: dict[str, Any],
):
    """Different inputs get their own conditioning without changing earlier results."""
    first = esmfold2_cpu_wrapper.featurize(short_structure, seed=7, num_loops=1).conditioning
    assert first is not None
    saved = copy.deepcopy(first)

    second = esmfold2_cpu_wrapper.featurize(
        other_short_structure, seed=29, num_loops=1
    ).conditioning
    assert second is not None

    # B has 10 residues rather than 18.
    assert second.z_trunk.shape[:3] == (1, 10, 10)
    assert second.s_inputs.shape[:2] == (1, 10)
    assert second.features["token_attention_mask"].sum().item() == 10
    assert second.num_atoms != first.num_atoms
    assert second.true_atom_array == other_short_structure["asym_unit"]

    # Processing B must leave all of A's conditioning unchanged.
    assert first.num_atoms == saved.num_atoms
    assert first.model_atom_array == saved.model_atom_array
    assert first.true_atom_array == saved.true_atom_array
    torch.testing.assert_close(first.s_inputs, saved.s_inputs, rtol=0, atol=0)
    torch.testing.assert_close(first.z_trunk, saved.z_trunk, rtol=0, atol=0)
    torch.testing.assert_close(
        first.relative_position_encoding, saved.relative_position_encoding, rtol=0, atol=0
    )
    for name in saved.features:
        torch.testing.assert_close(first.features[name], saved.features[name], rtol=0, atol=0)


@pytest.mark.slow
def test_featurize_is_reproducible_with_fixed_seed(
    esmfold2_cpu_wrapper: ESMFold2Wrapper, short_structure: dict[str, Any]
):
    """The same input and seed reproduce conditioning regardless of surrounding random state."""
    first = esmfold2_cpu_wrapper.featurize(short_structure, seed=7, num_loops=1).conditioning
    assert first is not None
    saved = copy.deepcopy(first)

    # Repeat A with its original seed but a different surrounding PyTorch random state.
    # Restore that surrounding state afterward so this test cannot affect other tests.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1234)
        repeated = esmfold2_cpu_wrapper.featurize(short_structure, seed=7, num_loops=1).conditioning
    assert repeated is not None

    # Compare against a saved copy so shared mutable state cannot make this pass accidentally.
    assert repeated.num_atoms == saved.num_atoms
    assert repeated.model_atom_array == saved.model_atom_array
    assert repeated.true_atom_array == saved.true_atom_array
    torch.testing.assert_close(repeated.s_inputs, saved.s_inputs, rtol=0, atol=0)
    torch.testing.assert_close(repeated.z_trunk, saved.z_trunk, rtol=0, atol=0)
    torch.testing.assert_close(
        repeated.relative_position_encoding, saved.relative_position_encoding, rtol=0, atol=0
    )
    assert repeated.features.keys() == saved.features.keys()
    for name in saved.features:
        torch.testing.assert_close(repeated.features[name], saved.features[name], rtol=0, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize("noise_levels", [10.0, torch.tensor([10.0, 1.0])], ids=["scalar", "batch"])
def test_step_matches_native_denoising(
    esmfold2_cpu_wrapper: ESMFold2Wrapper,
    short_structure: dict[str, Any],
    noise_levels: float | torch.Tensor,
):
    """Unpadded predictions match native denoising for each sample and noise level."""
    features = esmfold2_cpu_wrapper.featurize(short_structure, seed=0, num_loops=1)
    conditioning = features.conditioning
    assert conditioning is not None
    native_features = conditioning.features
    levels = torch.as_tensor(noise_levels).reshape(-1)
    coords = torch.randn(len(levels), conditioning.num_atoms, 3)
    original_coords = coords.clone()

    with torch.no_grad():
        prediction = esmfold2_cpu_wrapper.step(coords, noise_levels, features=features)
        assert prediction.shape == coords.shape
        assert torch.isfinite(prediction).all()
        torch.testing.assert_close(coords, original_coords, rtol=0, atol=0)

        # Compare with native denoising using the same batch and explicitly padded inputs.
        padded_coords = torch.zeros(len(levels), native_features["ref_pos"].shape[1], 3)
        padded_coords[:, : conditioning.num_atoms] = coords
        native = esmfold2_cpu_wrapper.model.structure_head.diffusion_module(
            x_noisy=padded_coords,
            t_hat=levels,
            s_inputs=conditioning.s_inputs,
            s_trunk=None,
            z_trunk=conditioning.z_trunk,
            relative_position_encoding=conditioning.relative_position_encoding,
            ref_pos=native_features["ref_pos"],
            ref_charge=native_features["ref_charge"],
            ref_mask=native_features["atom_attention_mask"],
            ref_element=native_features["ref_element"],
            ref_atom_name_chars=native_features["ref_atom_name_chars"],
            ref_space_uid=native_features["ref_space_uid"],
            tok_idx=native_features["atom_to_token"],
            asym_id=native_features["asym_id"],
            residue_index=native_features["residue_index"],
            entity_id=native_features["entity_id"],
            token_index=native_features["token_index"],
            sym_id=native_features["sym_id"],
            token_attention_mask=native_features["token_attention_mask"],
            num_diffusion_samples=len(levels),
        )["x_denoised"]
        assert native is not None
        torch.testing.assert_close(prediction, native[:, : conditioning.num_atoms], rtol=0, atol=0)


@pytest.mark.slow
def test_step_allows_coordinate_gradients_with_reused_conditioning(
    esmfold2_cpu_wrapper: ESMFold2Wrapper, short_structure: dict[str, Any]
):
    """Reusing conditioning allows successive backward passes through predicted coordinates."""
    features = esmfold2_cpu_wrapper.featurize(short_structure, seed=0, num_loops=1)
    conditioning = features.conditioning
    assert conditioning is not None

    # Each call must build a fresh graph while using the same cached trunk outputs.
    for noise_level in (10.0, 1.0):
        coords = torch.randn(2, conditioning.num_atoms, 3, requires_grad=True)
        prediction = esmfold2_cpu_wrapper.step(coords, noise_level, features=features)
        assert prediction.shape == coords.shape
        assert torch.isfinite(prediction).all()

        # A scalar objective must have finite, nonzero gradients for both samples.
        (gradient,) = torch.autograd.grad(prediction.square().sum(), coords)
        assert torch.isfinite(gradient).all()
        assert (gradient.abs().sum(dim=(1, 2)) > 0).all()
