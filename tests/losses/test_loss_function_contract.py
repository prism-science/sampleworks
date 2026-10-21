"""Loss-agnostic contract tests, parametrized over every loss function.

These tests exercise behavior that any `LossFunctionProtocol` implementation must
satisfy (see `sampleworks.core.losses.protocol`). They were extracted from
`test_real_space_density_loss.py` (whose header asked for exactly this generalization)
so that new loss functions inherit the shared contract by adding a single bundle entry
to `_LOSS_BUNDLES`.

`TestLossCorrelation` is where the package's sign convention is enforced: losses are
minimized, so moving away from the target must not lower the value (see
`LossFunctionProtocol`). A term written as a score to maximize fails here.

Only *loss-agnostic* checks live here:
- Absolute loss thresholds ARE shared, but the value is per-loss (see `_LOSS_THRESHOLDS`):
  RealSpace's loss is MSE on normalized density (sigma units), so a wrong/random model
  scores ~O(1) while the true model scores ~0. Correlation is still checked relatively
  (true < perturbed < random, monotonic) on top of the absolute bar.
- The batch=1 occupancy-gradient check IS shared (every loss consumes the single
  conformer's occupancy); see `test_gradients_wrt_occupancies` below. Tests that change
  atom count / topology (single-atom) or rely on `precompute_unique_combinations` /
  `structure_to_loss_input` stay in the loss-specific files, since not every loss
  is independent of topology information or implements those methods.

Each loss runs against its own self-consistent data bundle (see `loss_case`): the
real_space case uses the recentered carved 1vme cif + `.ccp4` map; diffuse_bragg scores
Bragg alone here, since its diffuse term is an ensemble property that is identically
zero for the single conformer the contract scores.
"""

from dataclasses import dataclass

import pytest
import torch
from sampleworks.core.losses.protocol import LossFunctionProtocol

from tests.losses.loss_input_helpers import build_loss_input_tensors_without_coords


# real_space and structure_factor exercise CUDA-targeted code on the `device` fixture
# (try_gpu), so those params are gpu-marked; diffuse_bragg runs on CPU in ~15 s for the
# whole module and is left unmarked so `tests` covers it. Deliberately NOT `slow`: measured
# warm per-test time is ~1s at most (the SFC gradient-descent loop; the rest are sub-0.5s).
# The fixed cost is a one-time import, the session-scoped synthetic target generation, and
# session-scoped loss construction, paid once per pytest invocation and not skippable by
# `slow`-marking these tests.


@dataclass(frozen=True)
class LossCase:
    """A loss function bundled with a standard set of per-atom inputs for one structure."""

    name: str
    loss_function: LossFunctionProtocol
    coords: torch.Tensor  # [N, 3]
    elements: torch.Tensor  # [N]
    b_factors: torch.Tensor  # [N]
    occupancies: torch.Tensor  # [N]

    def batch(self, n: int = 1, coords: torch.Tensor | None = None) -> dict:
        """Build a [B, ...] kwargs dict by expanding the stored inputs over a batch."""
        c = self.coords if coords is None else coords
        return dict(
            coordinates=c.unsqueeze(0).expand(n, -1, -1),
            elements=self.elements.unsqueeze(0).expand(n, -1),
            b_factors=self.b_factors.unsqueeze(0).expand(n, -1),
            occupancies=self.occupancies.unsqueeze(0).expand(n, -1),
        )


# Per-loss bundles: each param resolves its OWN coordinates/atom_array/loss so the
# inputs are self-consistent with that loss's target. real_space uses the recentered
# carved 1vme (matches its .ccp4 map frame); structure_factor uses the crystal-frame
# chain-A model the synthetic MTZ was generated from (recentering corrupts SF symmetry
# mates). New losses register here alongside a matching entry in `_LOSS_THRESHOLDS`.
_LOSS_BUNDLES = {
    "real_space": ("test_coordinates_1vme", "loss_function_1vme"),
    "structure_factor": ("test_coordinates_1vme_sf", "loss_function_1vme_sf"),
    "diffuse_bragg": ("test_coordinates_1vme_diffuse_bragg", "loss_function_1vme_diffuse_bragg"),
}
_GPU_BUNDLES = {"real_space", "structure_factor"}

# Absolute loss bar for the TRUE structure, per loss. RealSpace's loss is MSE on
# normalized density (sigma units): for our 1VME case the true structure scores ~0.0018,
# a 0.5 A perturbation ~0.034, and random coordinates ~O(1). The 0.01 bar sits ~5x above
# the true loss (robust to device/precision variance) yet ~3x below the 0.5 A-perturbed
# loss, so it comfortably passes the truth while meaningfully failing a wrong structure.
# SFC (normalize_amplitude=True -> normalized E-values) measured on the synthetic MTZ:
# true ~2e-14, a 0.5 A perturbation ~0.34, and random ~0.44, so the 0.1 bar sits well
# above numerical zero and ~3x below the smallest perturbation signal. diffuse_bragg
# (normalize=True, fit_scale=True -> 1 - cos^2 on |F|) measured on the single-configuration
# target: true ~1e-14, a 0.5 A perturbation ~0.3, so 1e-3 sits far from both.
_LOSS_THRESHOLDS = {
    "real_space": 0.01,
    "structure_factor": 0.1,
    "diffuse_bragg": 1e-3,
}


@pytest.fixture(
    params=[
        pytest.param(name, marks=[pytest.mark.gpu] if name in _GPU_BUNDLES else [])
        for name in _LOSS_BUNDLES
    ]
)
def loss_case(request, device: torch.device) -> LossCase:
    """Resolve one loss's self-consistent (loss, coords, per-atom inputs) bundle."""
    coords_fixture, loss_fixture = _LOSS_BUNDLES[request.param]
    coords, atom_array = request.getfixturevalue(coords_fixture)
    elements, b_factors, occupancies = build_loss_input_tensors_without_coords(atom_array, device)

    loss_function = request.getfixturevalue(loss_fixture)
    return LossCase(request.param, loss_function, coords, elements, b_factors, occupancies)


class TestLossFunctionInterface:
    """Output type/shape contract any loss must satisfy."""

    def test_loss_function_conforms_to_protocol(self, loss_case):
        """The loss satisfies the LossFunctionProtocol interface."""
        assert isinstance(loss_case.loss_function, LossFunctionProtocol)

    def test_loss_function_call_shapes(self, loss_case):
        """Single [N,3] and batched [B,N,3] inputs both return a scalar."""
        for n in (1, 3):
            loss = loss_case.loss_function(**loss_case.batch(n))
            assert loss.shape == torch.Size([])
            assert loss.ndim == 0

    def test_loss_function_output_is_scalar(self, loss_case):
        """A single structure returns a non-negative scalar loss."""
        loss = loss_case.loss_function(**loss_case.batch(1))
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert loss.item() >= 0.0

    def test_loss_function_deterministic(self, loss_case):
        """Same inputs give the same output."""
        loss1 = loss_case.loss_function(**loss_case.batch(1))
        loss2 = loss_case.loss_function(**loss_case.batch(1))
        torch.testing.assert_close(loss1, loss2)


class TestLossCorrelation:
    """Loss must rank structures by correctness (absolute bar + relative ordering)."""

    def test_perfect_structure_has_low_loss(self, loss_case):
        """The true structure clears the per-loss absolute bar (normalized scales)."""
        loss = loss_case.loss_function(**loss_case.batch(1)).item()
        assert loss < _LOSS_THRESHOLDS[loss_case.name]

    def test_perturbed_structure_has_higher_loss(self, loss_case):
        """Perturbed coordinates give higher loss than the true structure."""
        loss_true = loss_case.loss_function(**loss_case.batch(1)).item()

        torch.manual_seed(42)
        coords_perturbed = loss_case.coords + torch.randn_like(loss_case.coords) * 0.5
        loss_perturbed = loss_case.loss_function(
            **loss_case.batch(1, coords=coords_perturbed)
        ).item()

        assert loss_perturbed > loss_true

    def test_random_structure_has_higher_loss(self, loss_case):
        """Random coordinates score worse than the true structure.

        The original RealSpace test required random > 2x true; that 2x margin is a
        RealSpace-scale assumption, so the shared contract only asserts the robust
        ordering (random > true). Tighten per-loss if desired.
        """
        loss_true = loss_case.loss_function(**loss_case.batch(1)).item()

        torch.manual_seed(42)
        coords_random = torch.randn_like(loss_case.coords) * 10.0
        loss_random = loss_case.loss_function(**loss_case.batch(1, coords=coords_random)).item()

        assert loss_random > loss_true

    def test_loss_monotonic_with_perturbation(self, loss_case):
        """Loss increases (non-strictly) with per-atom perturbation magnitude."""
        torch.manual_seed(42)
        # Per-atom unit direction (normalized over dim=-1, NOT the whole tensor like before).
        direction = torch.randn_like(loss_case.coords)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        losses = []
        for scale in [0.0, 0.1, 0.25, 0.5, 1.0]:
            coords_pert = loss_case.coords + direction * scale
            losses.append(loss_case.loss_function(**loss_case.batch(1, coords=coords_pert)).item())

        for i in range(len(losses) - 1):
            assert losses[i + 1] >= losses[i] - 1e-6


class TestLossGradientFlow:
    """Gradients must flow w.r.t. coordinates for guided sampling / optimization."""

    def test_gradients_wrt_coordinates(self, loss_case):
        """Gradients flow through coordinates and are finite, non-zero, and bounded."""
        coords_opt = loss_case.coords.clone().unsqueeze(0).requires_grad_(True)
        loss = loss_case.loss_function(
            coordinates=coords_opt,
            elements=loss_case.elements.unsqueeze(0),
            b_factors=loss_case.b_factors.unsqueeze(0),
            occupancies=loss_case.occupancies.unsqueeze(0),
        )
        loss.backward()

        assert coords_opt.grad is not None
        assert torch.all(torch.isfinite(coords_opt.grad))
        assert torch.any(coords_opt.grad != 0)
        # Magnitude sanity: non-zero but not exploding (NaN/inf already ruled out above).
        assert 0 < coords_opt.grad.norm().item() < 1e6

    def test_gradients_wrt_occupancies(self, loss_case):
        """Occupancy must be differentiable (batch=1).

        At batch=1 every loss consumes the single conformer's occupancy. Per-conformer
        (batch>1) occupancy is NOT part of the shared contract — a loss may lack a batch
        occupancy axis — so that asymmetry lives in the loss-specific tests. Mirrors
        test_gradients_wrt_coordinates: finiteness only, no nonzero assertion (the gradient
        can vanish near the synthetic-data minimum).
        """
        occupancies_opt = loss_case.occupancies.clone().unsqueeze(0).requires_grad_(True)
        loss = loss_case.loss_function(
            coordinates=loss_case.coords.unsqueeze(0),
            elements=loss_case.elements.unsqueeze(0),
            b_factors=loss_case.b_factors.unsqueeze(0),
            occupancies=occupancies_opt,
        )
        loss.backward()

        assert occupancies_opt.grad is not None
        assert torch.all(torch.isfinite(occupancies_opt.grad))

    def test_gradient_descent_improves_loss(self, loss_case):
        """Gradient descent on coordinates reduces the loss."""
        torch.manual_seed(42)
        perturbation = torch.randn_like(loss_case.coords) * 0.5
        coords_opt = (loss_case.coords + perturbation).unsqueeze(0).requires_grad_(True)
        optimizer = torch.optim.Adam([coords_opt], lr=0.01)

        def loss_fn():
            return loss_case.loss_function(
                coordinates=coords_opt,
                elements=loss_case.elements.unsqueeze(0),
                b_factors=loss_case.b_factors.unsqueeze(0),
                occupancies=loss_case.occupancies.unsqueeze(0),
            )

        loss_initial = loss_fn().item()
        for _ in range(10):
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            optimizer.step()
        loss_final = loss_fn().item()

        assert loss_final < loss_initial


# Batch sizes span a single conformer up to a 20-member ensemble. Even batch=20 is
# sub-second on the GPU losses (measured ~0.5s), so none is `slow`.
@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(1, id="single"),
        pytest.param(3, id="ensemble-3"),
        pytest.param(5, id="ensemble-5"),
        pytest.param(20, id="ensemble-20"),
    ],
)
class TestLossBatchHandling:
    """The batch dim is an ensemble: an occupancy-weighted mixture of every conformer.

    The pipeline gives each of ``B`` conformers occupancy ``1/B`` and every loss sums the
    conformers' contributions (density, or complex structure factors). The tests use
    perturbed coordinates rather than the true structure, keeping the loss large enough for
    reliable relative comparisons.
    """

    def test_identical_conformers_at_split_occupancy_match_one_conformer(
        self, loss_case, batch_size
    ):
        """``B`` copies of a conformer at occupancy/B score like that conformer alone.

        Fails if the batch is averaged instead of summed, or only one conformer is scored.
        """
        torch.manual_seed(42)
        coords = loss_case.coords + torch.randn_like(loss_case.coords) * 0.5  # [N, 3]

        loss_single = loss_case.loss_function(**loss_case.batch(1, coords=coords))

        ensemble = loss_case.batch(batch_size, coords=coords)
        ensemble["occupancies"] = ensemble["occupancies"] / batch_size
        loss_ensemble = loss_case.loss_function(**ensemble)

        assert loss_ensemble.shape == torch.Size([])
        assert loss_single.item() > _LOSS_THRESHOLDS[loss_case.name]
        torch.testing.assert_close(loss_ensemble, loss_single, rtol=1e-4, atol=0.0)

    def test_moving_only_the_last_conformer_changes_the_loss(self, loss_case, batch_size):
        """Displacing the last conformer away from the truth raises the ensemble's loss.

        Fails if any conformer is dropped from the sum (e.g. only ``[0]`` scored), since
        the last conformer would then never be seen.
        """
        torch.manual_seed(42)
        displaced = loss_case.coords + torch.randn_like(loss_case.coords) * 1.0  # [N, 3]

        true_ensemble = loss_case.coords.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, N, 3]
        moved_ensemble = true_ensemble.clone()
        moved_ensemble[-1] = displaced

        def score(ensemble_coords: torch.Tensor) -> torch.Tensor:
            inputs = loss_case.batch(batch_size)
            inputs["coordinates"] = ensemble_coords
            inputs["occupancies"] = inputs["occupancies"] / batch_size
            return loss_case.loss_function(**inputs)

        assert score(moved_ensemble).item() > score(true_ensemble).item()


class TestLossEdgeCases:
    """Edge cases that keep the (prepared) atom topology intact."""

    def test_numerical_stability(self, loss_case):
        """Extreme coordinate values still produce a finite loss."""
        coords_far = loss_case.coords + torch.randn_like(loss_case.coords) * 1e9
        loss = loss_case.loss_function(**loss_case.batch(1, coords=coords_far))
        assert torch.isfinite(loss)
