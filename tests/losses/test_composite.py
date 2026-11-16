"""Tests for weighted loss combination."""

from pathlib import Path

import pytest
import torch
from sampleworks.core.losses.composite import CompositeLoss
from sampleworks.core.losses.config import build_loss, LossConfig, LossEntry
from sampleworks.core.losses.protocol import LossFunctionProtocol, LossInputs
from sampleworks.core.losses.real_space_density import RealSpaceLossFunction
from sampleworks.core.losses.registry import LossBuildContext
from sampleworks.utils.guidance_constants import Losses

from tests.losses.loss_input_helpers import build_loss_input_tensors_without_coords


class QuadraticLoss:
    """Loss = 0.5 * scale * ||coords||^2, so the gradient is scale * coords."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(
        self,
        coordinates: torch.Tensor,
        elements: torch.Tensor | None = None,
        b_factors: torch.Tensor | None = None,
        occupancies: torch.Tensor | None = None,
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return 0.5 * self.scale * (coordinates**2).sum()


class PreparableQuadraticLoss(QuadraticLoss):
    """A quadratic loss that also binds to the loss inputs."""

    def __init__(self, scale: float = 1.0):
        super().__init__(scale)
        self.prepared_atom_counts: list[int] = []

    def prepare(self, loss_inputs: LossInputs, *, device="cpu") -> None:
        self.prepared_atom_counts.append(loss_inputs.b_factors.shape[-1])


def coords(value: float = 2.0) -> torch.Tensor:
    return torch.full((1, 3, 3), value)


def per_atom(n_atoms: int = 3) -> dict:
    return dict(
        elements=torch.ones(1, n_atoms, dtype=torch.long),
        b_factors=torch.full((1, n_atoms), 20.0),
        occupancies=torch.ones(1, n_atoms),
    )


def density_options(density_path: Path, loss_order: int = 1) -> dict:
    return {"density": str(density_path), "resolution": 1.8, "loss_order": loss_order}


def batched_inputs(test_coordinates_1vme, device) -> dict:
    """The 1vme occupied-atom coordinates and their per-atom inputs, as a batch of one."""
    coords_1vme, atom_array = test_coordinates_1vme
    elements, b_factors, occupancies = build_loss_input_tensors_without_coords(atom_array, device)
    return dict(
        coordinates=coords_1vme.unsqueeze(0),
        elements=elements.unsqueeze(0),
        b_factors=b_factors.unsqueeze(0),
        occupancies=occupancies.unsqueeze(0),
    )


class TestCompositeValue:
    def test_is_a_loss_function(self):
        assert isinstance(CompositeLoss([QuadraticLoss()]), LossFunctionProtocol)

    def test_value_is_the_weighted_sum_of_its_terms(self):
        """The composite value is each term's value scaled by its weight, then summed.

        The weights here sum to one, so this checks the weighted sum itself, not the
        normalization covered by ``test_a_single_term_keeps_its_weight``.
        """
        terms = [QuadraticLoss(1.0), QuadraticLoss(3.0)]
        composite = CompositeLoss(terms, [0.25, 0.75])
        x = coords()

        combined = composite(x, **per_atom())

        expected = 0.25 * terms[0](x) + 0.75 * terms[1](x)
        assert torch.isclose(combined, expected)

    def test_a_single_term_keeps_its_weight(self):
        """Weights are used as given, not renormalized to sum to one."""
        term = QuadraticLoss(2.0)
        composite = CompositeLoss([term], [0.5])
        x = coords()

        assert torch.equal(composite(x, **per_atom()), 0.5 * term(x))

    def test_default_weights_average_the_terms(self):
        composite = CompositeLoss([QuadraticLoss(1.0), QuadraticLoss(3.0)])

        assert torch.isclose(composite(coords(), **per_atom()), QuadraticLoss(2.0)(coords()))

    def test_gradient_is_the_weighted_sum_of_gradients(self):
        composite = CompositeLoss([QuadraticLoss(1.0), QuadraticLoss(3.0)], [0.5, 0.5])
        x = coords().requires_grad_(True)

        composite(x, **per_atom()).backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, 2.0 * coords())

    def test_a_single_term_is_returned_unweighted(self):
        """A one-term composite must not quietly halve the gradient."""
        composite = CompositeLoss([QuadraticLoss(2.0)])

        assert torch.isclose(composite(coords(), **per_atom()), QuadraticLoss(2.0)(coords()))


class TestCompositeValidation:
    def test_no_losses_is_rejected(self):
        with pytest.raises(ValueError, match="needs at least one loss function"):
            CompositeLoss([])

    def test_mismatched_weight_count_is_rejected(self):
        with pytest.raises(ValueError, match="one to one"):
            CompositeLoss([QuadraticLoss()], [0.5, 0.5])

    def test_negative_weight_is_rejected(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            CompositeLoss([QuadraticLoss()], [-1.0])


def test_prepare_is_forwarded_only_to_terms_that_need_it():
    n_atoms = 6
    loss_inputs = LossInputs(
        elements=torch.zeros(1, n_atoms, dtype=torch.long),
        b_factors=torch.full((1, n_atoms), 20.0),
        occupancies=torch.ones(1, n_atoms),
        input_coords=torch.zeros(1, n_atoms, 3),
    )
    preparable = PreparableQuadraticLoss()
    composite = CompositeLoss([QuadraticLoss(), preparable])

    composite.prepare(loss_inputs, device="cpu")

    assert preparable.prepared_atom_counts == [n_atoms]


class TestBuildLoss:
    """build_loss turns a configuration into the loss a run scores against."""

    def test_a_single_loss_at_full_weight_is_not_wrapped(
        self, density_path_1vme, device, structure_1vme_density
    ):
        config = LossConfig.single(Losses.REAL_SPACE_DENSITY, **density_options(density_path_1vme))

        loss = build_loss(config, LossBuildContext(structure=structure_1vme_density, device=device))

        assert isinstance(loss, RealSpaceLossFunction)

    @pytest.mark.gpu
    def test_a_single_loss_scores_at_its_configured_weight(
        self, density_path_1vme, device, structure_1vme_density, test_coordinates_1vme
    ):
        context = LossBuildContext(structure=structure_1vme_density, device=device)
        full_weight = build_loss(
            LossConfig.single(Losses.REAL_SPACE_DENSITY, **density_options(density_path_1vme)),
            context,
        )
        half_weight = build_loss(
            LossConfig(
                entries=(
                    LossEntry(
                        loss=Losses.REAL_SPACE_DENSITY,
                        weight=0.5,
                        options=density_options(density_path_1vme),
                    ),
                )
            ),
            context,
        )
        inputs = batched_inputs(test_coordinates_1vme, device)

        assert torch.equal(half_weight(**inputs), 0.5 * full_weight(**inputs))

    @pytest.mark.gpu
    def test_several_losses_are_combined_with_their_weights(
        self, density_path_1vme, device, structure_1vme_density, test_coordinates_1vme
    ):
        context = LossBuildContext(structure=structure_1vme_density, device=device)
        first_order = build_loss(
            LossConfig.single(Losses.REAL_SPACE_DENSITY, **density_options(density_path_1vme, 1)),
            context,
        )
        second_order = build_loss(
            LossConfig.single(Losses.REAL_SPACE_DENSITY, **density_options(density_path_1vme, 2)),
            context,
        )
        combined = CompositeLoss([first_order, second_order], weights=[0.25, 0.75])
        inputs = batched_inputs(test_coordinates_1vme, device)

        assert combined.weights == [0.25, 0.75]
        assert torch.isclose(
            combined(**inputs), 0.25 * first_order(**inputs) + 0.75 * second_order(**inputs)
        )
