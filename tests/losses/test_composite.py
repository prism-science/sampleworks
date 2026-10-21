"""Tests for weighted loss combination."""

import pytest
import torch
from sampleworks.core.losses.composite import CompositeLoss
from sampleworks.core.losses.config import build_loss, LossConfig
from sampleworks.core.losses.protocol import LossFunctionProtocol, LossInputs
from sampleworks.core.losses.registry import LossBuildContext
from sampleworks.utils.guidance_constants import Losses


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


class TestCompositeValue:
    def test_is_a_loss_function(self):
        assert isinstance(CompositeLoss([QuadraticLoss()]), LossFunctionProtocol)

    def test_value_is_the_weighted_sum_of_its_terms(self):
        terms = [QuadraticLoss(1.0), QuadraticLoss(3.0)]
        composite = CompositeLoss(terms, [0.25, 0.75])

        combined = composite(coords(), **per_atom())

        expected = 0.25 * terms[0](coords()) + 0.75 * terms[1](coords())
        assert torch.isclose(combined, expected)

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

    def test_a_single_loss_at_full_weight_is_not_wrapped(self, monkeypatch):
        monkeypatch.setattr(
            "sampleworks.core.losses.config.build_single_loss",
            lambda loss, options, context: QuadraticLoss(),
        )
        config = LossConfig.single(Losses.REAL_SPACE_DENSITY, density="m.ccp4", resolution=1.8)

        loss = build_loss(config, LossBuildContext(structure={}))

        assert isinstance(loss, QuadraticLoss)

    def test_several_losses_are_combined_with_their_weights(self, monkeypatch):
        scales = {Losses.REAL_SPACE_DENSITY: 1.0, Losses.STRUCTURE_FACTOR: 3.0}
        monkeypatch.setattr(
            "sampleworks.core.losses.config.build_single_loss",
            lambda loss, options, context: QuadraticLoss(scales[loss]),
        )
        config = LossConfig.from_mapping(
            {
                "real_space_density": {"weight": 0.25},
                "structure_factor": {"weight": 0.75},
            }
        )

        loss = build_loss(config, LossBuildContext(structure={}))

        assert isinstance(loss, CompositeLoss)
        assert loss.weights == [0.25, 0.75]
        assert torch.isclose(loss(coords(), **per_atom()), QuadraticLoss(2.5)(coords()))
