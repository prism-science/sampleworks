"""Mock implementations of model wrapper protocols for testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from biotite.structure import AtomArray
from jaxtyping import Float
from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.framework_utils import Array
from sampleworks.utils.torch_utils import gaussian_prior
from torch import Tensor


@dataclass(frozen=True)
class MockConditioning:
    """Simple conditioning for tests.

    Parameters
    ----------
    sequence_length
        Number of residues represented in the sequence.
    num_atoms
        Atom count in the model representation.
    model_atom_array
        Optional model-space atom template used to build an
        :class:`sampleworks.utils.atom_reconciler.AtomReconciler`.
    """

    sequence_length: int
    num_atoms: int
    model_atom_array: AtomArray | None = None


@dataclass(frozen=True)
class MismatchCase:
    """Declarative atom mismatch scenario for integration tests.

    Parameters
    ----------
    id
        Stable identifier used in pytest parametrization IDs.
    description
        Human-readable scenario description.
    predicted_atom_array
        Atom array in model space.
    input_atom_array
        Atom array in structure/input space.
    expected_n_common
        Expected common-atom count after reconciliation.
    expected_has_mismatch
        Expected mismatch flag from :class:`sampleworks.utils.atom_reconciler.AtomReconciler`.
    """

    id: str
    description: str
    predicted_atom_array: AtomArray
    input_atom_array: AtomArray
    expected_n_common: int
    expected_has_mismatch: bool

    @property
    def n_model(self) -> int:
        """int: Number of model-space atoms."""
        return len(self.predicted_atom_array)

    @property
    def n_struct(self) -> int:
        """int: Number of structure-space atoms."""
        return len(self.input_atom_array)

    def clone(self) -> MismatchCase:
        """Return a copy with independent atom arrays."""
        return MismatchCase(
            id=self.id,
            description=self.description,
            predicted_atom_array=self.predicted_atom_array.copy(),
            input_atom_array=self.input_atom_array.copy(),
            expected_n_common=self.expected_n_common,
            expected_has_mismatch=self.expected_has_mismatch,
        )


class MockFlowModelWrapper:
    """Mock FlowModelWrapper that satisfies the protocol without a real model.

    This mock provides deterministic behavior for testing:
    - featurize() returns a GenerativeModelInput with conditioning data
    - step() returns a slightly denoised version of the input
    - initialize_from_prior() returns random noise

    Parameters
    ----------
    num_atoms
        Number of atoms for default structure.
    device
        PyTorch device for tensors.
    target
        Optional target tensor for convergence testing. When provided, step()
        outputs this target instead of zero.
    """

    def __init__(
        self,
        num_atoms: int = 100,
        device: torch.device | None = None,
        target: Tensor | None = None,
    ):
        self.num_atoms = num_atoms
        self.device = device or torch.device("cpu")
        self.target = target
        self.prior_initialization_features: list[GenerativeModelInput[MockConditioning] | None] = []

    def featurize(self, structure: dict, **kwargs: Any) -> GenerativeModelInput[MockConditioning]:
        """Featurize a structure dict into model inputs."""
        conditioning = MockConditioning(sequence_length=10, num_atoms=self.num_atoms)
        return GenerativeModelInput(conditioning=conditioning)

    def step(
        self,
        x_t: Float[Tensor, "*batch atoms 3"],
        t: Float[Array, "*batch"],
        *,
        features: GenerativeModelInput[MockConditioning] | None = None,
    ) -> Float[Tensor, "*batch atoms 3"]:
        """Perform one denoising step.

        Returns the denoised prediction that is equal to target (or zero if no target).
        """
        if features is None:
            raise ValueError("features required for step()")

        if self.target is not None:
            target = self.target.to(x_t.device)
            if target.shape != x_t.shape:
                target = target.expand_as(x_t)
            return target

        return torch.zeros_like(x_t)

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[MockConditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Initialize coordinates from prior distribution (random noise)."""
        self.prior_initialization_features.append(features)
        if features is None and shape is None:
            raise ValueError("Either features or shape must be provided")
        if shape is not None:
            return torch.randn(batch_size, *shape, device=self.device)
        if features is None or features.conditioning is None:
            raise ValueError("features with conditioning required")
        return torch.randn(batch_size, features.conditioning.num_atoms, 3, device=self.device)


class MismatchCaseWrapper:
    """Wrapper driven by a :class:`MismatchCase`.

    Parameters
    ----------
    case
        Declarative mismatch case providing model and structure atom arrays.
    device
        Device used for tensor allocations.
    """

    def __init__(
        self,
        case: MismatchCase,
        device: torch.device | None = None,
    ):
        self.case = case
        self.device = device or torch.device("cpu")
        self.n_model = len(case.predicted_atom_array)

    def featurize(self, structure: dict, **kwargs: Any) -> GenerativeModelInput[MockConditioning]:
        """Return conditioning with case model atom array for reconciliation."""
        conditioning = MockConditioning(
            sequence_length=10,
            num_atoms=self.n_model,
            model_atom_array=self.case.predicted_atom_array,
        )
        return GenerativeModelInput(conditioning=conditioning)

    def step(
        self,
        x_t: Float[Tensor, "*batch atoms 3"],
        t: Float[Array, "*batch"],
        *,
        features: GenerativeModelInput[MockConditioning] | None = None,
    ) -> Float[Tensor, "*batch atoms 3"]:
        """Return a non-degenerate denoised prediction in model space."""
        if features is None:
            raise ValueError("features required")
        return x_t * 0.1

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[MockConditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Sample prior coordinates with the case's model atom count."""
        if shape is not None:
            return torch.randn(batch_size, *shape, device=self.device)
        return torch.randn(batch_size, self.n_model, 3, device=self.device)


class GaussianDenoiserWrapper:
    r"""Exact FlowModelWrapper for the Gaussian data distribution N(0, sigma_data^2 I).

    For this target and the VE noising ``x_t = x_0 + sigma * eps``, the posterior mean is
    known in closed form, :math:`\hat{x}_0 = \sigma_\text{data}^2 /
    (\sigma_\text{data}^2 + \sigma^2) \cdot x_t`, so the statistics a sampler should produce
    can be predicted analytically.

    Parameters
    ----------
    sigma_data
        Standard deviation of every coordinate of the target distribution.
    num_atoms
        Number of atoms per sample.
    device
        PyTorch device for tensors.
    """

    def __init__(self, sigma_data: float, num_atoms: int, device: torch.device | None = None):
        self.sigma_data = sigma_data
        self.num_atoms = num_atoms
        self.device = device or torch.device("cpu")

    def featurize(self, structure: dict, **kwargs: Any) -> GenerativeModelInput[MockConditioning]:
        """Return features carrying only the atom count; the target ignores the structure."""
        conditioning = MockConditioning(sequence_length=0, num_atoms=self.num_atoms)
        return GenerativeModelInput(conditioning=conditioning)

    def step(
        self,
        x_t: Float[Tensor, "*batch atoms 3"],
        t: Float[Array, "*batch"],
        *,
        features: GenerativeModelInput[MockConditioning] | None = None,
    ) -> Float[Tensor, "*batch atoms 3"]:
        """Return the optimal denoised prediction for noise level ``t``."""
        noise_variance = torch.as_tensor(t, device=x_t.device) ** 2
        return self.sigma_data**2 / (self.sigma_data**2 + noise_variance) * x_t

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[MockConditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Draw standard Gaussian coordinates, as the real wrappers do."""
        return gaussian_prior(batch_size, self.num_atoms, self.device, shape=shape)
