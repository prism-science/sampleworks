"""Step scaler implementations for per-step guidance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch import Tensor


if TYPE_CHECKING:
    from sampleworks.core.samplers.protocol import StepParams
    from sampleworks.models.protocol import FlowModelWrapper


def _raise_value_error_if_atom_count_mismatch(
    state: Float[Tensor, "*batch atoms 3"],
    context: StepParams,
) -> None:
    """Raise ValueError if state and loss_inputs have different atom counts."""
    assert context.loss_inputs is not None
    n_state_atoms = state.shape[-2]
    n_loss_atoms = context.loss_inputs.elements.shape[-1]
    if n_state_atoms != n_loss_atoms:
        raise ValueError(
            f"State atom count ({n_state_atoms}) != loss_inputs atom count ({n_loss_atoms}). "
            "Ensure loss_inputs are built with the correct number of atoms when a mismatch "
            "exists."
        )


class NoScalingScaler:
    """No-op scaler that returns zero guidance."""

    def scale(
        self,
        state: Float[Tensor, "*batch atoms 3"],
        context: StepParams,
        *,
        model: FlowModelWrapper | None = None,
    ) -> tuple[Float[Tensor, "*batch atoms 3"], Float[Tensor, " batch"]]:
        zeros = torch.zeros_like(state)
        loss = torch.zeros(state.shape[0], device=state.device)
        return zeros, loss

    def guidance_strength(self, context: StepParams) -> Float[Tensor, " batch"]:
        return torch.zeros_like(torch.as_tensor(context.t_effective))


class DataSpaceDPSScaler:
    r"""Step scaler that operates in the data space.

    Computes gradients only on the denoised prediction :math:`\hat{x}_\theta`, avoiding backprop
    through the model. This is faster but may be less accurate than full backprop.
    """

    def __init__(self, step_size: float = 0.1, gradient_normalization: bool = False):
        self.step_size = step_size
        self.gradient_normalization = gradient_normalization

    def scale(
        self,
        state: Float[Tensor, "*batch atoms 3"],
        context: StepParams,
        *,
        model: FlowModelWrapper | None = None,
    ) -> tuple[Float[Tensor, "*batch atoms 3"], Float[Tensor, " batch"]]:
        if context.loss_function is None or context.loss_inputs is None:
            raise ValueError(
                "StepParams missing loss_function/loss_inputs. "
                "Use context.with_loss() before calling scale()."
            )

        _raise_value_error_if_atom_count_mismatch(state, context)

        x0_for_grad = state.detach().requires_grad_(True)
        loss = context.loss_function(
            coordinates=x0_for_grad,
            elements=context.loss_inputs.elements,
            b_factors=context.loss_inputs.b_factors,
            occupancies=context.loss_inputs.occupancies,
        )
        loss.backward()

        grad = x0_for_grad.grad
        if grad is None:
            raise RuntimeError(f"Gradient computation failed in {self.__class__.__name__}")

        if self.gradient_normalization:
            grad_norm = grad.norm(dim=(-1, -2), keepdim=True)
            grad = grad / (grad_norm + 1e-8)

        return grad, loss.detach()

    def guidance_strength(self, context: StepParams) -> Float[Tensor, " batch"]:
        return torch.ones_like(torch.as_tensor(context.t_effective)) * self.step_size


class NoiseSpaceDPSScaler:
    """Step scaler with full backprop through model, operating in the noise space.

    Computes gradients through the full model forward pass by taking gradients
    with respect to the noisy input x_t. More accurate but slower than data space.

    NOTE: For this scaler to work, the sampler must:
    1. Set requires_grad=True on noisy_state BEFORE model.step()
    2. Pass the noisy_state tensor to scale() as part of context.metadata["x_t"]
    """

    def __init__(self, step_size: float = 0.1, gradient_normalization: bool = False):
        self.step_size = step_size
        self.gradient_normalization = gradient_normalization
        self.requires_gradients = True

    def scale(
        self,
        state: Float[Tensor, "*batch atoms 3"],
        context: StepParams,
        *,
        model: FlowModelWrapper | None = None,
    ) -> tuple[Float[Tensor, "*batch atoms 3"], Float[Tensor, " batch"]]:
        if context.loss_function is None or context.loss_inputs is None:
            raise ValueError("StepParams missing loss_function/loss_inputs")

        if context.metadata is None or "x_t" not in context.metadata:
            raise ValueError(
                "NoiseSpaceDPSScaler requires 'x_t' in context.metadata. "
                "Sampler must pass noisy_state with requires_grad=True to be compatible with this"
                " scaler."
            )

        x_t = context.metadata["x_t"]

        _raise_value_error_if_atom_count_mismatch(state, context)

        loss = context.loss_function(
            coordinates=state,  # x0_pred, connected to x_t via model
            elements=context.loss_inputs.elements,
            b_factors=context.loss_inputs.b_factors,
            occupancies=context.loss_inputs.occupancies,
        )

        # Use loss.backward() instead of torch.autograd.grad() to work with
        # gradient checkpointing (which models like Boltz/Protenix use)
        if x_t.grad is not None:
            x_t.grad.zero_()
        loss.backward()

        grad = x_t.grad
        if grad is None:
            raise RuntimeError(
                "Gradient computation failed. Ensure x_t has requires_grad=True "
                "before model forward pass."
            )

        if self.gradient_normalization:
            grad_norm = grad.norm(dim=(-1, -2), keepdim=True)
            grad = grad / (grad_norm + 1e-8)

        return grad, loss.detach()

    def guidance_strength(self, context: StepParams) -> Float[Tensor, " batch"]:
        t = context.t_effective
        return torch.ones_like(torch.as_tensor(t)) * self.step_size
