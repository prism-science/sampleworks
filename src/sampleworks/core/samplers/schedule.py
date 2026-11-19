"""Noise-schedule configuration shared by the trajectory samplers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float


@dataclass(frozen=True, slots=True)
class NoiseScheduleConfig:
    r"""Noise levels and spacing of a variance-exploding (VE) diffusion schedule.

    Default values match the AF3 parameterization of the EDM framework
    (Karras et al., 2022). ``sigma_data`` should match what the model was
    trained on; ``s_max``/``s_min``/``p`` only choose how the same noise
    process is discretized.

    Parameters
    ----------
    sigma_data
        Assumed standard deviation of the training data distribution, which
        scales every noise level of the schedule.
    s_max
        Upper bound of the noise schedule in units of :math:`\sigma_\text{data}`.
        Controls the starting (maximum) noise level.
    s_min
        Lower bound of the noise schedule in units of :math:`\sigma_\text{data}`.
        Controls the ending (minimum) noise level.
    p
        Exponent :math:`\rho` controlling the shape of the interpolation
        between ``s_max`` and ``s_min`` (Karras spacing). Higher values
        concentrate more steps at lower noise levels. Called ``rho`` in
        Karras et al., Table 5. ``None`` (or ``math.inf``) gives geometric
        spacing, uniform in :math:`\log \sigma`: the VE discretization of
        Song et al. and the :math:`\rho \to \infty` limit of the Karras formula.

    References
    ----------
    Karras et al. "Elucidating the Design Space of Diffusion-Based Generative
    Models" (NeurIPS 2022). https://arxiv.org/abs/2206.00364
    """

    sigma_data: float = 16.0
    s_max: float = 160.0
    s_min: float = 4e-4
    p: float | None = 7.0

    def __post_init__(self) -> None:
        if self.p is not None and self.p <= 0:
            raise ValueError(f"p ({self.p}) must be positive, or None for geometric spacing")
        if self.s_max <= 0 or self.s_min <= 0:
            raise ValueError(f"s_max ({self.s_max}) and s_min ({self.s_min}) must be positive")
        if self.s_min >= self.s_max:
            raise ValueError(f"s_min ({self.s_min}) must be less than s_max ({self.s_max})")
        if self.sigma_data <= 0:
            raise ValueError(f"sigma_data ({self.sigma_data}) must be positive")

    def sigma(self, t: Float[torch.Tensor, "*t"]) -> Float[torch.Tensor, "*t"]:
        r"""Noise level at schedule position ``t``.

        With a finite ``p`` (Karras spacing):

        .. math::

            \sigma(t) = \sigma_{\text{data}} \cdot
            \left(s_{\max}^{1/p} + t \cdot (s_{\min}^{1/p} - s_{\max}^{1/p})\right)^p

        With geometric spacing:

        .. math::

            \sigma(t) = \sigma_{\text{data}} \cdot
            \exp\left((1 - t) \log s_{\max} + t \log s_{\min}\right)

        Parameters
        ----------
        t
            Schedule position in [0, 1]: 0 is the maximum noise level
            ``sigma_data * s_max`` and 1 the minimum ``sigma_data * s_min``.

        Returns
        -------
        Float[torch.Tensor, "*t"]
            Noise levels, same shape and device as ``t``.
        """
        p = self.p
        if p is None or math.isinf(p):
            log_sigmas = torch.lerp(
                torch.full_like(t, math.log(self.s_max)),
                torch.full_like(t, math.log(self.s_min)),
                t,
            )
            return self.sigma_data * torch.exp(log_sigmas)

        return (
            self.sigma_data
            * (self.s_max ** (1 / p) + t * (self.s_min ** (1 / p) - self.s_max ** (1 / p))) ** p
        )
