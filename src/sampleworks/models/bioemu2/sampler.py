"""``AF3EDMSampler`` configured the way BioEmu2 samples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig


if TYPE_CHECKING:
    from sampleworks.models.bioemu2.wrapper import BioEmu2Wrapper, SamplingMode


def build_bioemu2_edm_sampler(
    model_wrapper: BioEmu2Wrapper,
    mode: SamplingMode = "equilibrium",
    *,
    device: str | torch.device | None = None,
    augmentation: bool = True,
    align_to_input: bool = True,
    alignment_reverse_diffusion: bool = True,
    **overrides,
) -> AF3EDMSampler:
    """Build an ``AF3EDMSampler`` with BioEmu2's own EDM sampler settings.

    ``sigma_data``, ``sigma_min`` and ``sigma_max`` come from the checkpoint: the denoiser
    preconditions its inputs with the same schedule, so they are fixed by the model. The rest
    are the defaults of ``bioemu2.denoising.edm.edm_sampler``: ``gamma_0=0.8``,
    ``gamma_min=1.0``, ``noise_scale=1.0``, and ``step_scale`` 1.0 in equilibrium mode or 1.5
    in structure mode.

    Parameters
    ----------
    model_wrapper : BioEmu2Wrapper
        Loaded BioEmu2 model; supplies the checkpoint's EDM schedule.
    mode : {"equilibrium", "structure"}
        Sampling mode the structure is featurized with; sets ``step_scale``.
    device : str | torch.device | None
        Device for the schedule tensors. Defaults to the wrapper's device.
    augmentation, align_to_input, alignment_reverse_diffusion : bool
        Passed to :class:`EDMSamplerConfig`. BioEmu2 aligns the noisy state to the
        denoised prediction before each step, so ``alignment_reverse_diffusion`` defaults
        to True.
    **overrides
        Other :class:`EDMSamplerConfig` fields, applied over BioEmu2's values.

    Returns
    -------
    AF3EDMSampler
        Sampler for ``model_wrapper``.
    """
    schedule = model_wrapper.model.edm_schedule
    settings = {
        "sigma_data": schedule.sigma_data,
        "s_max": schedule.sigma_max,
        "s_min": schedule.sigma_min,
        "gamma_0": 0.8,
        "gamma_min": 1.0,
        "noise_scale": 1.0,
        "step_scale": 1.0 if mode == "equilibrium" else 1.5,
        **overrides,
    }
    config = EDMSamplerConfig(
        device=device if device is not None else model_wrapper.device,
        augmentation=augmentation,
        align_to_input=align_to_input,
        alignment_reverse_diffusion=alignment_reverse_diffusion,
        **settings,
    )
    return AF3EDMSampler(config=config)
