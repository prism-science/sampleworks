"""BioEmu2's EDM sampler settings for :class:`~sampleworks.core.samplers.edm.AF3EDMSampler`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sampleworks.core.samplers.schedule import NoiseScheduleConfig


if TYPE_CHECKING:
    from sampleworks.models.bioemu2.wrapper import BioEmu2Wrapper, SamplingMode


def bioemu2_edm_settings(model_wrapper: BioEmu2Wrapper, mode: SamplingMode = "equilibrium") -> dict:
    """Return BioEmu2's EDM sampler settings as :class:`EDMSamplerConfig` keyword arguments.

    Build a sampler by spreading these into an
    :class:`~sampleworks.core.samplers.edm.EDMSamplerConfig` (overriding any field as needed)
    and passing the config to :class:`~sampleworks.core.samplers.edm.AF3EDMSampler`.

    The noise schedule (``sigma_data``, ``sigma_min``, ``sigma_max``) comes from the checkpoint:
    the denoiser preconditions its inputs with the same schedule, so it is fixed by the model.
    The rest are the defaults of ``bioemu2.denoising.edm.edm_sampler``: ``gamma_0=0.8``,
    ``gamma_min=1.0``, ``noise_scale=1.0``, and ``step_scale`` 1.0 in equilibrium mode or 1.5
    in structure mode.

    Parameters
    ----------
    model_wrapper : BioEmu2Wrapper
        Loaded BioEmu2 model; supplies the checkpoint's EDM schedule.
    mode : {"equilibrium", "structure"}
        Sampling mode the structure is featurized with; sets ``step_scale``.

    Returns
    -------
    dict
        ``EDMSamplerConfig`` field values: ``schedule``, ``gamma_0``, ``gamma_min``,
        ``noise_scale`` and ``step_scale``.
    """
    schedule = model_wrapper.model.edm_schedule
    return {
        "schedule": NoiseScheduleConfig(
            sigma_data=schedule.sigma_data, s_max=schedule.sigma_max, s_min=schedule.sigma_min
        ),
        "gamma_0": 0.8,
        "gamma_min": 1.0,
        "noise_scale": 1.0,
        "step_scale": 1.0 if mode == "equilibrium" else 1.5,
    }
