"""``FKSteering`` configured like BioEmu2's own FK steering sampler."""

from sampleworks.core.scalers.fk_steering import FKSteering


def build_bioemu_fksteering(
    num_steps: int = 200,
    *,
    ensemble_size: int = 1,
    resampling_interval: int = 3,
    fk_lambda: float = 4.0,
    resampling_group_size: int | None = None,
    guidance_t_start: float = 0.0,
    t_start: float = 0.0,
) -> FKSteering:
    """Build an ``FKSteering`` that resamples the way BioEmu2's ``edm_fk`` preset does.

    BioEmu2's ``GuidedFKSampler`` (``bioemu2.denoising.steering.edm_steered_sampler``) is FK
    steering per Singhal et al.: every ``resampling_interval`` steps and at the last step it
    weighs each particle by ``exp(fk_lambda * (E_previous - E_now))`` on the denoised
    prediction, times a proposal correction for gradient guidance, and resamples
    systematically within groups of ``resampling_group_size``. Every particle is a single
    structure and every particle is returned. The defaults here are BioEmu2's.

    Pair it with :func:`sampleworks.models.bioemu2.sampler.build_bioemu2_edm_sampler`, whose
    defaults match the EDM settings of BioEmu2's steered sampler. The sampleworks loss takes
    the place of BioEmu2's potentials, and the step scaler passed to
    :meth:`FKSteering.sample` takes the place of BioEmu2's gradient descent on the denoised
    prediction. For resampling without gradient guidance (BioEmu2's ``num_gd_steps=0``), pass
    ``DataSpaceDPSScaler(step_size=0.0)``: it reports the loss but does not move particles.
    (``NoScalingScaler`` reports a zero loss, so nothing would be resampled.)
    ``FKSteering.sample(..., num_particles=n)`` returns ``n`` structures, BioEmu2's
    ``batch_size``.

    Parameters
    ----------
    num_steps : int
        Diffusion steps.
    ensemble_size : int
        Structures per particle. BioEmu2 uses 1; larger values resample whole ensembles, scored
        together by the loss, and return ``num_particles * ensemble_size`` structures.
    resampling_interval : int
        Steps between resamples.
    fk_lambda : float
        Multiplier on the loss in the FK weights.
    resampling_group_size : int | None
        Particles per independent resampling group; ``None`` resamples all particles together.
    guidance_t_start : float
        Fraction of the trajectory after which guidance and resampling start. BioEmu2 steers
        from the first step.
    t_start : float
        Starting time fraction for partial diffusion. BioEmu2 starts from noise.

    Returns
    -------
    FKSteering
        Trajectory scaler that resamples systematically and returns every particle.
    """
    return FKSteering(
        ensemble_size=ensemble_size,
        num_steps=num_steps,
        resampling_interval=resampling_interval,
        fk_lambda=fk_lambda,
        guidance_t_start=guidance_t_start,
        t_start=t_start,
        resampling_method="systematic",
        resampling_group_size=resampling_group_size,
        return_all_particles=True,
    )
