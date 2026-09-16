"""Feynman-Kac Steering scaler implementation.

Implements TrajectoryScalerProtocol for particle-based guided diffusion sampling
using Feynman-Kaç steering. All per-step guidance comes from a given StepScalerProtocol.

Per Singhal et al. (arXiv 2501.06848) and Boltz-1x (doi:10.1101/2024.11.19.624167)
"""

import einx
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from sampleworks.core.rewards.protocol import prepare_reward_if_needed, RewardFunctionProtocol
from sampleworks.core.samplers.protocol import (
    SamplerStepOutput,
    StepParams,
    TrajectorySampler,
)
from sampleworks.core.scalers.protocol import GuidanceOutput, StepScalerProtocol
from sampleworks.models.protocol import FlowModelWrapper, GenerativeModelInput
from sampleworks.utils.structure_utils import (
    process_structure_to_trajectory_input,
    SampleworksProcessedStructure,
)


class FKSteering:
    """Feynman-Kac Steering trajectory scaler.

    The key difference from the standard FKSteering implementation is that "particles"
    are ensembles of structures, rather than individual structures.

    Parameters
    ----------
    ensemble_size : int
        Number of structures per particle. Default is 1.
    num_steps : int
        Number of diffusion steps. Default is 200.
    resampling_interval : int
        Steps between FK resampling. Default is 1 (every step).
    fk_lambda : float
        Weight for FK log-likelihood in resampling. Default is 1.0.
    t_start : float
        Starting time fraction for partial diffusion. Default is 0.0.
    guidance_t_start : float
        Fraction of trajectory after which to start guidance. Default is 0.0.
    """

    def __init__(
        self,
        ensemble_size: int = 1,
        num_steps: int = 200,
        resampling_interval: int = 1,
        fk_lambda: float = 1.0,
        guidance_t_start: float = 0.0,
        t_start: float = 0.0,
    ):
        self.ensemble_size = ensemble_size
        self.num_steps = num_steps
        self.resampling_interval = resampling_interval
        self.fk_lambda = fk_lambda
        self.guidance_start = int(guidance_t_start * num_steps)
        self.starting_step = int(t_start * num_steps)

    def sample(
        self,
        structure: dict,
        model: FlowModelWrapper,
        sampler: TrajectorySampler,
        step_scaler: StepScalerProtocol[FlowModelWrapper],
        reward: RewardFunctionProtocol,
        num_particles: int = 1,
    ) -> GuidanceOutput:
        """Generate samples using Feynman-Kac steering.

        Parameters
        ----------
        structure : dict
            Input atomworks structure dictionary.
        model : FlowModelWrapper
            Model wrapper for denoising steps.
        sampler : TrajectorySampler
            Sampler for trajectory generation (provides schedule and step mechanics).
        step_scaler : StepScalerProtocol
            Step scaler for per-step guidance.
        reward : RewardFunctionProtocol
            Reward function for computing particle energies.
        num_particles : int
            Number of particles. Default is 1, i.e., no population.

        Returns
        -------
        GuidanceOutput
            Output containing final state, trajectory, losses, and metadata.
        """

        features = model.featurize(structure)

        # Each particle is independently sampled from the prior to start
        coords = torch.as_tensor(
            model.initialize_from_prior(
                batch_size=self.ensemble_size * num_particles, features=features
            ),
        )

        processed: SampleworksProcessedStructure = process_structure_to_trajectory_input(
            structure=structure,
            coords_from_prior=coords[: self.ensemble_size],
            features=features,
            ensemble_size=self.ensemble_size,
        )

        reconciler = processed.reconciler.to(coords.device)
        reward_inputs = processed.to_reward_inputs(device=coords.device)
        # Two-phase rewards are bound to the same inputs every step hands to `reward`: model
        # atom order, the structure's B-factors on the common atoms, and the reconciled
        # reference coordinates. `processed` (a frozen dataclass) and its atom arrays are
        # never written to.
        prepare_reward_if_needed(reward, reward_inputs, device=coords.device)

        schedule = sampler.compute_schedule(self.num_steps)
        loss_history: list[torch.Tensor] = []

        trajectory_denoised: list[torch.Tensor] = []
        trajectory_next_step: list[torch.Tensor] = []
        losses: list[float | None] = []

        loss_prev: torch.Tensor | None = None
        log_proposal_correction_prev: torch.Tensor | None = None

        if self.starting_step > 0:
            logger.info(
                f"Partial diffusion starting from step {self.starting_step} of {self.num_steps}."
            )
            starting_context = sampler.get_context_for_step(self.starting_step - 1, schedule)
            # coords will be a noisy version of input coords at this t
            # input_coords: (ensemble, atoms, 3), coords: (particles * ensemble, atoms, 3)
            coords = einx.add(
                "e a c, (p e) a c -> (p e) a c",
                processed.input_coords,
                coords * torch.as_tensor(starting_context.noise_scale),
            )

        coords_4d_current = coords.reshape(num_particles, self.ensemble_size, -1, 3)

        pbar = tqdm(range(self.starting_step, self.num_steps), desc="FK Steering")
        for i in pbar:
            context = sampler.get_context_for_step(i, schedule)
            apply_guidance = i >= self.guidance_start

            if apply_guidance:
                context = context.with_reward(reward, reward_inputs)

            context = context.with_reconciler(
                reconciler=reconciler,
                alignment_reference=processed.input_coords,
            )

            # RESAMPLE
            if apply_guidance and loss_prev is not None and self._should_resample(i, context):
                coords = self._resample_particles(
                    coords=coords,
                    loss_curr=loss_prev,
                    loss_history=loss_history,
                    log_proposal_correction=log_proposal_correction_prev,
                    num_particles=num_particles,
                )

            # PROPOSE
            step_output, denoised_4d = self._run_step(
                coords=coords,
                model=model,
                sampler=sampler,
                features=features,
                context=context,
                step_scaler=step_scaler if apply_guidance else None,
                num_particles=num_particles,
            )

            # Store for next iteration's resampling
            loss = step_output.loss
            if loss is not None:
                loss = torch.as_tensor(loss)
                loss_prev = loss
                loss_history.append(loss.clone())

            if step_output.log_proposal_correction is not None:
                log_proposal_correction_prev = torch.as_tensor(step_output.log_proposal_correction)

            if denoised_4d is not None:
                trajectory_denoised.append(denoised_4d.clone().cpu())

            current_loss = (
                loss.mean().item()
                if loss is not None
                else (loss_history[-1].mean().item() if loss_history else 0.0)
            )
            losses.append(current_loss)
            pbar.set_postfix({"loss": current_loss})

            coords = step_output.state
            coords_4d_current = coords.reshape(num_particles, self.ensemble_size, -1, 3)
            trajectory_next_step.append(coords_4d_current.clone().cpu())

        # Select the particle (ensemble) with the best final reward.
        # final_coords_4d: (particles, ensemble, atoms, 3)
        # lowest_loss_coords: (ensemble, atoms, 3) — the winning ensemble's
        # denoised coordinates.
        final_coords_4d = coords_4d_current
        lowest_loss_index = torch.argmin(loss_history[-1]) if loss_history else 0
        lowest_loss_coords = final_coords_4d[lowest_loss_index]

        metadata: dict = {"trajectory_denoised": trajectory_denoised}

        # If we had a mismatch, we need to add this key to the metadata so the save_everything
        # function can get the right number of atoms.
        if reconciler.has_mismatch and processed.model_atom_array is not None:
            metadata["model_atom_array"] = processed.model_atom_array

        return GuidanceOutput(
            structure=structure,
            final_state=lowest_loss_coords,
            trajectory=trajectory_next_step,
            losses=losses,
            metadata=metadata,
        )

    def _run_step(
        self,
        coords: torch.Tensor,
        model: FlowModelWrapper,
        sampler: TrajectorySampler,
        features: GenerativeModelInput,
        context: StepParams,
        step_scaler: StepScalerProtocol | None,
        num_particles: int,
    ) -> tuple[SamplerStepOutput, torch.Tensor | None]:
        r"""Run a single denoising step, optionally with per-particle guidance.

        In FK steering, each "particle" is an ensemble of structures/trajectories.
        The particle dimension indexes independent
        populations that are resampled according to FK weights.

        When step_scaler is provided, loops over particles to ensure each particle
        receives independent guidance. This gives correct FK semantics where each
        particle's loss gradient is computed only from its own state.

        Parameters
        ----------
        coords : torch.Tensor
            Current coordinates, shape ``(particles * ensemble, atoms, 3)``.
            Comes from the previous step's output (or ``initialize_from_prior``
            on the first step).
        model : FlowModelWrapper
            Model wrapper for denoising.
        sampler : TrajectorySampler
            Sampler that handles the step mechanics.
        features : GenerativeModelInput
            Model features/inputs.
        context : StepParams
            Step context with time info and optionally reward.
        step_scaler : StepScalerProtocol | None
            Step scaler for guidance (None if guidance not active).
        num_particles : int
            Number of particles.

        Returns
        -------
        tuple[SamplerStepOutput, torch.Tensor | None]
            ``(step_output, denoised_4d)`` where
            - ``step_output.state`` has shape ``(particles * ensemble, atoms, 3)``
            - ``denoised_4d`` has shape ``(particles, ensemble, atoms, 3)``
        """
        coords_4d = coords.reshape(num_particles, self.ensemble_size, -1, 3)

        if step_scaler is None:
            # No guidance - batch all particles together (fast path)
            step_output = sampler.step(
                state=coords,
                model_wrapper=model,
                context=context,
                scaler=None,
                features=features,
            )

            denoised_4d = None
            if step_output.denoised is not None:
                denoised_4d = step_output.denoised.reshape(num_particles, self.ensemble_size, -1, 3)

            return step_output, denoised_4d

        return self._run_guided_step_per_particle(
            coords_4d, model, sampler, features, context, step_scaler, num_particles
        )

    def _run_guided_step_per_particle(
        self,
        coords_4d: torch.Tensor,
        model: FlowModelWrapper,
        sampler: TrajectorySampler,
        features: GenerativeModelInput,
        context: StepParams,
        step_scaler: StepScalerProtocol,
        num_particles: int,
    ) -> tuple[SamplerStepOutput, torch.Tensor | None]:
        r"""Run the denoising step one particle at a time, then restack to a batch.

        Each particle is stepped independently so its guidance gradient is computed
        only from its own state (correct FK semantics). This is the guided
        counterpart to the batched fast path in :meth:`_run_step`; the two are kept
        separate because the fast path runs a single forward pass over all particles,
        which cannot provide per-particle gradients.

        Parameters
        ----------
        coords_4d : torch.Tensor
            Current coordinates, shape ``(particles, ensemble, atoms, 3)``.
        model : FlowModelWrapper
            Model wrapper for denoising.
        sampler : TrajectorySampler
            Sampler that handles the step mechanics.
        features : GenerativeModelInput
            Model features/inputs.
        context : StepParams
            Step context with time info and optionally reward.
        step_scaler : StepScalerProtocol
            Active step scaler providing per-particle guidance.
        num_particles : int
            Number of particles to iterate over.

        Returns
        -------
        tuple[SamplerStepOutput, torch.Tensor | None]
            ``(step_output, denoised_4d)`` matching :meth:`_run_step`, with
            ``step_output.state`` shape ``(particles * ensemble, atoms, 3)`` and
            ``denoised_4d`` shape ``(particles, ensemble, atoms, 3)``.
        """
        states_per_particle: list[torch.Tensor] = []
        denoised_per_particle: list[torch.Tensor] = []
        losses_per_particle: list[torch.Tensor] = []
        log_proposal_corrections_per_particle: list[torch.Tensor] = []

        for p in range(num_particles):
            particle_coords = coords_4d[p]  # (ensemble, atoms, 3)

            particle_output = sampler.step(
                state=particle_coords,
                model_wrapper=model,
                context=context,
                scaler=step_scaler,
                features=features,
            )

            states_per_particle.append(particle_output.state)

            if particle_output.denoised is not None:
                denoised_per_particle.append(particle_output.denoised)

            if particle_output.loss is not None:
                loss = particle_output.loss
                if loss.ndim > 0:
                    loss = loss.mean()
                losses_per_particle.append(torch.as_tensor(loss))

            if particle_output.log_proposal_correction is not None:
                log_proposal_correction = particle_output.log_proposal_correction
                if log_proposal_correction.ndim > 0:
                    log_proposal_correction = log_proposal_correction.mean()
                log_proposal_corrections_per_particle.append(
                    torch.as_tensor(log_proposal_correction)
                )

        # Stack results back to 4D: (particles, ensemble, atoms, 3)
        state_4d = torch.stack(states_per_particle, dim=0)

        denoised_4d = None
        if denoised_per_particle:
            denoised_4d = torch.stack(denoised_per_particle, dim=0)

        loss_per_particle = None
        if losses_per_particle:
            loss_per_particle = torch.stack(losses_per_particle, dim=0)

        log_proposal_per_particle = None
        if log_proposal_corrections_per_particle:
            log_proposal_per_particle = torch.stack(log_proposal_corrections_per_particle, dim=0)

        combined_output = SamplerStepOutput(
            state=state_4d.reshape(-1, state_4d.shape[-2], 3),
            denoised=(
                denoised_4d.reshape(-1, denoised_4d.shape[-2], 3)
                if denoised_4d is not None
                else None
            ),
            loss=loss_per_particle,
            log_proposal_correction=log_proposal_per_particle,
        )

        return combined_output, denoised_4d

    def _should_resample(self, step: int, context: StepParams) -> bool:
        """Determine if FK resampling should occur at this step.

        Called once per step with the StepParams produced by the sampler's
        schedule.  The noise_scale check guards against misconfigured schedules.
        """
        if context.noise_scale is None:
            message = "Cannot determine FK resampling step due to None noise_scale in StepParams."
            logger.error(message)
            raise ValueError(message)

        is_resampling_step = step % self.resampling_interval == 0 and float(context.noise_scale) > 0
        is_final_step = step == self.num_steps - 1
        return is_resampling_step or is_final_step

    def _resample_particles(
        self,
        coords: torch.Tensor,
        loss_curr: torch.Tensor,
        loss_history: list[torch.Tensor],
        log_proposal_correction: torch.Tensor | None,
        num_particles: int,
    ) -> torch.Tensor:
        r"""Resample particles using FK weights before the next step.

        Each *particle* is an ensemble of ``self.ensemble_size`` structures.
        Resampling duplicates or drops entire ensembles — individual structures
        within a particle are never split across particles.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates, shape ``(particles * ensemble, atoms, 3)``.
        loss_curr : torch.Tensor
            Loss per particle from the current step, shape ``(particles,)``.
        loss_history : list[torch.Tensor]
            History of per-particle loss tensors from previous steps.
        log_proposal_correction : torch.Tensor | None
            Log-ratio of base to guided proposal, shape ``(particles,)``, or
            ``None`` when the sampler does not produce a correction term.
        num_particles : int
            Number of FK particles.

        Returns
        -------
        torch.Tensor
            Resampled coordinates, shape ``(particles * ensemble, atoms, 3)``.
        """
        if len(loss_history) < 2:
            log_G = -self.fk_lambda * loss_curr
        else:
            log_G = self.fk_lambda * (loss_history[-2] - loss_history[-1])

        if log_proposal_correction is not None:
            log_G = log_G + log_proposal_correction

        weights = F.softmax(log_G, dim=0)
        indices = torch.multinomial(weights, num_particles, replacement=True)

        # Resample coordinates
        coords_4d = coords.reshape(num_particles, self.ensemble_size, -1, 3)
        resampled = coords_4d[indices]

        # Also resample loss history to keep particle correspondence
        for j in range(len(loss_history)):
            loss_history[j] = loss_history[j][indices]

        return resampled.reshape(-1, coords_4d.shape[-2], 3)
