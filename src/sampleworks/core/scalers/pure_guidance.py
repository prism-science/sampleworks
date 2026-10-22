"""
Pure diffusion guidance, as described in [DriftLite](http://arxiv.org/abs/2509.21655)
"""

import torch
from loguru import logger
from tqdm import tqdm

from sampleworks.core.rewards.protocol import prepare_reward_if_needed, RewardFunctionProtocol
from sampleworks.core.samplers.protocol import TrajectorySampler
from sampleworks.core.scalers.protocol import GuidanceOutput, StepScalerProtocol
from sampleworks.models.protocol import FlowModelWrapper
from sampleworks.utils.structure_utils import process_structure_to_trajectory_input


class PureGuidance:
    """Pure guidance scaler - applies per-step guidance without resampling."""

    def __init__(
        self,
        ensemble_size: int = 1,
        num_steps: int = 200,
        t_start: float = 0.0,
        guidance_t_start: float = 0.0,
    ):
        """Initializes Pure Guidance scaler.

        Parameters
        ----------
        ensemble_size : int
            Number of structures in the ensemble to sample. Default is 1 (single structure).
        num_steps : int
            Number of diffusion steps to perform. Default is 200, matching AF3 defaults.
        t_start : float
            Starting "reverse" time t ∈ [0, 1] for guidance application. Default is 0
            (start from beginning).
        guidance_t_start : float
            Fraction of total steps after which to start applying guidance. Default is 0.
        """
        logger.info("Initialized Pure Guidance scaler.")
        self.ensemble_size = ensemble_size
        self.num_steps = num_steps
        self.guidance_start = int(guidance_t_start * num_steps)
        self.starting_step = int(t_start * num_steps)

    def sample(
        self,
        structure: dict,
        model: FlowModelWrapper,
        sampler: TrajectorySampler,
        step_scaler: StepScalerProtocol,
        reward: RewardFunctionProtocol,
        num_particles: int = 1,
    ) -> GuidanceOutput:
        """Samples an ensemble using pure guidance.

        Parameters
        ----------
        structure : dict
            Input atomworks structure dictionary. This may have optional configuration keys that are
            used for initialization of the features for the model.
        model : FlowModelWrapper
            FlowModelWrapper to use for sampling.
        sampler : TrajectorySampler
            Sampler to use for the diffusion trajectory.
        step_scaler : StepScalerProtocol
            StepScalerProtocol to use for guidance scaling.
        reward : RewardFunctionProtocol
            Reward function to use for guidance.
        num_particles : int (optional)
            Number of particles to sample in parallel. For PureGuidance, this is ignored since
            no reweighting/resampling is performed.
        """
        features = model.featurize(structure)

        coords = torch.as_tensor(
            model.initialize_from_prior(
                batch_size=self.ensemble_size,
                features=features,
            ),
        )

        processed_structure = process_structure_to_trajectory_input(
            structure=structure,
            coords_from_prior=coords,
            features=features,
            ensemble_size=self.ensemble_size,
        )

        reconciler = processed_structure.reconciler.to(coords.device)
        reward_inputs = processed_structure.to_reward_inputs(device=coords.device)
        # Two-phase rewards are bound to the same inputs every step hands to `reward`: model
        # atom order, the structure's B-factors on the common atoms, and the reconciled
        # reference coordinates. `processed_structure` (a frozen dataclass) and its atom
        # arrays are never written to.
        prepare_reward_if_needed(reward, reward_inputs, device=coords.device)

        trajectory_denoised: list[torch.Tensor] = []
        trajectory_next_step: list[torch.Tensor] = []
        losses: list[float | None] = []

        schedule = sampler.compute_schedule(num_steps=self.num_steps)
        if self.starting_step > 0:
            logger.info(
                f"Partial diffusion starting from step {self.starting_step} of {self.num_steps}."
            )
            starting_context = sampler.get_context_for_step(self.starting_step - 1, schedule)
            # coords will be a noisy version of input coords at this t
            coords = processed_structure.input_coords + coords * torch.as_tensor(
                starting_context.noise_scale
            )

        for i in tqdm(range(self.starting_step, self.num_steps)):
            context = sampler.get_context_for_step(i, schedule)
            apply_guidance = i >= self.guidance_start

            if apply_guidance:
                context = context.with_reward(reward, reward_inputs)

            context = context.with_reconciler(
                reconciler=reconciler,
                alignment_reference=processed_structure.input_coords,
            )

            step_output = sampler.step(
                state=coords,
                model_wrapper=model,
                context=context,
                scaler=step_scaler if apply_guidance else None,
                features=features,
            )

            coords = step_output.state
            trajectory_next_step.append(coords.clone().cpu())

            if step_output.denoised is not None:
                trajectory_denoised.append(step_output.denoised.clone().cpu())

            if step_output.loss is not None:
                losses.append(step_output.loss.mean().item())
            else:
                losses.append(None)

        metadata: dict = {"trajectory_denoised": trajectory_denoised}

        # Mismatch outputs need the model topology, filtered input topology, and canonical CPU
        # mapping so save_everything can restore input identities without stale coordinates.
        if reconciler.has_mismatch and processed_structure.model_atom_array is not None:
            metadata["model_atom_array"] = processed_structure.model_atom_array
            metadata["struct_atom_array"] = processed_structure.atom_array
            metadata["reconciler"] = processed_structure.reconciler

        return GuidanceOutput(
            structure=structure,
            final_state=coords,
            trajectory=trajectory_next_step,
            losses=losses,
            metadata=metadata,
        )
