"""Baseline: BioEmu2's own FKC steering sampler driven by a sampleworks loss.

``BioEmu2NativeSteering`` has the same ``sample()`` signature as ``FKSteering``, but runs
``bioemu2.denoising.steering.edm_fkc_sampler`` instead of ``AF3EDMSampler``. The loss enters
through :class:`SampleworksLossPotential`, a BioEmu2 ``Potential`` that aligns each particle's
denoised coordinates to the input frame and evaluates the loss on it. This gives a reference
point for sampleworks guidance on BioEmu2 that uses BioEmu2's own transport, annealing and
importance weights.

Each BioEmu2 particle is one structure, so the energy is a per-structure loss: an ensemble loss
that couples members (e.g. a multi-conformer density) sees each member on its own here.
"""

from typing import cast, Literal

import torch
from bioemu2.denoising.steering import edm_fkc_sampler, Potential
from bioemu2.model import InferenceModel
from loguru import logger
from torch import Tensor

from sampleworks.core.losses.protocol import (
    LossFunctionProtocol,
    LossInputs,
    prepare_loss_if_needed,
)
from sampleworks.core.scalers.protocol import GuidanceOutput
from sampleworks.models.bioemu2.wrapper import BioEmu2Conditioning, BioEmu2Wrapper
from sampleworks.utils.atom_reconciler import AtomReconciler
from sampleworks.utils.framework_utils import match_batch
from sampleworks.utils.structure_utils import process_structure_to_trajectory_input


WeightUpdate = Literal["smc", "guidance"]


class SampleworksLossPotential(Potential):
    """A sampleworks loss as a BioEmu2 steering potential.

    Parameters
    ----------
    loss_function : LossFunctionProtocol
        Prepared loss, evaluated on model-order coordinates in the input frame.
    loss_inputs : LossInputs
        Inputs for ``loss_function`` for a single structure.
    reconciler : AtomReconciler
        Maps model atoms to input atoms for the rigid alignment.
    alignment_reference : Tensor
        Input coordinates in model order, shape ``(1, atoms, 3)``.
    num_atoms : int
        Real atoms; BioEmu2 pads coordinates past this.
    scale : float
        Multiplier on the loss. BioEmu2's tilt is ``exp(-beta * U)`` with ``beta`` reaching 1 at
        the clean endpoint, so the scale sets how sharply particles are selected.
    """

    def __init__(
        self,
        loss_function: LossFunctionProtocol,
        loss_inputs: LossInputs,
        reconciler: AtomReconciler,
        alignment_reference: Tensor,
        num_atoms: int,
        scale: float = 1.0,
    ):
        self.loss_function = loss_function
        self.loss_inputs = loss_inputs
        self.reconciler = reconciler
        self.alignment_reference = alignment_reference
        self.num_atoms = num_atoms
        self.scale = scale

    def prepare(self, system) -> None:
        """Nothing to cache: the loss is prepared from the sampleworks structure."""

    def __call__(self, coords_angstrom: Tensor, *, t: float) -> Tensor:
        """Return ``scale * loss`` per particle, shape ``(particles,)``."""
        coords = coords_angstrom[:, : self.num_atoms]
        reference = match_batch(
            self.alignment_reference.to(coords), target_batch_size=coords.shape[0]
        )
        aligned, _ = self.reconciler.align(coords, reference, allow_gradients=True)
        energies = [
            self.loss_function(
                coordinates=aligned[i : i + 1],
                elements=self.loss_inputs.elements,
                b_factors=self.loss_inputs.b_factors,
                occupancies=self.loss_inputs.occupancies,
            ).reshape(())
            for i in range(aligned.shape[0])
        ]
        return self.scale * torch.stack(energies)


class BioEmu2NativeSteering:
    """Sample a BioEmu2 ensemble with BioEmu2's FKC sampler and a sampleworks loss.

    ``ensemble_size * num_particles`` particles are sampled, resampled within ``ensemble_size``
    independent groups of ``num_particles``; the lowest-energy particle of each group becomes one
    ensemble member.

    Parameters
    ----------
    ensemble_size : int
        Number of structures returned.
    num_steps : int
        Positive levels on BioEmu2's Karras grid.
    loss_scale : float
        Multiplier on the loss energy (see :class:`SampleworksLossPotential`).
    weight_update : {"smc", "guidance"}
        ``"smc"`` reweights with denoised energies only. ``"guidance"`` also shifts particles
        along the energy gradient, backpropagating through the denoiser.
    ess_threshold : float
        Effective-sample-size fraction below which a group is resampled.
    """

    def __init__(
        self,
        ensemble_size: int = 1,
        num_steps: int = 200,
        loss_scale: float = 1.0,
        weight_update: WeightUpdate = "smc",
        ess_threshold: float = 0.5,
    ):
        self.ensemble_size = ensemble_size
        self.num_steps = num_steps
        self.loss_scale = loss_scale
        self.weight_update = weight_update
        self.ess_threshold = ess_threshold

    def sample(
        self,
        structure: dict,
        model: BioEmu2Wrapper,
        sampler=None,
        step_scaler=None,
        loss_function: LossFunctionProtocol | None = None,
        num_particles: int = 2,
    ) -> GuidanceOutput:
        """Run the steered sampler; ``sampler`` and ``step_scaler`` are unused.

        Returns
        -------
        GuidanceOutput
            ``final_state`` is ``(ensemble, atoms, 3)`` in model order, aligned to the input.
            ``losses`` holds the unscaled loss of each returned member.
        """
        if loss_function is None:
            raise ValueError("BioEmu2NativeSteering needs a loss_function")
        if num_particles < 2:
            raise ValueError("BioEmu2NativeSteering needs num_particles >= 2 to resample")

        with torch.no_grad():
            features = model.featurize(structure)
        cond = cast(BioEmu2Conditioning, features.conditioning)
        prior = model.initialize_from_prior(1, features)
        processed = process_structure_to_trajectory_input(
            structure=structure, coords_from_prior=prior, features=features, ensemble_size=1
        )
        device = prior.device
        reconciler = processed.reconciler.to(device)
        loss_inputs = processed.to_loss_inputs(device=device)
        prepare_loss_if_needed(loss_function, loss_inputs, device=device)
        reference = torch.as_tensor(processed.input_coords, device=device, dtype=torch.float32)

        potential = SampleworksLossPotential(
            loss_function,
            loss_inputs,
            reconciler,
            reference,
            num_atoms=cond.num_atoms,
            scale=self.loss_scale,
        )
        multiplicity = self.ensemble_size * num_particles
        logger.info(
            f"BioEmu2 FKC ({self.weight_update}): {self.ensemble_size} groups of "
            f"{num_particles} particles, {self.num_steps} steps"
        )
        result = edm_fkc_sampler(
            batch=cond.batch,
            num_steps=self.num_steps,
            score_model=cast(InferenceModel, model.model),
            trunk_embeddings=cond.trunk_embeddings,
            multiplicity=multiplicity,
            potentials=[potential],
            resampling_group_size=num_particles,
            ess_threshold=self.ess_threshold,
            weight_update=self.weight_update,
            # Guidance weights are only defined for reheating to the previous grid level.
            reheat="sigma" if self.weight_update == "guidance" else "gamma",
        )

        coords = result.coords[:, : cond.num_atoms].float()
        with torch.no_grad():
            energies = potential(result.coords.float(), t=0.0) / self.loss_scale
            aligned, _ = reconciler.align(
                coords, match_batch(reference, target_batch_size=coords.shape[0])
            )
        # Particles are grouped contiguously; keep the best of each group.
        groups = energies.reshape(self.ensemble_size, num_particles)
        best = (
            groups.argmin(dim=1) + torch.arange(self.ensemble_size, device=device) * num_particles
        )
        final_state = aligned[best].detach()
        member_losses = energies[best].tolist()
        logger.info(f"BioEmu2 FKC member losses: {[round(x, 5) for x in member_losses]}")

        metadata: dict = {"trajectory_denoised": []}
        if reconciler.has_mismatch and processed.model_atom_array is not None:
            metadata["model_atom_array"] = processed.model_atom_array
            metadata["struct_atom_array"] = processed.atom_array
            metadata["reconciler"] = processed.reconciler

        return GuidanceOutput(
            structure=structure,
            final_state=final_state,
            trajectory=[final_state.unsqueeze(0).cpu()],
            losses=cast(list[float | None], member_losses),
            metadata=metadata,
        )
