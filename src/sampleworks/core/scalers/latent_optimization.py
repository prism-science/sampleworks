"""Inference-time latent-space optimization (IT-opt).

A Sampleworks port of the method in Maddipatla et al., "Inference-time optimization for
experiment-grounded protein ensemble generation" (ICML 2026, https://arxiv.org/abs/2602.24007).
Reference implementation: https://github.com/sai-advaith/it_opt -- cited paths such as
``run_it_optimization`` in ``protenix/it_optimization_manager.py`` are relative to that repo, and
name symbols rather than line numbers, which drift as that repo changes.

It treats the frozen structure model as a differentiable sampler and optimizes its cached
post-trunk latents -- the single representation ``s`` and the pair representation ``z`` -- against
an experimental reward evaluated on the *denoised* structure. No model weights are updated. See
``docs/IT_OPT_DESIGN.md`` for the design and its mapping to the reference.

The algorithm (kept faithful to the reference), with the reference's bugs fixed:

    extract (s, z) from the trunk once  ->  make them optimizable leaves
    for each optimization round (fresh diffusion noise each round):
        build ONE Adam over [s, z]                       # reference rebuilt it per step
        for each diffusion step:
            x0_hat  = differentiable denoise(noisy_t, s, z)   # one forward
            loss    = reward(x0_hat) + anchor(s, z)
            loss.backward();  per-latent grad-clip (s, then z);  adam.step()
            advance the trajectory one step (latents frozen)
    final clean sampling round with the optimized latents -> saved ensemble

Terminology: a **round** is one full traversal of the ``num_steps`` diffusion schedule; a **step**
is one point on it, costing one denoiser forward. ``outer_steps`` counts the optimization rounds,
and one final sampling round follows them. "Pass" is used only in its usual sense of a forward
pass through the model, never as a synonym for either unit.

Only Boltz1 and RF3 propagate gradients cleanly to the cached latents out of the
box. Protenix now does too -- its ``step()`` keeps an injected latent leaf attached
via ``detach_unless_leaf`` -- but still needs its diffusion cache disabled for z;
Boltz2 needs its diffusion conditioning recomputed. See ``docs/IT_OPT_TESTING.md``.

v1 does ONLY latent optimization: coordinate-space guidance is not applied (the
attached scaler produces a zero coordinate direction), so the sole steering comes
from the evolving latents.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from sampleworks.core.rewards.geometry import BondGeometryReward
from sampleworks.core.rewards.protocol import RewardFunctionProtocol
from sampleworks.core.samplers.protocol import TrajectorySampler
from sampleworks.core.scalers.protocol import GuidanceOutput, StepScalerProtocol
from sampleworks.core.scalers.step_scalers import NoScalingScaler
from sampleworks.eval.structure_utils import process_structure_to_trajectory_input
from sampleworks.models.latent_adapter import AttrLatentIO
from sampleworks.models.protocol import FlowModelWrapper, GenerativeModelInput


# ---- Reading the type hints in this file --------------------------------------
# A ": type" after a name (or "-> type" after a function) is only a HINT -- it CLAIMS what a
# value should be, but nothing enforces it: pass the wrong type and Python still runs the code,
# and deleting every hint changes nothing. Hints are for humans (and optional checkers like ty).
# (In the table, "|" means "or".)
#
#   with the hint                  plain Python             what it claims (useless in runtime)
#   num_steps: int = 200           num_steps = 200          should be an int
#   learning_rate: float = 0.05    learning_rate = 0.05     should be a float
#   optimize_single: bool = True   optimize_single = True   should be True or False
#   structure: dict                structure                should be a dict
#   model: FlowModelWrapper        model                    should be a FlowModelWrapper
#   weights: Sequence[float]       weights                  should be a list/tuple of floats
#   latents: Sequence[Tensor]      latents                  should be a list/tuple of Tensors
#   losses: list[float]            losses                   should be a list of floats
#   denoised: Tensor | None        denoised                 should be a Tensor, or None
#   f(...) -> GuidanceOutput       f(...)                   f should return a GuidanceOutput
# -------------------------------------------------------------------------------


class _GradEnablingScaler(NoScalingScaler):
    """``NoScalingScaler`` that also asks the sampler to run its denoiser under autograd.

    The zero-guidance behavior is inherited unchanged; the flag below is the only difference.
    ``AF3EDMSampler`` runs its denoiser forward under autograd only when the attached scaler
    exposes ``requires_gradients=True``, and it reads that with ``getattr(..., False)``. Plain
    ``NoScalingScaler`` therefore leaves :attr:`SamplerStepOutput.denoised` detached, and IT-opt
    would get no gradient back to the injected latent leaves.

    Keeping guidance at zero leaves the diffusion *advance* unguided, realizing "coordinate
    guidance disabled" without editing the sampler. The reward is recomputed by the trajectory
    scaler on ``denoised``, so there is exactly one denoiser forward and one reward evaluation
    per step (the reference's Tier-2 single-forward optimization, for free).
    """

    requires_gradients = True


class LatentAnchor:
    r"""On-manifold prior penalizing drift of the latents from their trunk baseline.

    ``penalty = Σ_i w_i · mean((latent_i − baseline_i)²)``.

    The reference ``AnchorLossFunction`` uses a per-sample Frobenius **norm**
    ``‖latent − baseline‖`` (see ``anchor_loss_function.py``). We use the *mean
    squared* deviation instead: it is shape-agnostic and normalizes for the very
    different element counts of ``s`` and ``z`` (``z`` is O(tokens) larger), so the
    weights ``w_s`` and ``w_z`` land on a comparable scale -- part of keeping the
    two latent updates harmonized. Retune the weights when switching objectives.

    Planned move: this belongs under ``core.rewards`` with
    ``BondGeometryReward``, so regularizers are collected and reusable. Deferred
    because its input is *latents*, not coordinates, so it fits neither
    ``RewardFunctionProtocol`` nor the coordinate-space shape of that package --
    the rewards interface has to admit latent-space terms first.
    """

    def __init__(self, weights: Sequence[float]):
        self.weights = list(weights)

    def __call__(self, latents: Sequence[Tensor], baselines: Sequence[Tensor]) -> Tensor:
        terms = [
            w * torch.mean((lat - base) ** 2)
            for w, lat, base in zip(self.weights, latents, baselines)
        ]
        return torch.stack(terms).sum()


class LatentOptimization:
    """Trajectory scaler that optimizes the model's ``s``/``z`` latents (IT-opt).

    Satisfies ``TrajectoryScalerProtocol`` and drops in alongside ``PureGuidance``
    / ``FKSteering``.
    """

    def __init__(
        self,
        ensemble_size: int = 1,
        num_steps: int = 200,
        guidance_t_start: float = 0.0,
        *,
        outer_steps: int = 1,
        learning_rate: float = 0.05,
        max_grad_norm: float = 1.0,
        optimize_single: bool = True,
        optimize_pair: bool = True,
        anchor_weight_single: float = 0.0,
        anchor_weight_pair: float = 0.0,
        bond_length_weight: float = 5e-5,  # smallest weight that fixes mean clash; see docstring
        single_attr: str = "s",
        pair_attr: str = "z",
    ):
        """Initialize the latent-optimization trajectory scaler.

        Parameters
        ----------
        ensemble_size
            Number of structures sampled in parallel (they share the latents).
        num_steps
            Number of diffusion steps in one round.
        guidance_t_start
            Fraction of ``num_steps`` after which to begin optimizing within each
            round. Before it, steps are plain frozen-latent diffusion. Default 0
            (optimize from the first step, as the reference does).
        outer_steps
            Number of optimization rounds. Each traverses the whole schedule with fresh
            prior noise, sharing the persisting latents (the reference's
            ``outer_diffusion_steps``). Default 1 for a cheap first run.
        learning_rate
            Adam learning rate. A *real* persistent Adam is built once per round --
            unlike the reference, which rebuilt it every diffusion step and thereby
            degenerated to signed gradient descent (its headline bug).
        max_grad_norm
            Per-latent gradient-clip threshold: each optimized latent (``s``, ``z``)
            is clipped to this norm **independently** (matching the reference), so
            ``s``'s update is not scaled down by ``z``'s much larger gradient. A single
            joint clip over ``[s, z]`` does the opposite -- ``z`` dominates the shared
            coefficient and starves ``s``.
        optimize_single, optimize_pair
            Which representations to optimize. Both by default.
        anchor_weight_single, anchor_weight_pair
            Weights of the on-manifold L2-to-baseline anchor per latent.
        bond_length_weight
            Weight of the coordinate-space bond-geometry penalty (bond-length + steric-clash
            hinges; see ``BondGeometryReward``). Pure latent opt (0) improves density fit but
            raises clashes; this penalty curbs that. Full-40 sweep -- mean/median clash, RSCC>=0.8
            (unguided baseline = 0.38 / 0.00 / 42%):
                0     -> 0.47 / 0.25 / 92%   (density gained, but clashes rose)
                5e-5  -> 0.38 / 0.25 / 92%   (mean clash back to baseline)   [DEFAULT]
                1e-4  -> 0.39 / 0.25 / 92%
                1e-3  -> 0.35 / 0.00 / 98%   (median clash 0; over-constrains, split 18->11%)
            Default 5e-5: smallest weight that fixes the mean clash while keeping the density gain
            and altloc diversity; use 1e-3 if you need median clash at 0 (accepts less spread).
        single_attr, pair_attr
            Conditioning attribute names for the single / pair representation
            (``"s"``/``"z"`` for Boltz, ``"s_trunk"``/``"z_trunk"`` for
            Protenix/RF3).
        """
        logger.info(
            f"Initialized LatentOptimization (IT-opt): outer_steps={outer_steps}, "
            f"num_steps={num_steps}, lr={learning_rate}, "
            f"optimize_single={optimize_single}, optimize_pair={optimize_pair}, "
            f"bond_length_weight={bond_length_weight}."
        )
        self.ensemble_size = ensemble_size
        self.num_steps = num_steps
        self.guidance_start = int(guidance_t_start * num_steps)
        self.outer_steps = outer_steps
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.optimize_single = optimize_single
        self.optimize_pair = optimize_pair
        self.anchor_weight_single = anchor_weight_single
        self.anchor_weight_pair = anchor_weight_pair
        self.bond_length_weight = bond_length_weight
        self.single_attr = single_attr
        self.pair_attr = pair_attr

    def sample(
        self,
        structure: dict,
        model: FlowModelWrapper,
        sampler: TrajectorySampler,
        step_scaler: StepScalerProtocol,
        reward: RewardFunctionProtocol,
        num_particles=1,
    ) -> GuidanceOutput:
        """Optimize the latents against ``reward``, then sample the final ensemble.

        ``step_scaler`` is ignored: v1 does latent optimization only (no
        coordinate-space guidance). ``reward`` is evaluated on the
        reconciler-aligned denoised prediction.
        """
        io = AttrLatentIO(
            single_attr=self.single_attr,
            pair_attr=self.pair_attr if self.optimize_pair else None,
        )

        # --- extract (s, z) once and make them optimizable leaves ---------------
        # no_grad avoids retaining a graph into the (frozen) trunk; we build leaves
        # from detached copies. (reference: get_msa_features + clone/detach.)
        with torch.no_grad():
            features = model.featurize(structure)
        features, latents, baselines, anchor_weights = self._leaf_latents(features, io)
        anchor = LatentAnchor(anchor_weights)

        # --- shared per-trajectory context (reconciler + reward inputs) ---------
        coords = torch.as_tensor(
            model.initialize_from_prior(batch_size=self.ensemble_size, features=features)
        )
        processed = process_structure_to_trajectory_input(
            structure=structure,
            coords_from_prior=coords,
            features=features,
            ensemble_size=self.ensemble_size,
        )
        reconciler = processed.reconciler.to(coords.device)
        reward_inputs = processed.to_reward_inputs(device=coords.device)
        schedule = sampler.compute_schedule(num_steps=self.num_steps)
        grad_enabler = _GradEnablingScaler()

        # --- optional coordinate-space geometry penalty -------------------------
        # BondGeometryReward penalizes stretched bonds and steric clashes in the denoised structure,
        # curbing the overshoot where an aggressive latent update trades geometry for density fit.
        if self.bond_length_weight > 0:
            # Prefer the model atom array; fall back to the input atom array if it is absent.
            geometry_atom_array = processed.model_atom_array or processed.atom_array
            bond_geometry = BondGeometryReward(
                geometry_atom_array, self.bond_length_weight, coords.device
            )
        else:
            bond_geometry = None

        # --- optimize the latents (outer resample × inner diffusion steps) ------
        # These two lists are diagnostics only -- reported in the returned metadata, never read by
        # the loop. One entry is appended per round.
        optimization_losses = []  # per round: the per-step data losses
        latent_drift = []  # per round: how far each latent moved from its baseline (see below)
        for outer in tqdm(range(self.outer_steps)):
            optimizer = torch.optim.Adam(latents, lr=self.learning_rate)  # a fresh, persistent Adam
            round_losses = self._optimize_one_round(
                model=model,
                sampler=sampler,
                reward=reward,
                features=features,
                latents=latents,
                baselines=baselines,
                anchor=anchor,
                optimizer=optimizer,
                schedule=schedule,
                reconciler=reconciler,
                alignment_reference=processed.input_coords,
                reward_inputs=reward_inputs,
                grad_enabler=grad_enabler,
                round_index=outer,
                bond_geometry=bond_geometry,
            )
            optimization_losses.append(round_losses)
            # relative drift of each latent from its trunk baseline -- shows which latent
            # actually moved (s vs z), which the loss trend alone cannot reveal.
            latent_drift.append(
                [
                    float((lat.detach() - base).norm() / (base.norm() + 1e-12))
                    for lat, base in zip(latents, baselines)
                ]
            )

        # --- final clean sampling round with the optimized latents --------------
        final_coords, trajectory, losses = self._sample_with_frozen_latents(
            model=model,
            sampler=sampler,
            reward=reward,
            io=io,
            features=features,
            latents=latents,
            schedule=schedule,
            reconciler=reconciler,
            alignment_reference=processed.input_coords,
            reward_inputs=reward_inputs,
        )

        metadata = {
            "optimization_losses": optimization_losses,
            "latent_drift": latent_drift,
        }
        if reconciler.has_mismatch and processed.model_atom_array is not None:
            metadata["model_atom_array"] = processed.model_atom_array

        return GuidanceOutput(
            structure=structure,
            final_state=final_coords,
            trajectory=trajectory,
            losses=losses,
            metadata=metadata,
        )

    def _leaf_latents(self, features: GenerativeModelInput, io: AttrLatentIO):
        """Replace ``s``/``z`` on the conditioning with fresh optimizable leaves.

        Returns the rewritten ``features`` plus parallel lists of leaves, their
        detached baselines (anchor targets), and per-latent anchor weights. Each
        leaf is a detached clone made ``requires_grad=True`` -- a true leaf severed
        from any trunk graph, so Adam updates it directly (leaves persist and are
        updated in place across rounds and steps). Shapes are preserved (whatever
        the wrapper caches), so no assumption is made about a batch dimension.
        """
        conditioning = features.conditioning
        latents: list[Tensor] = []
        baselines: list[Tensor] = []
        anchor_weights: list[float] = []

        # One row per optimizable latent: (optimize it?, its attribute on the conditioning, its
        # anchor weight). We handle the single representation, then the pair.
        specs = (
            (self.optimize_single, io.single_attr, self.anchor_weight_single),
            (self.optimize_pair, io.pair_attr, self.anchor_weight_pair),
        )
        for enabled, attr, anchor_weight in specs:
            if not enabled:
                continue
            baseline = getattr(conditioning, attr, None)
            if baseline is None:
                # Enabled but absent means the configured attribute name is wrong for this model.
                # Skipping it would optimize fewer latents than asked for and look like success.
                raise ValueError(
                    f"LatentOptimization was asked to optimize {attr!r}, but the model's "
                    "conditioning does not expose it. Check the attribute names for this model."
                )
            baseline = baseline.detach()
            leaf = baseline.clone().requires_grad_(True)
            # The conditioning is a frozen dataclass, so setattr would raise; replace() returns a
            # copy with this one field swapped and every sidecar field left untouched.
            conditioning = dataclasses.replace(conditioning, **{attr: leaf})
            latents.append(leaf)
            baselines.append(baseline)
            anchor_weights.append(anchor_weight)

        # A missing attribute already raised above, so reaching here means neither latent was
        # enabled -- nothing to optimize, which is a misconfiguration rather than a valid run.
        if not latents:
            raise ValueError(
                "LatentOptimization has no latent enabled "
                f"(optimize_single={self.optimize_single}, optimize_pair={self.optimize_pair})."
            )
        # Rebuild with the rewritten conditioning only. #330 removed x_init from
        # GenerativeModelInput; initial coords now come from initialize_from_prior at sampling time.
        features = GenerativeModelInput(conditioning=conditioning)
        return features, latents, baselines, anchor_weights

    def _optimize_one_round(
        self,
        *,
        model,
        sampler,
        reward,
        features,
        latents,
        baselines,
        anchor,
        optimizer,
        schedule,
        reconciler,
        alignment_reference,
        reward_inputs,
        grad_enabler,
        round_index,
        bond_geometry,
    ) -> list[float]:
        """One optimization round: a full traversal of the schedule that updates the latents.

        Fresh prior noise; the persisting latents are updated once per diffusion
        step against the reward on that step's denoised prediction, then the
        trajectory is advanced with the latents frozen. Returns the per-step data
        losses.
        """
        coords = torch.as_tensor(
            model.initialize_from_prior(batch_size=self.ensemble_size, features=features)
        )
        losses: list[float] = []
        steps = tqdm(range(self.num_steps), f"IT-opt round {round_index}")
        for i in steps:
            optimize = i >= self.guidance_start
            context = sampler.get_context_for_step(i, schedule)
            if optimize:
                context = context.with_reward(reward, reward_inputs)
            context = context.with_reconciler(
                reconciler=reconciler, alignment_reference=alignment_reference
            )

            step_output = sampler.step(
                state=coords,
                model_wrapper=model,
                context=context,
                scaler=grad_enabler if optimize else None,
                features=features,
            )

            if optimize:
                data_loss = self._latent_adam_step(
                    denoised=step_output.denoised,
                    reward=reward,
                    reward_inputs=reward_inputs,
                    optimizer=optimizer,
                    anchor=anchor,
                    latents=latents,
                    baselines=baselines,
                    bond_geometry=bond_geometry,
                )
                losses.append(data_loss)
                steps.set_postfix(loss=data_loss)

            coords = step_output.state.detach()
        return losses

    def _latent_adam_step(
        self,
        *,
        denoised: Tensor | None,
        reward: RewardFunctionProtocol,
        reward_inputs,
        optimizer: torch.optim.Optimizer,
        anchor: LatentAnchor,
        latents: Sequence[Tensor],
        baselines: Sequence[Tensor],
        bond_geometry: BondGeometryReward | None = None,
    ) -> float:
        """Score the denoised structure, backprop to the latents, take one Adam step.

        ``denoised`` is the sampler's aligned prediction and carries a graph back to
        the latent leaves. When ``bond_geometry`` is supplied, its bond-length/clash
        penalty is added to the loss alongside the anchor. Each latent is
        gradient-clipped **separately** so ``s``'s step is not scaled down by ``z``'s
        much larger gradient. Returns the data-only reward for logging.
        """
        if denoised is None:
            raise RuntimeError(
                "Sampler returned no denoised prediction; the grad-enabling scaler "
                "should have been attached this step so the latent gradient exists."
            )
        data_loss = reward(
            coordinates=denoised,
            elements=reward_inputs.elements,
            b_factors=reward_inputs.b_factors,
            occupancies=reward_inputs.occupancies,
        )
        loss = data_loss + anchor(latents, baselines)
        # Add the coordinate-space geometry penalty (bond-length + clash) when it is enabled, so the
        # latent update is pushed toward density fits that keep valid geometry; it backpropagates
        # through the same denoised prediction to the latents.
        if bond_geometry is not None:
            loss = loss + bond_geometry(denoised)

        optimizer.zero_grad()
        loss.backward()
        # Clip each latent SEPARATELY (the reference makes two clip_grad_norm_ calls too), never
        # as one joint [s, z] group. NB: with the density reward the gradients are tiny (grad-norm
        # ~1e-4, far below a max_grad_norm of 1.0), so this clip is effectively inert here -- it
        # bites only for rewards whose gradients exceed the threshold, where separate (not joint)
        # clips keep s's step decoupled from z's much larger gradient scale.
        for latent in latents:
            torch.nn.utils.clip_grad_norm_(latent, self.max_grad_norm)
        optimizer.step()
        return float(data_loss.detach())

    def _sample_with_frozen_latents(
        self,
        *,
        model,
        sampler,
        reward,
        io,
        features,
        latents,
        schedule,
        reconciler,
        alignment_reference,
        reward_inputs,
    ):
        """Final clean sampling round with the optimized latents held fixed.

        No optimization and no coordinate guidance (``scaler=None``). This produces
        the ensemble that is returned/saved -- the reference's
        ``run_diffusion_process_it_optimized``. The per-step reward (evaluated under
        no-grad on the denoised prediction) is returned for logging.
        """
        # Re-inject detached copies so the final round is purely a frozen sampler.
        conditioning = features.conditioning
        detached = [lat.detach() for lat in latents]
        if self.optimize_single and io.read_single(conditioning) is not None:
            conditioning = io.write_single(conditioning, detached.pop(0))
        if self.optimize_pair and io.read_pair(conditioning) is not None:
            conditioning = io.write_pair(conditioning, detached.pop(0))
        frozen_features = GenerativeModelInput(conditioning=conditioning)  # #330: conditioning-only

        coords = torch.as_tensor(
            model.initialize_from_prior(batch_size=self.ensemble_size, features=frozen_features)
        )
        trajectory: list[Tensor] = []
        losses: list[float | None] = []
        steps = tqdm(range(self.num_steps), "IT-opt final sampling")
        for i in steps:
            context = sampler.get_context_for_step(i, schedule).with_reconciler(
                reconciler=reconciler, alignment_reference=alignment_reference
            )
            step_output = sampler.step(
                state=coords,
                model_wrapper=model,
                context=context,
                scaler=None,
                features=frozen_features,
            )
            coords = step_output.state.detach()
            trajectory.append(coords.clone().cpu())

            if step_output.denoised is not None:
                with torch.no_grad():
                    loss = reward(
                        coordinates=step_output.denoised,
                        elements=reward_inputs.elements,
                        b_factors=reward_inputs.b_factors,
                        occupancies=reward_inputs.occupancies,
                    )
                losses.append(float(loss))
            else:
                losses.append(None)
        return coords, trajectory, losses
