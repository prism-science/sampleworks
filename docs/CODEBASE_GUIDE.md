# Sampleworks Codebase Guide

A function-level tour of `src/sampleworks/`, tracing how every part connects: what calls
what, and why. Read [AGENTS.md](../AGENTS.md) first for the design philosophy — this
document is the *call graph*, showing how that philosophy is wired together in practice.

> **The one-sentence model.** Sampleworks treats a structure-prediction model (Boltz,
> Protenix, RF3) as a *prior* over realistic structures, then steers its diffusion
> sampling with the *gradient of an experimental-fit reward* (X-ray / cryo-EM density),
> drawing samples from the Bayesian posterior. Everything below is machinery for that one
> idea.

---

## 1. The 30-second map

```
ENTRY POINTS (pyproject.toml [project.scripts])
  sampleworks-guidance  → cli/guidance.py       (one run, in-process)
  sampleworks-runs      → runs/cli.py           (TOML-preset grid search; spawns one
                                                subprocess per job, each re-entering the spine)
  sampleworks-analysis  → runs/analysis_cli.py  (post-hoc eval; never enters the spine)
        │
        │  the two guidance paths converge on ↓
        ▼
  run_guidance()  in  utils/guidance_script_utils.py   ← THE SPINE
        │
        ├─ get_reward_function_and_structure()   → RealSpaceRewardFunction + atomworks structure
        ├─ get_model_and_device()                → Boltz/Protenix/RF3 wrapper on a device
        ├─ AF3EDMSampler(EDMSamplerConfig)        → the diffusion solver
        ├─ {DataSpace,NoiseSpace,No}DPSScaler     → per-step guidance rule
        └─ {PureGuidance | FKSteering | LatentOptimization}.sample(structure, model, sampler, ...)
               │
               └─ FOR each diffusion step:
                    sampler.step(coords, model, context, scaler)
                        ├─ model.step(noisy, t)          → x̂₀  (denoised prediction)
                        ├─ reconciler.align(x̂₀, ref)     → into experimental frame
                        └─ scaler.scale(x̂₀, context)     → reward(x̂₀), ∇reward → guidance
        ▼
  save_everything()  → refined.cif, trajectory/*.cif, losses.txt, job_metadata.json
```

The five collaborators (`model`, `sampler`, `step_scaler`, `trajectory_scaler`, `reward`)
never import each other's concrete classes. They meet only through four **Protocols**,
which is what lets any model pair with any reward and any guidance strategy (the O(N+M)
promise in AGENTS.md).

### 1.1 Directory orientation

| Path under `src/sampleworks/` | What lives here | Guide section |
|-------------------------------|-----------------|---------------|
| `__init__.py` | Package init; the `NAN_CHECK` env toggle (`should_check_nans`) that turns NaN assertions on/off globally | §8 |
| `cli/` | `sampleworks-guidance` entry point | §10 |
| `runs/` | `sampleworks-runs` / `sampleworks-analysis` TOML-preset orchestrator | §10 |
| `core/samplers/` | Diffusion solver protocol + `AF3EDMSampler` | §4.2, §2 |
| `core/scalers/` | Step scalers (DPS) + trajectory scalers (PureGuidance/FKSteering/LatentOptimization) | §4.1, §4.3, §2 |
| `core/rewards/` | Reward protocol + `RealSpaceRewardFunction` | §5, §2 |
| `core/forward_models/` | Differentiable X-ray density calc + vendored qFit crystallography | §5 |
| `models/` | Model-wrapper protocols + Boltz/Protenix/RF3 wrappers + latent adapter | §6, §2 |
| `utils/` | The `run_guidance` spine, alignment, config, MSA, CIF/density helpers | §3, §7, §8 |
| `eval/` | Offline eval harness + synthetic-data generation + the sampling→reward bridge | §9, §4.1 |
| `metrics/` | LDDT / RMSD / sidechain quality metric framework | §9 |
| `data/` | Reference data — `protein_configs.csv` (per-protein map/selection configs) | §9 |

Note the per-protein worker script `run_grid_search.py` lives at the **repository root**
(not under `src/`); the grid-search orchestrator spawns it as a subprocess (§10).

---

## 2. The four protocols (the contracts everything speaks)

These are the load-bearing interfaces. All are `@runtime_checkable` `typing.Protocol`s —
structural typing, no inheritance.

| Protocol | File | Key methods | Implemented by |
|----------|------|-------------|----------------|
| **Model wrappers** | [models/protocol.py](../src/sampleworks/models/protocol.py) | `featurize()`, `step()`, `initialize_from_prior()` | Boltz1/2, Protenix, RF3 |
| **Samplers** | [core/samplers/protocol.py](../src/sampleworks/core/samplers/protocol.py) | `compute_schedule()`, `get_context_for_step()`, `step()` | `AF3EDMSampler` |
| **Scalers** | [core/scalers/protocol.py](../src/sampleworks/core/scalers/protocol.py) | `StepScalerProtocol.scale()` · `TrajectoryScalerProtocol.sample()` | DPS scalers · PureGuidance/FKSteering/LatentOptimization |
| **Rewards** | [core/rewards/protocol.py](../src/sampleworks/core/rewards/protocol.py) | `__call__()`, `precompute_unique_combinations()` | `RealSpaceRewardFunction` |

### 2.1 The data objects that flow between them

Understanding these five dataclasses is enough to read any part of the sampling loop.

- **`GenerativeModelInput`** ([models/protocol.py:25](../src/sampleworks/models/protocol.py))
  — `{conditioning}`. Produced by `featurize()`. `conditioning` is a model-specific object
  holding the cached trunk/pairformer output. Starting coordinates are not carried here; they
  come from `initialize_from_prior(batch_size=...)` at sampling time.

- **`StepParams`** ([core/samplers/protocol.py:44](../src/sampleworks/core/samplers/protocol.py))
  — the per-step context bundle. Carries `t, dt, noise_scale` (diffusion timing) plus
  optional `reward`, `reward_inputs`, `reconciler`, `alignment_reference`, `metadata`. Built
  fresh each step and *enriched* immutably via `.with_reward()`, `.with_reconciler()`,
  `.with_metadata()`. This is how the reward and alignment reach the sampler without the
  sampler knowing what they are.

- **`SamplerStepOutput`** ([core/samplers/protocol.py:176](../src/sampleworks/core/samplers/protocol.py))
  — `{state, denoised, loss, log_proposal_correction}`. `log_proposal_correction` is the
  log-ratio of base-to-guided proposal density, consumed only by FK-steering resampling.

- **`RewardInputs`** ([core/rewards/protocol.py:17](../src/sampleworks/core/rewards/protocol.py))
  — `{elements, b_factors, occupancies, input_coords}`, pre-extracted once from the atom
  array (`from_atom_array()`) so `scale()` doesn't re-extract every step. Validates: no NaN
  coords/B-factors, occupancy in [0,1]. Tiles across ensemble/particle dims.

- **`GuidanceOutput`** ([core/scalers/protocol.py:40](../src/sampleworks/core/scalers/protocol.py))
  — `{structure, final_state, trajectory, losses, metadata}`. What a trajectory scaler
  returns; `metadata` carries `trajectory_denoised` and `model_atom_array`.

---

## 3. The spine: `run_guidance()` step by step

File: [utils/guidance_script_utils.py](../src/sampleworks/utils/guidance_script_utils.py).
Both guidance entry points reach `run_guidance()` — directly from `sampleworks-guidance`, or
once per spawned subprocess under `sampleworks-runs`. §10 walks both end to end.

`run_guidance()` ([:427](../src/sampleworks/utils/guidance_script_utils.py)) is a thin
wrapper — it sets up per-job logging, times the run, catches exceptions into a `JobResult`,
and writes `job_metadata.json`. The real work is in `_run_guidance()` ([:481](../src/sampleworks/utils/guidance_script_utils.py)):

1. **Load inputs & build reward** — `get_reward_function_and_structure()` ([:273](../src/sampleworks/utils/guidance_script_utils.py)):
   - `resolve_mixed_hetatm_atom_altlocs()` fixes a CIF edge case (mixed ATOM/HETATM altlocs
     that atomworks would misparse as an insertion), then `atomworks.parse(hydrogen_policy="remove")`.
   - `XMap.fromfile(density, resolution)` loads the experimental map.
   - `setup_scattering_params(em_mode)` builds the scattering-factor lookup table.
   - Returns a **`RealSpaceRewardFunction`** + the structure dict.

2. **Model-specific structure annotation** — dispatched on wrapper class name:
   `annotate_structure_for_protenix` / `annotate_structure_for_rf3` /
   `process_structure_for_boltz` (the last writes NPZ/manifest/MSA into the job's
   `output_dir` so parallel grid jobs don't race). This stamps `ensemble_size`,
   `recycling_steps`, etc. onto the structure dict for the wrapper's `featurize()` to read.

3. **Build the sampler** — `AF3EDMSampler(EDMSamplerConfig(...))` ([:563](../src/sampleworks/utils/guidance_script_utils.py)).
   Note `alignment_reverse_diffusion` defaults on only for Boltz.

4. **Build the step scaler** — `dataspace` → `DataSpaceDPSScaler`, `noisespace` →
   `NoiseSpaceDPSScaler`, `none` → `NoScalingScaler` ([:582](../src/sampleworks/utils/guidance_script_utils.py)).

5. **Build the trajectory scaler & run** — `PureGuidance`, `FKSteering`, or `LatentOptimization`,
   then call `.sample(structure, model_wrapper, sampler, step_scaler, reward_function[, num_particles])`.
   `guidance_start` and `partial_diffusion_step` are converted from step counts to fractions
   of `num_steps` here. `LatentOptimization` optimizes the model's cached trunk latents instead
   of steering coordinates, so it ignores `step_scaler`.

6. **Save** — `save_everything()` ([:320](../src/sampleworks/utils/guidance_script_utils.py))
   writes `refined.cif` (final ensemble coords written into the atom-array template, plus the
   config injected as a `sampleworks` CIF category via `add_category_to_cif`), the denoised and
   next-step trajectories (sub-sampled every 10 steps), and `losses.txt`.

`run_guidance_job_queue()` ([:820](../src/sampleworks/utils/guidance_script_utils.py)) is
the batch entry: unpickle a list of `GuidanceConfig`, load the model **once**, loop
`run_guidance()` over jobs, emptying CUDA cache between them.

---

## 4. The sampling loop in detail

### 4.1 Trajectory scalers — the loop *around* the sampler

All three live in `core/scalers/`. They implement `TrajectoryScalerProtocol.sample()`, whose job
is: featurize → sample prior → build reconciler → run the step loop → package
`GuidanceOutput`.

**`PureGuidance.sample()`** ([core/scalers/pure_guidance.py:46](../src/sampleworks/core/scalers/pure_guidance.py)) — standard guided diffusion, no resampling:

```
features   = model.featurize(structure)
coords     = model.initialize_from_prior(ensemble_size, features)      # Gaussian noise
processed  = process_structure_to_trajectory_input(...)                # → reconciler + reward_inputs
schedule   = sampler.compute_schedule(num_steps)
for i in range(starting_step, num_steps):
    context = sampler.get_context_for_step(i, schedule)                 # StepParams(t, dt, noise_scale)
    if i >= guidance_start:
        context = context.with_reward(reward, reward_inputs)           # attach reward
    context = context.with_reconciler(reconciler, alignment_reference) # attach alignment
    out = sampler.step(coords, model, context,
                       scaler=step_scaler if guiding else None,
                       features=features)
    coords = out.state                                                  # advance
```

**`FKSteering.sample()`** ([core/scalers/fk_steering.py:67](../src/sampleworks/core/scalers/fk_steering.py))
— Feynman-Kač steering. Same skeleton but with **`num_particles`** whole ensembles evolving
in parallel and periodic **resampling** toward low loss:

- `_run_step()` ([:225](../src/sampleworks/core/scalers/fk_steering.py)) — with no guidance,
  runs all particles in one batched `sampler.step`; **with** guidance, loops per particle so
  each gets its own gradient (required for correct FK weights).
- `_should_resample()` ([:352](../src/sampleworks/core/scalers/fk_steering.py)) — fires every
  `resampling_interval` steps while noise > 0.
- `_resample_particles()` ([:367](../src/sampleworks/core/scalers/fk_steering.py)) — weights
  `log_G = fk_lambda·(loss_prev − loss_curr) + log_proposal_correction`, then
  `softmax → multinomial` to duplicate/drop whole ensembles. "Particles" are ensembles, not
  single structures.
- Returns the single lowest-loss particle ensemble.

**`LatentOptimization.sample()`** ([core/scalers/latent_optimization.py:123](../src/sampleworks/core/scalers/latent_optimization.py))
— inference-time latent optimization (IT-opt). Instead of steering coordinates, it makes the
model's cached trunk latents (`s`, `z`) optimizable leaves, runs one or more full diffusion
rounds updating them against the reward, then samples the final ensemble with the latents
frozen. Coordinate guidance is not applied, so `step_scaler` is ignored. See
[IT_OPT_DESIGN.md](IT_OPT_DESIGN.md) for the algorithm and its mapping to the reference.

All three call `process_structure_to_trajectory_input()`
([eval/structure_utils.py:106](../src/sampleworks/eval/structure_utils.py)), which cleans the
atom array, builds the **`AtomReconciler`**, tiles coordinates across the batch, and returns a
frozen `SampleworksProcessedStructure`. Its `.to_reward_inputs()` produces the `RewardInputs`.

### 4.2 The sampler — `AF3EDMSampler.step()`

File: [core/samplers/edm.py](../src/sampleworks/core/samplers/edm.py). This is the
Karras-EDM sampler (Karras et al. 2022, the Euler variant) as used in AlphaFold3 —
`step()` follows AF3 Supplementary Algorithm 18. `EDMSamplerConfig` defaults match the AF3
parameterization *except* `gamma_min = 0.2` (AF3 uses 1.0). One `step()`
([:363](../src/sampleworks/core/samplers/edm.py)):

1. `check_context()` — validate `t, dt, noise_scale` present.
2. Center coords; if `augmentation`, apply a random SO(3) rotation + translation
   (`create_random_transform`).
3. Add stochastic noise: `noisy = augmented + eps·noise_scale`; set
   `noisy.requires_grad_(scaler.requires_gradients)`.
4. **`x̂₀ = model_wrapper.step(noisy, t_hat, features)`** — the model denoises. *This is the
   only call into the neural network per step.*
5. **Align** `x̂₀` into the experimental frame: if a reconciler is present,
   `reconciler.align()`; else `align_to_reference_frame()`. Noise is carried into the aligned
   frame via `transform_coords_and_noise_to_frame()` (rotation-only for the noise part, since
   noise is translation-invariant).
6. Compute drift `delta = (noisy − x̂₀) / t_hat`.
7. If a scaler is passed, `_apply_scaler_guidance()` ([:301](../src/sampleworks/core/samplers/edm.py)):
   calls `scaler.scale(x̂₀, context.with_metadata({"x_t": noisy}), model)` → guidance
   direction + loss, scales by `guidance_strength()`, rotates the direction into the aligned
   frame, optionally rescales to the diffusion magnitude, folds into `delta`, and computes the
   `log_proposal_correction`.
8. Euler step: `next = noisy + step_scale·dt·delta`.
9. Return `SamplerStepOutput(state, denoised=x̂₀, loss, log_proposal_correction)`.

The schedule (`EDMSchedule`, [:31](../src/sampleworks/core/samplers/edm.py)) is precomputed
once by `compute_schedule()` ([:210](../src/sampleworks/core/samplers/edm.py)) using the EDM
sigma schedule `σ = σ_data·(s_max^{1/p} + t·(s_min^{1/p} − s_max^{1/p}))^p` over
`num_steps+1` points; `gamma` (stochastic churn) is applied only where `σ > gamma_min`.
`get_context_for_step()` is an O(1) lookup that packages `t_hat, dt, eps_scale` into a
`StepParams`.

### 4.3 Step scalers — the per-step guidance rule

File: [core/scalers/step_scalers.py](../src/sampleworks/core/scalers/step_scalers.py). All
implement `StepScalerProtocol.scale(state, context, model) → (guidance_direction, loss)`.

- **`DataSpaceDPSScaler`** ([:51](../src/sampleworks/core/scalers/step_scalers.py)) — enable
  grad on `x̂₀`, compute `loss = reward(x̂₀, …)`, backprop → `∂loss/∂x̂₀`. Cheap; no backprop
  through the model. `requires_gradients = False`.
- **`NoiseSpaceDPSScaler`** ([:100](../src/sampleworks/core/scalers/step_scalers.py)) — reads
  `context.metadata["x_t"]` (the noisy state), computes `loss = reward(x̂₀)`, backprops
  **through the model** to `∂loss/∂x_t`. More faithful DPS; `requires_gradients = True` (which
  is what makes the sampler set `requires_grad` on the noisy state in step 3 above).
- **`NoScalingScaler`** ([:33](../src/sampleworks/core/scalers/step_scalers.py)) — returns
  zeros (baseline/unguided).

Both DPS variants optionally normalize the gradient. `guidance_strength()` returns the
per-step weight (`step_size`).

---

## 5. The reward and forward model (coords → scalar → gradient)

This is where "fit to experiment" becomes a differentiable number.
Files: [core/rewards/real_space_density.py](../src/sampleworks/core/rewards/real_space_density.py)
and [core/forward_models/xray/real_space_density.py](../src/sampleworks/core/forward_models/xray/real_space_density.py).

```
RealSpaceRewardFunction.__call__(coords, elements, b_factors, occupancies)   [rewards/…:291]
    │
    ├─ DifferentiableTransformer.forward(coords, …)                          [forward_models/…:501]
    │     ├─ _compute_radial_densities()   → per-atom radial density profiles
    │     │     └─ torch.vmap over unique (element, b_factor) pairs
    │     │           └─ GaussLegendreQuadrature ∫ scattering_integrand(s) ds
    │     ├─ _compute_grid_coordinates()   → Cartesian → fractional → grid
    │     ├─ dilate points onto the grid   → CUDA kernel (dilate_atom_centric)
    │     │                                   or pure-torch (dilate_points_torch + scatter_add_)
    │     └─ apply crystallographic symmetry (R,t per space-group op; F.grid_sample)
    │          → density grid  [batch, Dz, Dy, Dx]
    ├─ .sum(0)                              → collapse batch → single grid
    └─ self.loss(density, xmap.array)       → L1/L2 vs. experimental map → SCALAR
```

Every operation is autograd-tracked, so `.backward()` yields `∂reward/∂coords` — the signal
the DPS scalers turn into guidance. Key details:

- **`precompute_unique_combinations()`** ([rewards/…:228](../src/sampleworks/core/rewards/real_space_density.py))
  runs `torch.unique` *outside* vmap to avoid dynamic shapes; the results are passed in so the
  vmapped radial integration sees only static shapes.
- **Crystallographic symmetry** is applied by the forward model, not the reward — the model
  predicts a P1 asymmetric unit but the map is in the full crystal frame. Two equivalent
  paths: CUDA (symmetry applied to atoms before dilation) or CPU (`XMap_torch.apply_symmetry`
  after dilation via `grid_sample`).
- `setup_scattering_params(em_mode)` ([rewards/…:25](../src/sampleworks/core/rewards/real_space_density.py))
  picks X-ray Cromer-Mann (`ATOM_STRUCTURE_FACTORS`) vs. electron
  (`ELECTRON_SCATTERING_FACTORS`) coefficients from the qFit dependency.
- Two extraction helpers sit alongside the reward: the module-level
  `extract_density_inputs_from_atomarray` ([rewards/…:69](../src/sampleworks/core/rewards/real_space_density.py))
  and the method `RealSpaceRewardFunction.structure_to_reward_input`
  ([rewards/…:258](../src/sampleworks/core/rewards/real_space_density.py)) — both turn an atom
  array / structure dict into the element/B-factor/occupancy/coord tensors the reward consumes.

**`real_space_density_deps/`** is vendored crystallography (from qFit): `spacegroups.py`
(space-group operators, 8k lines), `sf.py` (scattering factors + structure-factor calc),
`unitcell.py`, `volume.py` (`XMap`, CCP4 I/O), `transformer.py`, plus `utils/quadrature.py`
(Gauss-Legendre) and `ops/dilate_points_cuda.py` (the custom autograd + vmap CUDA kernel).

---

## 6. Model wrappers (the pluggable priors)

Files: [models/](../src/sampleworks/models/). Each wraps an external model behind
`FlowModelWrapper`. Common shape:

- **`featurize(structure)`** — convert the atomworks dict into the model's native input,
  run the expensive trunk/pairformer **once**, and cache it in a model-specific `Conditioning`
  dataclass. Also loads the **model-space atom array** for reconciliation.
- **`step(x_t, t, features)`** — run only the diffusion/structure module on the cached
  conditioning; return predicted clean coords. Called every diffusion step.
- **`initialize_from_prior(batch_size, features)`** — Gaussian noise at the model's atom count.

| Wrapper | File | Notes |
|---------|------|-------|
| `Boltz1Wrapper` / `Boltz2Wrapper` | [boltz/wrapper.py](../src/sampleworks/models/boltz/wrapper.py) | Caches pairformer (`s_trunk, z_trunk`); Boltz2 also caches diffusion conditioning. Preprocessing writes NPZ/manifest/MSA. `process_structure_for_boltz` reconstructs the atom array from the processed NPZ. |
| `ProtenixWrapper` | [protenix/wrapper.py](../src/sampleworks/models/protenix/wrapper.py) | AF3 reimpl. `structure_processing.py` builds the Protenix JSON (entities, modifications, covalent bonds). Optional diffusion-shared-vars cache (`pair_z, p_lm, c_l`). |
| `RF3Wrapper` | [rf3/wrapper.py](../src/sampleworks/models/rf3/wrapper.py) | Baker AF3 replica. Uses an inference engine + trunk-with-recycling generator. Optional chiral-feature tracking/disabling (writes `chiral_grad_stats.json`). |

The wrappers guarantee the atom array they hand back has valid coords/occupancy/B-factors, so
downstream `RewardInputs.from_atom_array()` never sees NaNs. Import failures are tolerated
([guidance_script_utils.py:51-75](../src/sampleworks/utils/guidance_script_utils.py)) because
Boltz/Protenix/RF3/Protpardelle have mutually incompatible dependencies and live in separate
pixi envs.

---

## 7. Alignment & atom reconciliation (the SE(3) glue)

Reward functions compare coordinates in the *fixed experimental frame*; models emit
arbitrary frames with possibly different atom sets. Two utilities bridge this — and the
**sampler**, not the reward, owns the timing (AGENTS.md "Alignment" pitfall).

- **`AtomReconciler`** ([utils/atom_reconciler.py](../src/sampleworks/utils/atom_reconciler.py))
  — `from_arrays(model_array, struct_array)` normalizes atom IDs
  (`chainidx_seqpos_atomname`, handling 0-based vs. author numbering), finds the common subset,
  and returns index maps (or `.identity()` when they match). `align()` computes a rigid
  transform on the common atoms and applies it to *all* model atoms. `struct_to_model()` maps
  structure coords into model space differentiably.
- **`weighted_rigid_align_differentiable()`** ([utils/frame_transforms.py:332](../src/sampleworks/utils/frame_transforms.py))
  — weighted Kabsch/Procrustes via SVD (float32 for stability, reflection-corrected),
  **preserving gradients** (unlike Boltz's detached version). `transform_coords_and_noise_to_frame()`
  ([:560](../src/sampleworks/utils/frame_transforms.py)) applies full transform to coords but
  rotation-only to noise.

---

## 8. Utils (the support layer)

Directory: [utils/](../src/sampleworks/utils/). Highlights beyond §3/§7:

| File | Role |
|------|------|
| `guidance_script_arguments.py` | `GuidanceConfig` (all run params, `from_cli()` two-pass parse), `JobResult`, `_resolve_checkpoint()` (env → baked → ACTL → legacy), `validate_model_checkpoint()`. `as_dict()` remaps container↔host paths. |
| `guidance_constants.py` | Enums: `GuidanceType`, `StructurePredictor`, `StepScalers`, `TrajectoryScalers`, `Rewards`. |
| `cif_utils.py` | `resolve_mixed_hetatm_atom_altlocs()`, `add_category_to_cif()` (writes the `sampleworks` metadata block). |
| `atom_array_utils.py` | `make_normalized_atom_id()`, `filter_to_common_atoms()` (used by the reconciler). |
| `density_utils.py` | `compute_density_from_atomarray()`, `build_density_transformer()` — reusable forward-model wrappers used by synthetic-data generation and RSCC eval. |
| `msa.py` / `mmseqs2.py` | `MSAManager` — SHA3-keyed MSA cache, ColabFold/Protenix-server/mmseqs2 fetch, per-model formatting (CSV for Boltz, A3M for RF3). |
| `elements.py` | `element_to_scattering_idx()` — element symbol → scattering-table index. |
| `frame_transforms.py` | Rigid-transform algebra (forward/inverse/apply, random augmentation). |
| `torch_utils.py` | `try_gpu()` — pick the least-loaded GPU via `nvidia-smi`. |
| `imports.py` | `BOLTZ_/PROTENIX_/RF3_AVAILABLE` flags + `@require_*` test decorators. |
| `protein_input.py` | CSV parser for batch protein specs. |

One package-level knob worth knowing: [`sampleworks/__init__.py`](../src/sampleworks/__init__.py)
reads the `NAN_CHECK` env var into `should_check_nans` (default on; set `NAN_CHECK=false`/`0`
to disable). [`torch_utils.py`](../src/sampleworks/utils/torch_utils.py) uses it to make
`assert_no_nans` either a real check or a no-op, gating expensive NaN assertions in hot paths.

---

## 9. Metrics & eval

**`metrics/`** — a pluggable metric framework used for validation/scoring:

- `metric.py` — `Metric` ABC (`compute()`, `kwargs_to_compute_args`) + `MetricManager`
  (tag-filtered batch computation).
- Concrete: `AllAtomLDDT` / `SelectedLDDT` ([lddt.py](../src/sampleworks/metrics/lddt.py)),
  `AllAtomRMSD` ([rmsd.py](../src/sampleworks/metrics/rmsd.py), optional Kabsch),
  `SidechainMetrics` ([sidechain_metrics.py](../src/sampleworks/metrics/sidechain_metrics.py),
  topology/bond/clash checks), `ExtraInfo` (metadata pass-through).
- `metric_utils.py` — binning/masking helpers for LDDT/PAE-style scores.

**`eval/`** — the offline evaluation and synthetic-data machinery:

- `structure_utils.py` — **`SampleworksProcessedStructure` + `process_structure_to_trajectory_input()`**
  (the bridge from §4.1 into `RewardInputs`), plus selection-string parsing and reference-structure loading.
- `eval_dataclasses.py` — `Trial`, `TrialList`, `ProteinConfig` (per-protein map/selection config, `from_csv()`).
- `grid_search_eval_utils.py` — `scan_grid_search_results()` walks a results tree of
  `refined.cif` files into `Trial`s; `setup_evaluation_parameters()` + `parse_eval_args()`
  standardize eval-script setup.
- `generate_synthetic_density.py` / `generate_synthetic_sf.py` — build synthetic maps/MTZs
  from a structure using the forward models (via `compute_density_from_atomarray` /
  `SFcalculator`); each has a CLI `main()` with single/batch modes.
- `metrics.py` — `rscc()` (real-space correlation coefficient).
- `occupancy_utils.py`, `synthetic_utils.py`, `constants.py` — altloc/occupancy handling for
  synthetic ensembles.

---

## 10. The two run paths, end to end

**Direct (`sampleworks-guidance`)** — [cli/guidance.py](../src/sampleworks/cli/guidance.py):
`GuidanceConfig.from_cli(argv)` → `get_model_and_device()` → `run_guidance()` → exit code. One
process, one run.

**Preset grid search (`sampleworks-runs`)** — [runs/](../src/sampleworks/runs/):
- `cli.py::main` → `run_cli()` with `EXPERIMENT_CLI_CONFIG` (presets in `experiments/`).
- `loader.py` reads a TOML preset (`_read_toml` → `_apply_overrides` (`--set a.b=c`) →
  `_resolve_variables` (`${VAR}`) → `_build_preset`), producing `Preset`/`Job` dataclasses
  ([schema.py](../src/sampleworks/runs/schema.py)).
- `runner.py::run` ([:619](../src/sampleworks/runs/runner.py)) resolves GPU assignments
  (`_resolve_gpu_assignments` via `nvidia-smi`), builds one `JobInvocation` per job
  (`_build_argv` picks the pixi-env Python and assembles the command), runs `pre_jobs`
  sequentially (`_run_sequential`), then **spawns main jobs in parallel** (`_spawn` →
  `subprocess.Popen` + `_tee` threads) and waits (`_wait_all`).
- The worker script defaults to `run_grid_search.py`, resolved by `_resolve_script_path`
  from the repository root (`./run_grid_search.py`), with container/workspace fallbacks
  (`/app/run_grid_search.py`, `/home/dev/workspace/run_grid_search.py`). A job may override it
  via `Job.script`.
- Each spawned `run_grid_search.py` process ultimately calls the same `run_guidance()`.
- `analysis_cli.py` reuses `run_cli()` with `ANALYSIS_CLI_CONFIG` (presets in `analyses/`) for
  post-hoc evaluation jobs.

So the grid-search path fans out into many subprocesses that each re-enter the spine of §3.

---

## 11. Where to start reading, by task

- **"How does one guided step work?"** → `AF3EDMSampler.step()` (§4.2), then a DPS scaler (§4.3).
- **"How is the loop orchestrated?"** → `PureGuidance.sample()` / `FKSteering.sample()` (§4.1).
- **"How does experiment become a gradient?"** → `RealSpaceRewardFunction.__call__` +
  `DifferentiableTransformer.forward` (§5).
- **"How do I add a model / reward / scaler?"** → satisfy the matching protocol in §2; see the
  AGENTS.md "Adding New Components" section.
- **"How is a whole run wired?"** → `_run_guidance()` (§3) is the single best file to read.
- **"Why is alignment everywhere?"** → §7 (rewards live in the crystal frame; models don't).
