# IT-Opt — Design (read this first)

Inference-time latent optimization (IT-opt) for sampleworks. This is the one document to read to
understand the feature: **what it is, the as-built algorithm, the components and where they live in
the code, per-model gradient readiness, and the design choices that make it correct.**

Companions (read only if you need them):
- [IT_OPT_TESTING.md](IT_OPT_TESTING.md) — how to run, debug, and verify it, plus the open problems.
- [IT_OPT_REFERENCE_NOTES.md](developer_notes/IT_OPT_REFERENCE_NOTES.md) — deep dive on the external `it_opt/`
  reference tree and its bug catalog (only relevant if you are re-porting from the reference).

---

## 1. What it is

A frozen structure predictor (Protenix / RF3 / Boltz) turns a sequence into cached post-trunk
latents — the **single** representation `s` and the **pair** representation `z` — and a diffusion
module decodes those latents into coordinates. IT-opt treats the model as a differentiable sampler
and **optimizes `s` and `z` themselves** (not the coordinates, not the weights) against an
experimental reward evaluated on the *denoised* structure. No model weights are trained.

It is a `TrajectoryScalerProtocol`, a peer of `PureGuidance` and `FKSteering`, living in
[core/scalers/latent_optimization.py](../src/sampleworks/core/scalers/latent_optimization.py). It is
a faithful port of the reference `run_it_optimization`, with the reference's bugs fixed (§6).

**v1 is latent-only.** Coordinate-space guidance is *not* applied — the attached step-scaler
returns a zero coordinate direction (§4, `_GradEnablingScaler`), so the only steering comes from the
evolving latents.

## 2. The as-built loop

```
extract (s, z) from the trunk once  ->  clone into optimizable leaves (requires_grad=True)
for each outer round (fresh prior noise each round):
    optimizer = Adam([s, z])                      # ONE persistent Adam per round
    for each diffusion step:
        x̂₀   = differentiable denoise(x_t, s, z)  # ONE forward, run under autograd
        loss  = reward(x̂₀) + anchor(s, z) + bond_geometry(x̂₀)
        loss.backward();  clip s and z INDEPENDENTLY;  optimizer.step()
        x_t   = step_output.state.detach()         # advance; coordinate graph cut here
final clean sampling pass with the optimized latents  ->  saved ensemble
```

The gradient that matters is `∂reward(x̂₀)/∂latent` through **one** denoiser forward. Because the
coordinate state is detached every step (`coords = step_output.state.detach()`), the gradient never
flows through the coordinate recursion — it is a greedy, per-step latent gradient, **not**
backprop-through-sampling. (Unrolling the full sampler is intractable: the EDM sampler re-noises and
detaches each step, and a true unroll is O(steps·N²·d) memory.)

## 3. Components and where they live

| Component | File | What it is |
|---|---|---|
| `LatentOptimization` | [core/scalers/latent_optimization.py](../src/sampleworks/core/scalers/latent_optimization.py) | The scaler / entry point. Owns the loop, the ensemble, and the final sampling pass. |
| `LatentAnchor` | same file | `Σ wᵢ·mean((latentᵢ − baselineᵢ)²)` — mean-squared drift from the baseline latents. Regularizer, added to the loss directly (not a reward). |
| `_GradEnablingScaler` | same file | Minimal `StepScalerProtocol` that only turns autograd on (§4). |
| `BondGeometryReward` | [core/rewards/geometry.py](../src/sampleworks/core/rewards/geometry.py) | Bond-length + steric-clash hinges on `x̂₀`. Built only when `bond_length_weight > 0`. Bounded hinges (not the reference's exploding `exp(relu(...))`). |
| `AttrLatentIO` / `LatentIO` | [models/latent_adapter.py](../src/sampleworks/models/latent_adapter.py) | Reads/writes `s`/`z` by attribute name; writes via `dataclasses.replace` (honors `frozen=True, slots=True`). The only model-specific knowledge is a pair of attribute names. |
| CLI wiring | [utils/guidance_script_utils.py](../src/sampleworks/utils/guidance_script_utils.py) (`_run_guidance`), [utils/guidance_script_arguments.py](../src/sampleworks/utils/guidance_script_arguments.py) (`add_latent_opt_args`), [utils/guidance_constants.py](../src/sampleworks/utils/guidance_constants.py) (`GuidanceType.LATENT_OPT`) | Builds the scaler from CLI flags. |

Private helpers, for orientation: `_leaf_latents` (promote s/z to leaves), `_optimize_one_round`
(one outer round), `_latent_adam_step` (score → backward → clip → step), `_sample_with_frozen_latents`
(final clean pass).

### Constructor contract

```python
LatentOptimization(
    ensemble_size=1, num_steps=200, guidance_t_start=0.0, *,
    outer_steps=1, learning_rate=0.05, max_grad_norm=1.0,
    optimize_single=True, optimize_pair=True,
    anchor_weight_single=0.0, anchor_weight_pair=0.0,
    bond_length_weight=0.0, single_attr="s", pair_attr="z",
)
```

- `ensemble_size` structures are sampled in parallel and **share** the latents (per-member latents
  are a documented follow-up).
- `guidance_t_start` is a fraction in `[0,1]`; stored as `guidance_start = int(guidance_t_start*num_steps)`.
  Steps before it are plain frozen-latent diffusion.
- `outer_steps` resample rounds, fresh prior noise each. **Constructor default `1`; CLI default `2`.**
- `learning_rate` Adam LR; one persistent Adam is built **once per round**.
- `max_grad_norm` clips `s` and `z` **independently** (see §6 — a joint clip starves `s`).
- `single_attr`/`pair_attr` default to Boltz names; the wiring always passes model-resolved names.
- The `metadata` on the returned `GuidanceOutput` carries `"optimization_losses"` (per-round,
  per-step data losses) and `"latent_drift"` (per-round relative L2 drift of each latent). It does
  **not** emit the siblings' `"trajectory_denoised"` — treat scaler-specific keys as optional.

## 4. The gradient gate

The load-bearing mechanism. `AF3EDMSampler.step` reads `getattr(scaler, "requires_gradients", False)`;
if true it runs the denoiser under `torch.set_grad_enabled(True)` and returns a `denoised` (`x̂₀`)
that still carries a graph back to the latent leaves:

```python
class _GradEnablingScaler:
    requires_gradients = True
    def scale(self, state, context, *, model=None):
        return torch.zeros_like(state), torch.zeros(state.shape[0], device=state.device)
```

The zero direction means the trajectory advance is unguided; the flag is the *only* thing that turns
autograd on. `requires_gradients` is duck-typed (read via `getattr`, not declared on the protocol).
The sampler does **not** re-detach `state`/`denoised` — the scaler loop detaches between iterations.

The denoised `x̂₀` is frame-aligned to the experimental reference by the sampler (via the
`AtomReconciler` + `alignment_reference`) *before* the reward sees it, so reward functions can assume
pre-aligned coordinates.

## 5. Per-model gradient readiness

The only model-specific concern is whether the cached latents can receive a gradient. The optimized
attributes are `s`/`z` for Boltz and `s_trunk`/`z_trunk` for Protenix/RF3
(`DEFAULT_SINGLE_REP_ATTR` / `DEFAULT_PAIR_REP_ATTR` in `latent_adapter.py`).

| Model | Status | Note |
|---|---|---|
| **Boltz1** | Clean | Trunk latents reach the diffusion module directly. |
| **RF3** | Clean | Same. |
| **Protenix** | Works, with two conditions | (a) `step()` keeps an injected leaf attached automatically via the `detach_unless_leaf` helper — a cached latent detaches under grad (avoids double-backward), but a leaf with `requires_grad=True` is kept attached. No manual edit needed. (b) For `z`, the diffusion shared-vars cache must be **off** (`pair_z`/`p_lm`/`c_l` are z-derived; a stale cache silently zeroes / corrupts the `z` gradient). `_run_guidance` disables it for `LATENT_OPT`. |
| **Boltz2** | Not yet | `z` (and part of `s`) are baked into a cached `diffusion_conditioning` at featurize time; needs the conditioning recomputed from the live latents. |

All four models' conditioning is a `@dataclass(frozen=True, slots=True)`, swapped via
`dataclasses.replace` — hence `AttrLatentIO`.

## 6. Design choices and invariants

**Kept from the reference (the algorithm):** `s`/`z` as the optimized variables with `s_inputs`
frozen; one Adam step per diffusion step on the denoised `x̂₀`; latents persist across steps and
rounds; an outer resample loop; an anchor to the trunk baseline; a final clean sampling pass.

**Fixed from the reference:**
- **Persistent Adam.** The reference rebuilt the optimizer *inside* the step loop, degenerating it to
  `lr·sign(grad)` (signSGD) — "not actually running Adam," which also made its LR and grad-clip
  inert. The port builds one Adam per round.
- **Independent per-latent clip.** `z`'s gradient is ~1e4× `s`'s, so a single joint `[s, z]` clip is
  set by `z` and starves `s`. Clipping them separately decouples `s`'s step from `z`'s scale (Adam's
  per-parameter normalization is what actually commensurates them). A joint clip was tried and
  reverted 2026-07-20.
- **One forward per step.** The reference did a second `no_grad` forward to advance; the port advances
  with `step_output.state.detach()` from the same differentiable step.
- Bounded geometry hinges instead of the reference's exploding `exp(relu(...))` clash term; no
  hardcoded save-step.

**Deliberate deviations:** the objective is a pluggable `RewardFunctionProtocol` (v1:
`RealSpaceRewardFunction`, density fit) rather than backbone-RMSD; the anchor is mean-squared rather
than the reference's Frobenius norm (shape-agnostic, so `w_s`/`w_z` are comparable); latents are
shared across the ensemble in v1.

**Invariants worth stating:**
- **Optimization is not sampling.** An optimized `(s, z)` is a point estimate; reporting it as a
  single state collapses the Boltzmann/population weighting. If populations matter, sample the latent
  rather than optimizing it. Ensembles here come from fresh prior noise per member, not from a
  posterior over latents.
- **On-manifold discipline.** Pushing `z` hard buys density fit with broken geometry; the anchor and
  `BondGeometryReward` are the counter-pressure. Efficacy must be judged on held-out fit against
  matched-compute baselines, not train-set loss (see [IT_OPT_TESTING.md](IT_OPT_TESTING.md)).

## 7. Why `s`/`z` and not the MSA

In the AF2/OpenFold tradition you could optimize the MSA representation `m`. The AF3-family models
(Protenix, Boltz) have **no persistent MSA latent** at the featurize→step boundary: the MSA module
writes only into `z`, and `s` is updated inside the Pairformer via pair-biased attention, seeded from
`s_inputs` rather than from an MSA row. So the post-trunk optimization levers are exactly `s_trunk`
(≈ the single rep) and `z_trunk` (≈ where the MSA information now lives). `m` is upstream of
featurize, rebuilt each recycle, and the trunk runs under `no_grad` — optimizing it would require
backprop through the trunk (ColabDesign/AfDesign territory) and is out of scope.

## 8. Running it

CLI: `sampleworks-guidance --model <protenix|rf3|boltz1> --guidance-type latent_opt …` with
`--which-latent {single,pair,both}` (default `pair`), `--learning-rate` (0.05), `--outer-steps` (2),
`--anchor-weight` (0.0), `--max-grad-norm` (1.0), `--bond-length-weight` (5e-5). Also reachable
programmatically through `run_guidance`. `_run_guidance` resolves the model name to the latent
attribute names and raises a clear `ValueError` for an unsupported model. See
[IT_OPT_TESTING.md](IT_OPT_TESTING.md) for the debug ladder and gradient check.
