---
name: model-wrapper
description: Build or revise PyTorch FlowModelWrappers for diffusion and flow models in sampleworks. Use when integrating a model or compatible checkpoints, mapping upstream initialization, sampling, and conditioning APIs, or diagnosing atom-ordering, coordinate/latent gradients, precision, cache, or unguided prediction parity problems.
---

# Build a sampleworks FlowModelWrapper

Follow this order: **goal → constraints → context gathering → evaluation**.
Establish evaluation criteria before implementing. Keep the wrapper independent of
particular guidance algorithms so future frameworks can consume it without model
code changes.

## 1. Goal

**First, validate that the target really is a diffusion or flow model.** Confirm
this from upstream documentation and its generative sampling implementation:
identify the evolving state, noise/time parameterization, and network evaluation
used by the solver. A class exposing `step()` or passing a runtime protocol check
does not establish compatibility. If it is not a diffusion/flow model, stop and
explain that it is outside this skill rather than forcing it into the interface.

Adapt that PyTorch model to `FlowModelWrapper`, preserving its unguided predictions
and supporting many compatible checkpoints through the same wrapper. The purpose
is experimental guidance, but the first acceptance gate is low-RMSD reproduction
of predictions from the installed model package, without guidance.

Before coding, establish with the user:

- Target model family, upstream version/commit, supported molecular inputs, and
  the set of compatible checkpoints/configurations to make plug and play.
- **Whether ITO (inference-time latent optimization) compatibility is required.**
  Require an explicit decision early; do not infer it from `PureGuidance` support
  or defer it until caching and conditioning have already been implemented.
  If required, identify which model latents will receive gradients.
- Available compute, weights, native CLI commands, prediction fixtures, and the
  low-RMSD acceptance thresholds and atom selections for baseline comparisons.
- Model-specific failures or assumptions the user knows but the repository does not
  record. Read existing context first; ask only about unresolved decisions.

Use three reviewable commits, following repository commit rules:

1. **Baseline fixtures and tests:** run the installed model package, save native
   predictions and reproduction metadata, and define comparison tests before
   implementing the wrapper. Record expected failures while the wrapper is absent.
2. **Wrapper implementation:** reuse upstream APIs, support compatible checkpoints,
   and pass unguided parity, representation, and differentiability checks,
   including latent access/gradient/cache checks when ITO is required.
3. **CLI integration and guidance tests:** wire the validated wrapper into
   sampleworks and validate `PureGuidance`, plus ITO if requested. Test FK steering
   only as a requested follow-up; guidance algorithms do not belong in the wrapper.

The initial implementation scope is the wrapper and validation that it runs
without guidance similarly to native predictions under a fixed seed where possible.
Complete that gate before proceeding to CLI/guidance work; do not expand into
framework redesigns. Produce a short plan recording the upstream functions to
reuse, initialization dataclass, checkpoint coverage, ITO decision, native
conventions, and acceptance checks for each commit. Follow the user's authorization
for committing or proceeding to later stages.

## 2. Constraints

- Read applicable `AGENTS.md`/`AGENT.md`. Use structural `typing.Protocol`
  interfaces, not a new wrapper inheritance hierarchy.
- Keep model adaptation in the wrapper, numerical integration in the sampler, and
  experimental objectives/guidance in losses/scalers. Reuse Atomworks, Biotite,
  Gemmi, and existing reconciliation/array utilities rather than rebuilding them.
- Do not put experimental-frame alignment or loss computation inside the wrapper.
  The sampler owns alignment timing and differentiable frame transforms.
- Prefer upstream featurizers and the native single-evaluation network entry point.
  Do not invoke an entire upstream sampling pipeline from a diffusion `step()`.
  Avoid modifying upstream source; if necessary, explain the limitation and agree
  on a narrow, versioned adaptation rather than silently patching an installation.
- Put initial model configuration and runtime setup in `__init__`, parametrized
  by a dataclass containing the relevant options. Keep it a thin orchestration of
  the original package's configuration/model loaders and standard PyTorch setup
  for weights, device, dtype, and evaluation mode. Do not reimplement model
  construction, checkpoint parsing, or upstream configuration logic unnecessarily.
- Keep structure-dependent feature/embedding production in `featurize()`, not in
  initialization. Do not defer model loading or change architecture/runtime
  settings inside `step()`. Distinguish initialization configuration from per-input
  featurization configuration; reuse existing dataclasses where they fit.
- Use the appropriate Pixi environment. Model dependencies are isolated and may be
  incompatible. Do not make shared modules require every model package at import time.
- Use PyTorch. Preserve the denoiser's gradient path to noisy coordinates for
  guidance; do not detach live inputs to make backward failures disappear.
  When ITO is required, also preserve gradients to the selected latents and
  invalidate/recompute conditioning derived from them. Evaluation mode is not the
  same as disabling autograd.
- Use jaxtyping annotations with meaningful axis names and prefer `einx` for
  tensor operations introduced by the wrapper. Follow repository NumPy-style
  docstrings and side-effect documentation.
- Do not hard-code one checkpoint's dimensions or inference configuration.
  Derive them from upstream configuration/APIs; validate and warn clearly about
  incompatible variants rather than silently ignoring mismatches.
- Match the native floating-point environment deliberately: parameter/input dtype,
  autocast, TF32/matmul precision, kernels, device, and dependency versions.
  Do not silently change precision to pass a test or fit a model into memory.
- Use behavioral tests with realistic inputs; avoid mocks of internal call chains.
  Mark checkpoint/GPU tests `slow` and GPU tests additionally `gpu`.
- Missing hardware, weights, or data means **unverified**, not passed. Do not claim
  support solely because protocol checks or toy-model tests pass.

## 3. Gather context

### Read the live contract and its consumers

Start with these repository-relative paths; inspect the relevant symbols instead
of loading every implementation:

| Question | Source |
|---|---|
| Exact method signatures and feature container? | `src/sampleworks/models/protocol.py`; `tests/models/test_model_wrapper_protocol.py` |
| How are state, prior, schedule, and gradients consumed? | `src/sampleworks/core/samplers/edm.py`; `src/sampleworks/core/scalers/{pure_guidance,step_scalers}.py` |
| How do topology, reference coordinates, and loss inputs meet? | `src/sampleworks/utils/{structure_utils,atom_reconciler,frame_transforms}.py`; `src/sampleworks/core/losses/protocol.py` |
| How does a user actually select and run the model? | `src/sampleworks/utils/{guidance_constants,guidance_script_arguments,guidance_script_utils,setup}.py`; `src/sampleworks/runs/` |
| Where are shared fixtures, environments, and checks? | `tests/conftest.py`; `pyproject.toml`; `.pre-commit-config.yaml`; `.github/workflows/` |

Verify the current `FlowModelWrapper` signatures:
`featurize(structure)`, `step(x_t, t, *, features=None)`, and
`initialize_from_prior(batch_size, features=None, *, shape=None)`.
`step()` returns a **predicted clean sample**, not an integrated next state.
Noise/velocity/score parameterizations generally need only a simple model-specific
transformation to clean coordinates; reuse the upstream conversion/preconditioning
and verify its convention rather than treating the parameterization as a blocker.

### Inspect initialization, then trace one evaluation and its inputs

Read the existing configuration dataclasses and wrapper constructors using
[model-notes.md](model-notes.md). Identify which upstream configuration objects,
loaders, inference engines, and standard PyTorch operations can do the initial
setup. Design the initialization dataclass around actual required options,
not hypothetical flexibility; retain upstream defaults and checkpoint-specific
configuration rather than copying architecture settings into the wrapper.

1. Locate the fundamental network evaluation used for one sampling step in the
   upstream repository. Record its symbol, source location, arguments, outputs,
   and surrounding preconditioning/conversions. Distinguish that evaluation from
   the solver update, which sampleworks supplies.
2. Identify everything it consumes: noisy coordinates/time, masks, topology,
   conditioning, upstream features, embeddings, trunk outputs, and any history.
   For each, record shape/axis meanings, dtype/device, producer function/source,
   and whether sampleworks or the installed model package provides it.
3. Trace producers back to the native input pipeline. Identify callable upstream
   APIs first, then existing sampleworks helpers. Import and call those functions
   instead of reimplementing them. If orchestration must be copied because no
   callable boundary exists, identify the exact minimal source fragment and pinned
   version; keep upstream computations as imported calls.
4. For each producer and cache, record the required differentiability: what may be
   computed once as a constant, what must remain connected to noisy coordinates,
   and what must be recomputed if its inputs change. Inspect inference decorators,
   kernels, conversions, and in-place operations along the actual path.
5. If ITO is required, trace each chosen latent through the denoiser and all derived
   caches. Establish how the scaler can read/replace those latents using
   `src/sampleworks/models/latent_adapter.py`; do not assume every model exposes
   single/pair representations. Surface an adapter mismatch before implementing
   a new abstraction. Read `docs/IT_OPT_DESIGN.md` and `docs/IT_OPT_TESTING.md` now,
   not only during later guidance testing.
6. Trace loading/configuration across compatible checkpoints. Separate genuine
   architecture incompatibilities from supported weight/config variants, and
   record a checkpoint coverage matrix with tested versus unverified entries.
7. Trace the native CLI's sampling settings and random-number consumption.
   Determine which schedule, prior, self-conditioning, augmentation, and precision
   settings must match, and whether the built-in sampler is actually compatible.
   Raise any sampler gap separately rather than hiding a solver in the wrapper.

Use [model-notes.md](model-notes.md) to choose the closest existing implementation
and read only the applicable model sections. Read [gotchas.md](gotchas.md) before
coding. Record applicable hazards and the upstream/repository code being reused.
For stage 3 only, trace model selection, constructor, structure annotation,
checkpoint resolution, sampler configuration, test registration, and output writing.

## 4. Evaluation

Agree on fixtures, metrics, tolerances, and compute budget before inspecting the
results. Validate in increasing cost order; stop and diagnose the first failing
layer rather than tuning guidance around an incorrect model.

### A. Native baseline fixtures and tests first

Run the installed model package through its own CLI, not through sampleworks or
the wrapper under test. For example, a BioEmu integration starts with a prediction
from BioEmu's CLI using recorded parameters. Save that actual prediction as a test
fixture with enough metadata to reproduce it:

- Exact command, input files/hashes, upstream package version/commit, checkpoint
  identity/hash and configuration, and selected output/sample.
- Seeds for relevant RNGs, sample/batch counts, schedule/step count, and sampling,
  self-conditioning, and augmentation settings.
- Hardware/device, dependency versions, dtype/autocast, matmul/TF32 precision,
  deterministic settings, and known nondeterministic kernels.

Prefer small, redistributable fixtures under `tests/resources/` and comparison
tests under `tests/models/<model>/`. Do not commit checkpoint weights, secrets,
or private/licensing-restricted inputs. Keep baseline generation separate from
tests; never regenerate an expected prediction using the wrapper being tested.

Define the atom correspondence, rigid alignment, RMSD atom selections and low-RMSD
thresholds before examining wrapper results. Use the strongest feasible assertions:
matched seeds/configuration and close numerical agreement where reproducible,
otherwise a justified low-RMSD tolerance. Exact equality is not a blanket requirement.
Align only to remove arbitrary rigid frames, not to conceal atom permutations,
reflections, missing coverage, or internal structural differences.

Record unavoidable mismatches in RNG consumption, configuration, or floating-point
environment. A fixed seed alone does not establish identical stochastic inputs.
If those differences prevent a meaningful low-RMSD comparison, stop and agree on
how to reproduce the baseline more closely; do not silently relax the threshold.
Use representative compatible checkpoints; report tested and unverified coverage
separately rather than extrapolating one checkpoint's success to all weights.

### B. Wrapper contract and representation

- Verify initialization from the configuration dataclass and compatible checkpoint
  variants, including the resulting device/dtype and evaluation mode. Check that
  model-wide settings are resolved during construction, not changed by denoising.
- Verify feature container, missing-input errors, explicit prior shape
  `(num_atoms, 3)`, shape precedence, dtype/device, and batch sizes 1 and >1.
- Exercise float, scalar tensor, and per-member time inputs where the consumer
  requires them, using low and high noise values from the actual schedule.
- Assert exact unpadded output shape and finite values. Verify **atom identities
  and ordering**, not just counts, through layout conversions, reconciliation,
  loss input construction, and serialized structure round trips.
- Cover supported multichain/non-protein cases and observed-vs-full-sequence
  inputs, including missing residues, terminal atoms, and invalid reference atoms.
  Unsupported cases must have a clear, documented boundary with relevant warnings;
  do not silently drop unsupported input content.

Reuse/extend `tests/models/test_model_wrapper_protocol.py` via the fixtures and
`MODEL_WRAPPER_REGISTRY` in `tests/conftest.py`. Add focused tests under
`tests/models/<model>/`; do not mistake the shared shape tests for complete coverage.
Use `tests/integration/test_mismatch_integration.py` for reconciliation patterns.

### C. Unguided prediction parity and differentiability

- With real weights and matched features, coordinates, noise level, and settings,
  compare the wrapper's single evaluation against the upstream entry point.
  Control randomness and reset mutable state; compare in a common frame if needed.
  Specify numerical tolerances and explain any intentional difference.
- Run sampleworks without guidance using the intended sampler and compare the
  final structure to the saved native CLI fixture. Require low RMSD under the
  agreed atom selections and coverage, with matched seed/configuration wherever
  feasible. Single-evaluation agreement does not replace final prediction parity.
- Check geometry as well as finiteness: appropriate bond lengths, clashes,
  chirality, chain continuity, and supported molecular topology. Reuse existing
  metrics and `tests/integration/test_no_guidance_geometry.py` where applicable;
  a protein-only bond check cannot validate ligand or nucleic-acid support.
- Backpropagate a meaningful scalar through the **real** denoiser to `x_t`.
  Check finite, nonzero gradients on a nondegenerate input,
  and compare a directional derivative against finite differences with stochastic
  behavior controlled. Repeat forward/backward using the same conditioning
  without `retain_graph=True` as a workaround.
- When ITO is required, inject optimizable latent leaves through the existing
  adapter, check finite/nonzero gradients for every requested latent, and verify
  that perturbing each changes the output with derived caches recomputed.
  Compare directional derivatives and repeat backward passes with controlled
  stochastic state. A nonzero latent gradient alone does not prove cache correctness.
  Freezing model weights must not disable gradients to those latents.
- Exercise independent structures and compatible checkpoints on one wrapper.
  For stateful models, test repeated evaluations, history reset, and changes in
  batch size. History must follow the right member, not leak across runs.
- Do not add loss-to-denoised-coordinate/data-space DPS tests to wrapper validation;
  that path is owned outside the wrapper.

### D. CLI and guidance after the unguided gate

In the third commit, add the necessary CLI wiring and run the actual sampleworks
CLI unguided against the baseline. Then validate `PureGuidance` with a small real
experimental input and an unguided control, checking fit and geometry. Keep
guidance code in the existing sampler/scaler layers.

If ITO was required at the outset, validate that workflow too, using the latent
access and cache behavior established in stage 2. If not required, report it as
outside the tested scope. FK steering remains a separate requested follow-up:
test particle branching/resampling and history ownership through that framework,
not a new wrapper-specific guidance implementation.

### E. Operational checks and handoff

Run lint/format/type checks routed to the model's environment. Run focused fast
tests before expensive checkpoint/GPU tests, then the CLI workflow in stage 3.
Read current Pixi task definitions when forwarding arguments; do not assume a test
selector selected the new model.

For example, after replacing `<model>` and `<selector>` with real values:

```bash
pixi run -e <model>-dev tests "tests/models/<model> tests/models/test_model_wrapper_protocol.py -k <selector>"
pixi run -e <model>-dev all-tests "tests/models/test_model_wrapper_protocol.py tests/integration/test_no_guidance_geometry.py -k <selector>"
```

The second command includes expensive tests and requires suitable hardware/weights.
Check collected cases, skips, and deselections; an unavailable wrapper may never be
registered. Use a platform-supported environment (`boltz-osx` exists for macOS),
or report that the required environment is unavailable.

If performance is part of acceptance, measure featurization separately from
per-step time and peak memory, both with and without gradients. Look for repeated
trunk work or growth across repeated steps; do not invent universal budgets.

Finish with evidence: upstream version, initialization configuration, checkpoint
compatibility/test matrix, ITO decision and verified latent paths when applicable,
baseline commands and fixture metadata, selected tests, final RMSD and geometry,
precision/configuration differences, guidance results only if run, performance if
measured, and every unrun or blocked check. Keep
“interface implemented,” “native behavior verified,” and “guided workflow verified”
as distinct claims.
