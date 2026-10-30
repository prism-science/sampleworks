# AGENTS.md

This file provides guidance to AI coding agents working with the sampleworks codebase. It covers the design philosophy, architectural principles, and domain context needed to make informed contributions.

## Code Style

Code should be direct, readable, and maximize clarity without verbosity. Name variables well. Write code that is functional and direct. Only comment when truly necessary (but ALWAYS annotate complex array shapes and note side effects). Always include NumPy-style docstrings for every function and class.

Respond in a measured, clear tone. Consider alternatives carefully. Include confidence estimates for claims (e.g., "I am about 75% confident").

Code reuse is paramount. Whenever possible, locate high-quality implementations in this codebase or high-quality open-source implementations for algorithms and use those instead of implementing something yourself. Chances are someone has already solved the problem, no need to reinvent the wheel. In particular, check whether something from atomworks, biotite, or gemmi could be used in place of a custom function.

## Project Overview

**sampleworks** is a Python framework for guiding generative biomolecular structure models with experimental data. It bridges the gap between structure prediction (single-state ML models) and experimental reality (thermodynamic ensembles).

**The core insight**: Structure prediction models like Boltz, AlphaFold, and RosettaFold capture aspects of the underlying distribution of realistic macromolecular structures, but collapse ensembles to single states. By treating these models as physics-informed samplers and applying experimental constraints and guidance during generation, we can recover the conformational ensemble present in the experiment.

**The core problem solved**: Without sampleworks, integrating N generative models with M experimental data types requires O(N×M) bespoke implementations. Sampleworks reduces this to O(N+M) through protocol-driven decoupling, allowing thorough evaluation, benchmarking, and comparison between models, guidance algorithms, and types of experimental data.

### Atomworks

[Atomworks](https://baker-laboratory.github.io/atomworks-dev/latest/) (built on Biotite) handles structure I/O. A `structure: dict` parameter anywhere in sampleworks is the dict returned by `atomworks.parse()`, whose `"asym_unit"` key holds a Biotite `AtomArray`/`AtomArrayStack`. To load a `.pdb`/`.cif` straight to an `AtomArray`, use `atomworks.io.utils.io_utils.load_any()`.

Because we're dealing with ensemble models here, we need to make sure we properly handle altlocs and multimodel structures at input, inference, and output. We have a variety of utility functions in `atom_array_utils`, `cif_utils`, and `structure_utils` to handle these cases. If you notice we need to extend these, block and notify about what is needed so we can carefully consider the best approach.

## Design Philosophy

### 1. Protocols Over Inheritance

Component interfaces are `typing.Protocol`s rather than base classes. Model wrappers (Boltz, Protenix, RF3, Protpardelle) are plain classes that inherit from nothing in sampleworks; they satisfy the protocols structurally. This keeps the wrapper classes free of a shared base-class hierarchy, so the code that wraps an external model's source (which we avoid modifying) does not need to subclass anything. Wrappers still import small sampleworks helpers and types, such as `GenerativeModelInput` from `models/protocol.py`.

### 2. Decouple Guidance from Generation

The separation of **Guidance** (Scalers/Losses) from **Generation** (ModelWrappers/Samplers) is the architectural core. A loss function is written once and applied to any supported generative model — the inverse-problem paradigm: define a forward model and target loss, optimize during guidance.

### 3. Restraints vs. Constraints

Following Chroma, **restraints** (soft) are additive potentials that bias the distribution ("fit this density map"), while **constraints** (hard) are geometric projections on the coordinates ("enforce C3 symmetry"). `StepScalerProtocol` implements per-step restraints/constraints; `TrajectoryScalerProtocol` implements population-level steering.

### 4. Ensemble as First-Class Citizen

Ensembles are managed _outside_ the model, by `TrajectoryScalerProtocol`. Current models aren't ensemble-native, so we run the generative process many times under experimental guidance and reweight trajectories by fit to data (e.g., Feynman-Kaç steering).

### 5. Time-Dependent Conditioning

For FlowModelWrappers, guidance must be time-aware: we may want to anneal potentials with noise level (global fold early, local chemistry late), gate terms that only make sense at certain stages (e.g., downsample a 1 Å target map while the structure is still an atom cloud), and keep guidance from swamping the model prior. `StepParams` carries the time ( t ∈ [1, 0]), which it may transform into the appropriate parameter (e.g. `t_hat`) for the model `step` and other per-step information scalers need.

### 6. Differentiability End-to-End

Gradients must flow from the experimental observable back to atomic coordinates (or arbitrary internal representations in the model, potentially all the way back to input features/embeddings), so every forward model (density, structure factors) must be differentiable.

### 7. Atomic Responsibility

One loss per experimental mismatch, one scaler per guidance strategy, one sampler per numerical solver, one wrapper per generative model. Compose; don't bundle.

## Architecture

### Data Flow

`TrajectoryScalerProtocol.sample()` owns a run end to end:

```
featurize(structure) → initialize_from_prior() → prepare loss (if preparable)
for each step in sampler.compute_schedule():
    StepParams ← sampler.get_context_for_step()
    sampler.step(): model.step() → x̂₀ (or other coordinate output for non-FlowModelWrappers) → StepScaler.scale() (once i ≥ guidance_start) → update rule → next state
    (FK steering: reweight/resample the population)
→ ensemble
```

### Key Protocols

**ModelWrapper** (`models/protocol.py`): `StructureModelWrapper`, `FlowModelWrapper` (diffusion/flow-matching), and `EnergyBasedModelWrapper`. `featurize()` converts atomworks structures to model input; `step()` runs one forward pass. `FlowModelWrapper` adds `initialize_from_prior()`. `FlowModelWrapper.step()` returns the predicted clean sample `x̂₀`; the other wrappers' `step()` returns coordinates directly.

**Sampler** (`core/samplers/protocol.py`): `step()` advances state by one iteration. `TrajectorySampler` adds `compute_schedule()` and `get_context_for_step()` for diffusion/flow time management.

**Scaler** (`core/scalers/protocol.py`): `StepScalerProtocol.scale()` returns a guidance direction + loss. `TrajectoryScalerProtocol.sample()` orchestrates full trajectory generation with population-level control.

**LossFunction** (`core/losses/protocol.py`): `LossFunctionProtocol` defines a callable computing scalar loss from coordinates. `PrecomputableLossFunctionProtocol` extends it with `precompute_unique_combinations()` for vmap compatibility. `LossInputs` dataclass bundles pre-extracted inputs (elements, b_factors, occupancies, coords).

### Reference Implementations

Paths are relative to `src/sampleworks/`.

- `core/losses/real_space_density.py` and `core/forward_models/xray/real_space_density.py`: reference loss and its differentiable forward model
- `core/scalers/step_scalers.py`: `DataSpaceDPSScaler`, `NoiseSpaceDPSScaler`
- `core/scalers/pure_guidance.py` (reference trajectory scaler) and `core/scalers/fk_steering.py`
- `models/boltz/wrapper.py`: reference model wrapper (pairformer caching, MSA management, atom reconciliation)
- `utils/atom_reconciler.py`: atom-count mismatches and differentiable alignment
- `utils/guidance_script_utils.py`: `run_guidance()`, which wires all components together

## Running Guidance Pipelines

Use the unified `sampleworks-guidance` CLI to run guidance with any supported model and trajectory scaler:

```bash
pixi run -e boltz sampleworks-guidance \
    --model boltz2 \
    --guidance-type pure_guidance \
    --protein 1VME \
    --model-checkpoint ~/.boltz/boltz2_conf.ckpt \
    --output-dir output/boltz2_pure_guidance \
    --structure tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif \
    --density tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4 \
    --resolution 1.8 \
    --ensemble-size 4 \
    --guidance-start 0.65 \
    --augmentation --align-to-input
```

Run `sampleworks-guidance --model <model> --guidance-type <type> --help` to see all available options.

### Choosing losses

`--loss-type` selects which loss guides the run (default `real_space_density`), and the
options listed under it in `--help` are that loss's own — `--density`/`--resolution` above
belong to the density loss, while `--loss-type structure_factor` takes `--mtzfile` and
friends instead. Passing one loss's option to another is an error rather than a silent
no-op.

To combine losses, or to keep a run's loss settings in version control, pass a
configuration file instead (YAML, JSON, or TOML; `${oc.env:VAR}` interpolation works in YAML):

```yaml
# losses.yaml -- sampleworks-guidance ... --loss-config losses.yaml
real_space_density:
  weight: 0.4
  loss_options:
    density: /data/1vme.ccp4
    resolution: 1.8
structure_factor:
  weight: 0.6
  loss_options:
    mtzfile: /data/1vme.mtz
    bulk_solvent: combined
```

Weights default to `1/N`, so combining losses does not change what the guidance step size
means. Omit them all or give them all; a partly-weighted configuration is an error.

**Adding a loss type** is three things and no argparse edits: implement the loss in
`core/losses/`, declare its options as a frozen dataclass in `core/losses/options.py`, and
register it in `core/losses/registry.py` with a `build_*` function that raises its own
errors for inputs it cannot do without. CLI flags, configuration-file schema, validation
messages, and run metadata all follow from that one declaration. A loss that needs the
model's atom ordering (structure factors, anything with a topology) implements
`prepare(loss_inputs, *, device)` from `PreparableLossFunctionProtocol`; the trajectory
scalers call it once the model atom array exists.

The `run_guidance()` function in `utils/guidance_script_utils.py` is the central orchestrator. It wires together the model wrapper, sampler (`AF3EDMSampler`), step scaler (`DataSpaceDPSScaler` or `NoiseSpaceDPSScaler`), trajectory scaler (`PureGuidance` or `FKSteering`), and loss function. When adding a new model or guidance strategy, this is the best reference for how components compose in practice.

## Development Environment

Dependencies are managed with [Pixi](https://pixi.sh). The model wrappers (Boltz, Protenix, RF3, Protpardelle) have mutually incompatible dependencies, so each model has its own `<model>` and `<model>-dev` environment; test in the matching `-dev` one. Other environments: `analysis[-dev]`, `boltz-analysis`.

| Command                           | Runs                                                |
| --------------------------------- | --------------------------------------------------- |
| `pixi run -e <env>-dev tests`     | Fast tests (`-m 'not slow and not gpu'`)            |
| `pixi run -e <env>-dev all-tests` | All tests, including slow and GPU                   |
| `pixi run -e <env>-dev gpu-tests` | GPU tests only (needs CUDA)                         |
| `pixi run test-fast` / `test-all` | `tests` / `all-tests` across every model `-dev` env |

Forward pytest options as one argument: `pixi run -e rf3-dev tests "-k integration"`.

Mark tests that need a GPU or model checkpoint with `@pytest.mark.slow`, and tests that need CUDA with `@pytest.mark.gpu` (on CPU they crash or exhaust memory).

Pre-commit hooks (ruff, ty, toml-sort, commitizen) run through [prek](https://github.com/j178/prek) and block commits on failure. `ty` is split per environment; see `.pre-commit-config.yaml` for the file routing. On macOS the protenix `ty` hook is skipped (protenix has no macOS build); CI still checks it.

```bash
pixi run -e boltz-dev prek install                          # git hook
pixi run -e boltz-dev prek install --hook-type commit-msg   # commit-message check
pixi run -e boltz-dev prek run -a                           # all hooks, all files
```

## Commits, PRs, and Releases

Releases are automated by [python-semantic-release](https://python-semantic-release.readthedocs.io/) on merge to `main`; never bump versions or edit `CHANGELOG.md` by hand. Commits must follow [Conventional Commits](https://www.conventionalcommits.org/). PRs are squash-merged, so the **PR title** becomes the commit on `main` and decides the version bump. Keep each PR to one logical change.

| Type                                               | Version bump |
| -------------------------------------------------- | ------------ |
| `feat`                                             | Minor        |
| `fix`, `perf`                                      | Patch        |
| `docs`, `refactor`, `test`, `chore`, `ci`, `style` | None         |

Breaking changes (`feat!:`, `refactor(scalers)!:`) bump the minor version while we're on 0.x (`major_on_zero = false`).

## Testing Philosophy

Write black-box tests that verify **behavior**, not implementation. Test public interfaces with realistic inputs. Verify outputs match contracts — shapes, value ranges, mathematical properties.

Avoid using mocks at all costs. If you find yourself wanting to mock, ask: can I test the expected behavior directly instead? Mocking internal methods creates brittle tests that break on refactor and don't verify real functionality.

```python
# GOOD: Verifies expected behavior analytically
def test_step_denoises_toward_clean_structure(wrapper, features, noisy_coords, clean_coords):
    output = wrapper.step(noisy_coords, t=0.5, features=features)
    initial_rmsd = compute_rmsd(noisy_coords, clean_coords)
    output_rmsd = compute_rmsd(output, clean_coords)
    assert output_rmsd < initial_rmsd


# BAD: Tests implementation details
def test_wrapper_calls_internal_method():
    with mock.patch.object(wrapper, "_internal_compute") as m:
        wrapper.step(...)
        m.assert_called_once()  # Breaks on refactor
```

Test structure: `tests/{cli,eval,forward_models,integration,metrics,mocks,models,losses,runs,synthetic,utils}/`, alongside `tests/resources/` for fixture files.

Mark any test that requires a GPU or model checkpoint with `@pytest.mark.slow` so it is excluded from fast CI runs:

```python
import pytest


@pytest.mark.slow
def test_boltz_full_inference(boltz_wrapper, features): ...
```

## Implementation Patterns

- **Immutable state**: frozen dataclasses with `with_*` functional updates (see `StepParams`).
- **Caching**: compute conditioning and `featurize()` output once per structure; cache pairformer output across denoising steps. Any cache derived from a representation that a later step changes (e.g., diffusion conditioning built from `s`/`z`) must be invalidated or disabled for that representation, or the change silently has no effect. Inference-time latent optimization (IT-opt, `core/scalers/latent_optimization.py`) is a targeted break of this pattern: it overwrites the cached `s`/`z` each step and needs Boltz2's diffusion conditioning recomputed and Protenix's diffusion cache disabled for `z`.
- **Gradients**: detach cached representations and other tensors that do not need gradients when gradients are enabled, to avoid double-backward errors. Methods that optimize a cached representation are a targeted exception: the optimized leaf must stay attached to the denoiser graph (Protenix keeps it attached via `detach_unless_leaf`), so detach everything else at the boundary rather than the leaf itself.
- **Shapes**: annotate array shapes with jaxtyping, e.g. `Float[Tensor, "batch atoms 3"]`.

## Adding New Components

Satisfy the matching protocol (see Key Protocols) in `models/`, `core/losses/`, `core/scalers/`, or `core/samplers/`; differentiable physics goes in `core/forward_models/`. Start from the reference implementations above. New loss types also need registering; see "Adding a loss type".

## Common Pitfalls

### Alignment and Atom-Count Mismatches

Most losses (e.g., real-space density) are **not** SE(3)-invariant: they compare coordinates in the experimental map frame, while models generate in an arbitrary frame and often with different atoms than the input structure (missing hydrogens, extra OXT, different residue coverage). `AtomReconciler` (`utils/atom_reconciler.py`) handles both: `from_arrays()` builds bidirectional index maps over the common atom subset, and `align()` does a differentiable rigid alignment on it.

- **The sampler owns alignment.** `AF3EDMSampler` aligns at each step using `StepParams.alignment_reference` and the reconciler. Losses assume pre-aligned coordinates; don't align inside losses or scalers. New samplers or trajectory scalers must align before the loss is evaluated.
- **Build `LossInputs` from the model atom array**, not the input structure (`SampleworksProcessedStructure.to_loss_inputs()` in `utils/structure_utils.py`). `PreparableLossFunctionProtocol` losses are prepared with those same `LossInputs`, never an atom array; `LossInputs.to_atom_array()` rebuilds a reference `AtomArray` (model topology, reconciled coordinates and B-factors) for forward models that need one. Don't write reconciled values back into the model atom array.

## Avoiding Technical Debt

- Fix root causes, not symptoms
- Follow existing patterns — check how similar problems are solved first (like we noted in the Code Style section, chances are someone has already solved the problem)
- No dead code, no compatibility shims for hypothetical users
- Type errors are real issues. Use `cast()` or `# ty:ignore[...]` with explanatory comments
- Fail fast with clear messages

## Domain Context

### Why Ensembles Matter

Proteins exist as thermodynamic ensembles, not static structures. Current generative models collapse this to single low-energy states. Sampleworks recovers the posterior distribution by treating generation as Bayesian inference — the model is the prior, experimental data defines the likelihood through differentiable score functions, and guided sampling draws from the posterior. This enables:

- **Ensemble refinement**: Fit multi-conformer ensembles to heterogeneous cryo-EM or X-ray density, rather than a single best-fit structure
- **Guided ensemble generation**: Sample de novo conformational populations conditioned on experimental observables
- **Multi-modal data fusion**: Combine multiple experimental data types as composable likelihood terms

### Experimental Data Types

Currently planned:

- Real-space electron density (X-ray crystallography) _implemented_
- Cryo-EM density _implemented_
- Structure factors (reciprocal space) _implemented_
- Diffuse scattering
- Cryo-EM image stacks

### Symmetry

Crystallographic symmetry is handled natively in forward models. Most ML models operate in P1 (asymmetric unit), but experimental maps are in the full crystal frame. The forward models bridge this gap.
