# AGENTS.md

This file provides guidance to AI coding agents working with the sampleworks codebase. It covers the design philosophy, architectural principles, and domain context needed to make informed contributions.

## Code Style

Code should be direct, readable, and maximize clarity without verbosity. Name variables well. Write code that is functional and direct. Only comment when truly necessary (but ALWAYS annotate complex array shapes and note side effects). Always include NumPy-style docstrings for every function and class.

Respond in a measured, clear tone. Consider alternatives carefully. Include confidence estimates for claims (e.g., "I am about 75% confident").

Code reuse is paramount. Whenever possible, locate high-quality implementations in this codebase or high-quality open-source implementations for algorithms and use those instead of implementing something yourself. Chances are someone has already solved the problem, no need to reinvent the wheel.

## Project Overview

**sampleworks** is a Python framework for guiding generative biomolecular structure models with experimental data. It bridges the gap between structure prediction (single-state ML models) and experimental reality (thermodynamic ensembles).

**The core insight**: Structure prediction models like Boltz, AlphaFold, and RosettaFold capture aspects of the underlying distribution of realistic macromolecular structures, but collapse ensembles to single states. By treating these models as physics-informed samplers and applying experimental constraints and guidance during generation, we can recover the conformational ensemble present in the experiment.

**The core problem solved**: Without sampleworks, integrating N generative models with M experimental data types requires O(N×M) bespoke implementations. Sampleworks reduces this to O(N+M) through protocol-driven decoupling.

### Atomworks

[Atomworks](https://baker-laboratory.github.io/atomworks-dev/latest/) is sampleworks' core dependency for structure I/O and representation. Atomworks is built atop biotite, a bioinformatics Python library. It provides:

- **`atomworks.parse()`**: The universal entry point for loading structure files (`.cif`, `.pdb`). Returns a dictionary containing an `"asym_unit"` key with a Biotite `AtomArray` or `AtomArrayStack`, plus metadata. This dictionary is the standard structure representation passed to `ModelWrapper.featurize()`.
- **`AtomArray` / `AtomArrayStack`** (from [Biotite](https://www.biotite-python.org/)): Per-atom annotations (element, residue ID, chain ID, B-factor, occupancy, coordinates). `AtomArrayStack` is the multi-model variant used for ensembles.
- **`atomworks.ml`**: ML utilities used by model wrappers for featurization.

Whenever you see a `structure: dict` parameter in sampleworks, it refers to an atomworks-parsed dictionary. Use `atomworks.parse()` to create one from a file, and use `load_any()` to load a `.pdb` or `.cif` to an AtomArray or AtomArrayStack.

Additionally, if you need to modify or transform coordinates or other atomic annotations, search Atomworks and biotite for the appropriate tools.

## Design Philosophy

### 1. Protocols Over Inheritance

All interfaces use `typing.Protocol` for structural subtyping. This is a deliberate choice for this domain:

- **Duck typing of external models**: We wrap models (Boltz, Protenix, RF3) from external codebases where modifying source is infeasible. Protocols let any class with matching methods participate — no inheritance required.
- **Natural composition**: Objects can satisfy multiple interfaces without diamond inheritance problems.
- **Minimal coupling**: Contributors don't need to import our framework to build compatible wrappers.

```python
class MyProtocol(Protocol):
    def method(self, x: Tensor) -> Tensor: ...


# Any class with matching signature works
class MyImpl:
    def method(self, x: Tensor) -> Tensor:
        return x * 2


assert isinstance(MyImpl(), MyProtocol)  # True if @runtime_checkable decorates MyProtocol
```

### 2. Decouple Guidance from Generation

The separation of **Guidance** (Scalers/Rewards) from **Generation** (ModelWrappers/Samplers) is the architectural core. A reward function (e.g., real-space density fit) is written once and applied to any supported generative model. This maps to the inverse problem paradigm: define a forward model and target reward, optimize during guidance.

### 3. Restraints vs. Constraints

Two types of conditioning, borrowed from Chroma's philosophy:

- **Restraints (Soft)**: Modify the energy/loss landscape. "Make it fit this density map." Implemented as additive potentials — biases the probability distribution while the model balances the prior (protein realism) with the condition.
- **Constraints (Hard)**: Modify coordinates directly. "Enforce C3 symmetry." Implemented as geometric projections — restricts the sampling manifold by construction.

Scalers implement both: `StepScalerProtocol` for per-step restraints/constraints, `TrajectoryScalerProtocol` for population-level steering.

### 4. Ensemble as First-Class Citizen

Ensembles are managed *outside* the model (`TrajectoryScalerProtocol`). Current SOTA models aren't ensemble-native, so we recover distributions by running the generative process multiple times under experimental constraints — reweighting trajectories based on fit to data (e.g., Feynman-Kaç steering).

At its core this is Bayesian inference: the generative model provides the prior over structures, and experimental data defines the likelihood via differentiable score functions. By sampling from the posterior, we perform experimentally-constrained ensemble generation — producing populations weighted by data that reveal cryptic pockets or dynamic loops invisible to single-structure refinement.

### 5. Time-Dependent Conditioning

Diffusion/flow models operate over time t ∈ [1, 0]. Effective conditioning requires time-awareness:

- **Annealing**: Scale potentials by signal-to-noise ratio. At high noise (t ≈ 1), strong gradients guide global fold. At low noise (t ≈ 0), subtle gradients preserve local chemistry.
- **Gating**: Some constraints only apply at specific stages. E.g., don't optimize fit of an unstructured atom cloud to a 1 Å map — downsample the target map according to timestep.
- **Scale awareness**: If auxiliary energy swamps the base model, you get geometric garbage that satisfies the condition. If too weak, the condition is ignored. `StepParams` bundles time and other step-specific info for scalers to use.

### 6. Differentiability End-to-End

Gradient-based guidance requires differentiability from experimental observable back to atomic coordinates. The forward models (density calculation, structure factors) must be differentiable. If coordinates or internal representations can't receive gradients, the potential can't guide the structure.

### 7. Atomic Responsibility

Each component does one thing well:
- A reward function computes one experimental mismatch
- A scaler applies one guidance strategy
- A sampler implements one numerical solver
- A model wrapper adapts one generative model

Don't bundle logic. Compose instead.

## Architecture

### Component Hierarchy

```
sampleworks
├── ModelWrappers — featurize structures, run model forward passes
│   ├── StructureModelWrapper
│   ├── FlowModelWrapper (diffusion/flow-matching)
│   └── EnergyBasedModelWrapper
├── Samplers — numerical solvers for sampling
│   └── TrajectorySampler (EDM)
├── Scalers — guidance strategies
│   ├── StepScalerProtocol — per-step (DPS, Tweedie)
│   └── TrajectoryScalerProtocol — population-level (PureGuidance, FK steering)
└── Rewards — experimental data fit via differentiable forward models
    └── RewardFunctionProtocol
```

### Data Flow

```
Atomworks Structure → ModelWrapper.featurize() → Features
                                                    ↓
                    ModelWrapper.step() → Denoised prediction
                                                    ↓
┌────────────────── Sampling Loop ──────────────────┐
│  Schedule → StepParams (t, dt, reward, etc.)     │
│                     ↓                             │
│  Sampler.step(state, model, context, scaler)      │
│       ├── Model forward pass                      │
│       ├── StepScaler.scale() → guidance signal    │
│       └── Update rule → next state                │
└───────────────────────────────────────────────────┘
                         ↓
              TrajectoryScaler (optional reweighting/resampling)
                         ↓
                   Final Ensemble
```

### Key Protocols

**ModelWrapper** (`models/protocol.py`): `featurize()` converts atomworks structures to model input; `step()` runs one forward pass. `FlowModelWrapper` adds `initialize_from_prior()`. All `step()` methods return predicted clean samples.

**Sampler** (`core/samplers/protocol.py`): `step()` advances state by one iteration. `TrajectorySampler` adds `compute_schedule()` and `get_context_for_step()` for diffusion/flow time management.

**Scaler** (`core/scalers/protocol.py`): `StepScalerProtocol.scale()` returns a guidance direction + loss. `TrajectoryScalerProtocol.sample()` orchestrates full trajectory generation with population-level control.

**RewardFunction** (`core/rewards/protocol.py`): `RewardFunctionProtocol` defines a callable computing scalar reward from coordinates. `PrecomputableRewardFunctionProtocol` extends it with `precompute_unique_combinations()` for vmap compatibility. `RewardInputs` dataclass bundles pre-extracted inputs (elements, b_factors, occupancies, coords).

### Module Layout

```
src/sampleworks/
├── core/
│   ├── forward_models/    # Differentiable physics (X-ray density, cryo-EM)
│   ├── rewards/           # Loss functions for experimental data fit
│   ├── scalers/           # Guidance strategies (DPS, FK steering)
│   └── samplers/          # Numerical solvers (EDM)
├── models/                # Generative model wrappers (Boltz, Protenix, RF3)
├── metrics/               # Quality metrics (LDDT, sidechain)
├── eval/                  # Evaluation utilities
├── synthetic/             # Synthetic data generation
├── data/                  # Reference data (protein configs)
├── runs/                  # `sampleworks-runs` CLI + TOML preset orchestrator
└── utils/                 # Shared utilities
```

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
    --guidance-start 130 \
    --augmentation --align-to-input
```

Run `sampleworks-guidance --model <model> --guidance-type <type> --help` to see all available options.

### Choosing rewards

`--reward-type` selects which reward guides the run (default `real_space_density`), and the
options listed under it in `--help` are that reward's own — `--density`/`--resolution` above
belong to the density reward, while `--reward-type structure_factor` takes `--mtzfile` and
friends instead. Passing one reward's option to another is an error rather than a silent
no-op.

To combine rewards, or to keep a run's reward settings in version control, pass a
configuration file instead (YAML, JSON, or TOML; `${oc.env:VAR}` interpolation works in YAML):

```yaml
# rewards.yaml -- sampleworks-guidance ... --reward-config rewards.yaml
real_space_density:
  weight: 0.4
  reward_options:
    density: /data/1vme.ccp4
    resolution: 1.8
structure_factor:
  weight: 0.6
  reward_options:
    mtzfile: /data/1vme.mtz
    bulk_solvent: combined
```

Weights default to `1/N`, so combining rewards does not change what the guidance step size
means. Omit them all or give them all; a partly-weighted configuration is an error.

**Adding a reward type** is three things and no argparse edits: implement the reward in
`core/rewards/`, declare its options as a frozen dataclass in `core/rewards/options.py`, and
register it in `core/rewards/registry.py` with a `build_*` function that raises its own
errors for inputs it cannot do without. CLI flags, configuration-file schema, validation
messages, and run metadata all follow from that one declaration. A reward that needs the
model's atom ordering (structure factors, anything with a topology) implements
`prepare(atom_array, *, device)` from `PreparableRewardFunctionProtocol`; the trajectory
scalers call it once the model atom array exists.

The `run_guidance()` function in `utils/guidance_script_utils.py` is the central orchestrator. It wires together the model wrapper, sampler (`AF3EDMSampler`), step scaler (`DataSpaceDPSScaler` or `NoiseSpaceDPSScaler`), trajectory scaler (`PureGuidance` or `FKSteering`), and reward function. When adding a new model or guidance strategy, this is the best reference for how components compose in practice.

## Development Environment

**Package Manager**: [Pixi](https://pixi.sh) for cross-platform dependency management.

```bash
pixi install          # Install dependencies
pixi shell            # Activate environment
pixi run test-fast    # Run fast tests across all model dev environments
pixi run test-all     # Run all tests (including slow tests) across all dev environments
pixi run -e boltz-dev tests  # Run fast tests in specific environment
```

**Environments**: `default`, `boltz[-dev]`, `boltz-analysis`, `protenix[-dev]`, `rf3[-dev]`, `analysis[-dev]`

Model wrappers for Boltz, Protenix, and RF3 have mutually incompatible dependencies — each lives in its own pixi environment. Use the appropriate `-dev` environment for testing.

**Test tasks** (defined in `pyproject.toml` under `[tool.pixi.tasks]`):

| Task | Command | Description |
|------|---------|-------------|
| `tests` | `pixi run -e <env>-dev tests` | Fast tests only (`-m 'not slow'`), single env |
| `all-tests` | `pixi run -e <env>-dev all-tests` | All tests including slow (GPU/weights), single env |
| `test-fast` | `pixi run test-fast` | Fast tests across all three model dev envs |
| `test-all` | `pixi run test-all` | All tests across all three model dev envs |

The `tests` and `all-tests` tasks accept a `flags` argument for forwarding pytest options:

```bash
pixi run -e boltz-dev tests -- -k integration   # Run only integration tests
pixi run -e rf3-dev tests -- -x                 # Stop on first failure
```

The `@pytest.mark.slow` marker gates tests that require a GPU or model checkpoint files. Fast test runs (`tests` / `test-fast`) skip these automatically.

**Pre-commit hooks**: ruff (lint/format), ty (type checking, per-environment), toml-sort. Hooks block commits on failure. We use [prek](https://github.com/j178/prek) (a Rust-reimplementation of pre-commit) as the hook runner.

```bash
pixi run -e boltz-dev prek install      # Install prek as a git hook
pixi run -e boltz-dev prek run -a       # Run all hooks on all files
```

Note: `ty` type checking is split per environment — Boltz files are checked in `boltz-dev`, Protenix files in `protenix-dev`, RF3 files in `rf3-dev`. See `.pre-commit-config.yaml` for the file routing rules.

## Release Process

Releases are fully automated via [python-semantic-release](https://python-semantic-release.readthedocs.io/) (PSR v10). No manual version bumps or changelog edits are needed.

### Conventional Commits (Required)

All commit messages **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<optional scope>): <summary>

[optional body]

[optional footer(s)]
```

**Types and their effect on versioning:**

| Type | Description | Version bump |
|------|-------------|-------------|
| `feat` | New feature | Minor (0.x.0) |
| `fix` | Bug fix | Patch (0.x.x) |
| `docs` | Documentation only | None |
| `refactor` | Code change (no new feature or fix) | None |
| `test` | Adding/updating tests | None |
| `chore` | Maintenance (CI, deps, tooling) | None |
| `perf` | Performance improvement | Patch (0.x.x) |

**Breaking changes** (append `!` after type, e.g. `feat!:`) bump the minor version while we're in 0.x (`major_on_zero = false`).

**Examples:**
```bash
feat(rewards): add cryo-EM image stack reward function
fix(boltz): correct atom reconciler index mapping for OXT atoms
docs: update AGENTS.md with new model wrapper example
refactor(scalers)!: rename StepScaler to StepGuide
chore(ci): pin Docker build action to v5
```

A **commitizen** pre-commit hook validates commit messages locally. Install it with:
```bash
pixi run -e boltz-dev prek install --hook-type commit-msg
```

### How Releases Work

1. Push/merge to `main` with `feat:` or `fix:` commits
2. The **Release** workflow runs PSR, which:
   - Analyzes commits since the last tag
   - Determines the version bump (or skips if no releasable commits)
   - Updates `version` in `pyproject.toml`
   - Updates `CHANGELOG.md`
   - Creates a version commit and `v{version}` tag
   - Pushes the commit and tag
3. The **Docker** workflow triggers on the new `v*.*.*` tag and builds images tagged with the version

### What Does NOT Trigger a Release

Commits with types `docs`, `refactor`, `test`, `chore`, `ci`, or `style` do not bump the version. They will appear in the next changelog under their respective sections when a releasable commit is included.

### Version Strategy

We use `major_on_zero = false`, meaning breaking changes bump minor (not major) while the version is 0.x. This will change when we decide to release 1.0.

### Merge Strategy

Use **squash merge** for all PRs. This collapses a PR's commits into a single commit on `main`, with the PR title as the commit message. The PR title **must** follow Conventional Commits format and controls the version bump:

- `feat(rewards): add cryo-EM image stack reward` → minor bump
- `fix(boltz): correct OXT atom mapping` → patch bump
- `docs: update installation guide` → no release

This keeps the changelog clean (one entry per PR), the version history predictable (one potential bump per PR), and requires no discipline around individual commit messages during development. PRs should be focused on a single logical change — avoid PRs that bundle unrelated features and fixes.

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

Test structure: `tests/{rewards,integration,mocks,models,utils,metrics,eval}/`

Mark any test that requires a GPU or model checkpoint with `@pytest.mark.slow` so it is excluded from fast CI runs:

```python
import pytest


@pytest.mark.slow
def test_boltz_full_inference(boltz_wrapper, features): ...
```

## Implementation Patterns

### Immutable State

Frozen dataclasses with functional updates:

```python
@dataclass(frozen=True)
class State:
    value: Tensor

    def with_value(self, new_value: Tensor) -> "State":
        return State(new_value)
```

### Caching

- **Conditioning**: Compute once, flow through trajectory
- **Features**: Cache `featurize()` output when structure unchanged
- **Pairformer**: Cache encoder output across denoising steps (Boltz/Protenix/RF3)

### Gradient Control

- **Detach cached representations** when gradients are enabled to avoid double-backward errors.

### Type Annotations

Use jaxtyping for array shapes:

```python
from jaxtyping import Float
from torch import Tensor


def process(coords: Float[Tensor, "batch atoms 3"]) -> Float[Tensor, "batch atoms 3"]: ...
```

## Adding New Components

1. **Model wrapper**: Implement the appropriate `ModelWrapper` protocol in `models/`
2. **Reward function**: Implement `RewardFunctionProtocol` in `core/rewards/`
3. **Scaler**: Implement `StepScalerProtocol` or `TrajectoryScalerProtocol` in `core/scalers/`
4. **Sampler**: Implement `Sampler` or `TrajectorySampler` protocol in `core/samplers/`
5. **Forward model**: Implement differentiable physics in `core/forward_models/`

All use structural typing — no inheritance needed. Just satisfy the protocol interface.

### Example: Adding a New Model Wrapper

A `FlowModelWrapper` needs three methods: `featurize()`, `step()`, and `initialize_from_prior()`. The minimal skeleton:

```python
# models/my_model/wrapper.py


class MyModelWrapper:
    """Wrapper for MyModel generative model."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.model = load_my_model(checkpoint_path).to(device)

    def featurize(self, structure: dict) -> GenerativeModelInput:
        """Convert atomworks structure dict to model-specific features."""
        atom_array = structure["asym_unit"]
        # ... model-specific featurization ...
        conditioning = my_model_features(atom_array)
        return GenerativeModelInput(conditioning=conditioning)

    def step(
        self,
        x_t: Tensor,
        t: Float[Array, "*batch"],
        *,
        features: GenerativeModelInput | None = None,
    ) -> Tensor:
        """Denoise x_t at timestep t → predicted clean structure x̂_θ."""
        return self.model(x_t, t, features.conditioning)

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Sample from the prior (typically Gaussian noise)."""
        n_atoms = features.conditioning.num_atoms if features else shape[-2]
        return torch.randn(batch_size, n_atoms, 3, device=self.device)
```

See `models/boltz/wrapper.py` for a production reference with pairformer caching, MSA management, and atom reconciliation.

## Common Pitfalls

### Alignment and SE(3) Invariance

Most reward functions (e.g., real-space density fit) are **not** SE(3)-invariant — they compare coordinates in a fixed reference frame (the crystallographic or cryo-EM map frame). But generative models produce structures in an arbitrary frame. This means:

- **Structures must be aligned** to the experimental reference frame before computing rewards. The `AtomReconciler` (`utils/atom_reconciler.py`) handles this, computing rigid alignment on the common atom subset between model and structure representations.
- **Atom count mismatches** are common. A model's internal representation may have different atoms than the input structure (e.g., missing hydrogens, extra OXT atoms, different residue coverage). `AtomReconciler.from_arrays()` detects this and provides bidirectional index mappings.
- **Alignment must be differentiable** when using gradient-based guidance. `AtomReconciler.align()` uses `weighted_rigid_align_differentiable()` from `utils/frame_transforms.py` to preserve gradients through the alignment step.
- **The sampler handles alignment timing**. `AF3EDMSampler` uses the `alignment_reference` field in `StepParams` and the reconciler to align at each step. Don't add alignment logic inside reward functions or scalers.

When writing new reward functions, assume coordinates arrive pre-aligned. When writing new samplers or trajectory scalers, ensure alignment happens before the reward is evaluated.

### Atom Count Mismatches Between Model and Structure

Different models may represent the same protein with different atom counts. The `AtomReconciler` bridges this gap:

```python
reconciler = AtomReconciler.from_arrays(model_atom_array, structure_atom_array)
if reconciler.has_mismatch:
    # reconciler.model_indices and reconciler.struct_indices map between spaces
    aligned_coords, transform = reconciler.align(model_coords, reference_coords)
```

Build reward inputs from the model atom array (not the input structure) when a mismatch exists. See `utils/structure_utils.py::SampleworksProcessedStructure.to_reward_inputs()` for the canonical pattern. Two-phase rewards (`PreparableRewardFunctionProtocol`) are prepared with those same `RewardInputs`, never with an atom array; `RewardInputs.to_atom_array()` rebuilds a reference `AtomArray` (model topology, reconciled coordinates and B-factors) for forward models that need one. Do not write reconciled values back into the model atom array.

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
- Real-space electron density (X-ray crystallography) *implemented*
- Cryo-EM density *implemented*
- Structure factors (reciprocal space) *implemented*
- Diffuse scattering
- Cryo-EM image stacks

### Symmetry

Crystallographic symmetry is handled natively in forward models. Most ML models operate in P1 (asymmetric unit), but experimental maps are in the full crystal frame. The forward models bridge this gap.

## Key Files

- **`models/protocol.py`**: ModelWrapper protocol definitions
- **`core/scalers/protocol.py`**: StepScalerProtocol, TrajectoryScalerProtocol
- **`core/samplers/protocol.py`**: Sampler, TrajectorySampler, StepParams
- **`core/rewards/protocol.py`**: RewardFunctionProtocol, PrecomputableRewardFunctionProtocol, RewardInputs
- **`core/rewards/real_space_density.py`**: Reference reward implementation
- **`core/forward_models/xray/real_space_density.py`**: Differentiable density calculation
- **`core/scalers/step_scalers.py`**: DataSpaceDPSScaler, NoiseSpaceDPSScaler implementations
- **`core/scalers/pure_guidance.py`**: PureGuidance trajectory scaler (reference TrajectoryScalerProtocol impl)
- **`core/scalers/fk_steering.py`**: Feynman-Kaç steering trajectory scaler
- **`models/boltz/wrapper.py`**: Reference model wrapper implementation
- **`utils/guidance_script_utils.py`**: Central orchestrator — `run_guidance()` wires all components together
- **`utils/atom_reconciler.py`**: Handles atom count mismatches and differentiable alignment
- **`scripts/`**: Entry-point scripts for running guidance pipelines
- **`pyproject.toml`**: Package metadata, dependencies, tool config
