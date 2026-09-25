# Sampleworks

> This repository is under active development. Please always use the latest version. If you encounter any problems, please [create an issue on GitHub](https://github.com/prism-science/sampleworks/issues) and include: the PDB ID, the CIF file you used, your density map(s), and log information.

> We would welcome contributions from the community. We are most interested in:
 - new ModelWrappers for additional structure prediction models (especially smaller models which may be more steerable)
 - fast, differentiable modules to allow guidance from other experimental data modalities besides X-ray electron density.

**Sampleworks** is a Python framework for integrating generative biomolecular structure models with experimental data. Read our [blog post](https://diffuse.science/posts/sampleworks/) for an introduction.

## Why sampleworks?

Biomolecular structure prediction and design models are currently trained on single state structures and fail to accurately predict the ensemble of conformations each macromolecule occupies. But there is still hope! Current models show promise in capturing the underlying distribution of realistic macromolecular structures. We want to utilize the prior represented in these models and experimental observations to improve the sampling of the underlying ensemble present in the experiment and use this information to both understand biomolecular function and improve ensemble prediction.

Currently, each structure prediction model has a different implementation, requiring bespoke boilerplate code to plug each model into experimental guidance. Our goal is to resolve this and expand the experimental methods we can provide guidance with. This will open new opportunities for model evaluation directly against experimental data, and help unlock new sources of data for training the next generation of biomolecular structure predictors.

## Citation
If you use **sampleworks**, please cite:

Chrispens, K., Collins, M., Mai, D., Wankowicz, S. A., Fraser, J. S., & van den Bedem, H. (2026). sampleworks: A Modular Platform for Experimentally Guided Biomolecular Ensemble Generation. https://doi.org/10.82153/jkxj-tw08

## Installation

**Requirements**: Linux x86-64, CUDA 12, Python ≥ 3.11, < 3.14

> **Note**: macOS (arm64) is supported for local development. The environments that pull CUDA builds are `linux-64` only; `pixi install -a` installs the rest, which is enough to develop and test most of the codebase.

### 1. Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### 2. Clone and install

```bash
git clone git@github.com:prism-science/sampleworks.git
cd sampleworks
```

Sampleworks is split across several Pixi environments: one per generative model, plus environments for analysis and development. For a complete list, run `pixi workspace environment list`. To install all that are compatible with your machine, run:

```bash
pixi install -a
```

Alternatively, install only what you need:

```bash
pixi install -e boltz      # Boltz-1 / Boltz-2
pixi install -e protenix   # Protenix
pixi install -e rf3        # RosettaFold3
```

### 3. Download model checkpoints

**Boltz-1 and Boltz-2** (stored in `~/.boltz/`):

```bash
pixi run -e boltz python -c "
from boltz.main import download_boltz1, download_boltz2
import pathlib
cache = pathlib.Path('~/.boltz/').expanduser()
download_boltz1(cache)
download_boltz2(cache)
"
```

**Protenix**: checkpoint is downloaded automatically on first use.

**RosettaFold3** (RF3): see the [RC-Foundry repository](https://github.com/RosettaCommons/foundry) for instructions. Default path: `~/.foundry/checkpoints/rf3_foundry_01_24_latest.ckpt`


## Quick Start

Run Boltz-2 pure guidance on the included 1VME example:

```bash
pixi run -e boltz sampleworks-guidance \
    --model boltz2 \
    --guidance-type pure_guidance \
    --protein 1VME \
    --model-checkpoint ~/.boltz/boltz2_conf.ckpt \
    --structure tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif \
    --density tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4 \
    --resolution 1.8 \
    --output-dir output/boltz2_pure_guidance \
    --guidance-start 130 \
    --ensemble-size 4 \
    --augmentation \
    --align-to-input
```

Output files appear in `output/boltz2_pure_guidance/`: `refined.cif` (final ensemble), `losses.txt`, `trajectory/`, `run.log`.

### CLI reference

`sampleworks-guidance` is the unified command-line interface for running guidance on a single structure.

**Required arguments:**

| Argument | Description |
|---|---|
| `--model` | `boltz1`, `boltz2`, `protenix`, or `rf3` |
| `--guidance-type` | `pure_guidance` or `fk_steering` |
| `--protein` | Protein identifier (should match naming used in grid search / evaluation) |
| `--structure` | Path to input structure file (CIF) |
| `--density` | Path to density map (CCP4/MRC/MAP) — required by the default reward |
| `--resolution` | Map resolution in Angstroms — required by the default reward |

Model-specific arguments (e.g. `--method` for boltz2, `--msa-path` for rf3) and guidance-type-specific arguments (e.g. `--num-particles` for fk_steering) are included automatically. Run `sampleworks-guidance --model <model> --guidance-type <type> --help` to see all available options.

### Rewards

`--reward-type` chooses what the run is guided by, and each reward brings its own options:

| Reward | Guided by | Its options |
|---|---|---|
| `real_space_density` (default) | Fit to a density map | `--density`, `--resolution`, `--loss-order`, `--em` |
| `structure_factor` | Fit to structure-factor amplitudes from an MTZ | `--mtzfile`, `--expcolumns`, `--resolution`, `--bulk-solvent`, `--scattering-factor-mode`, ... |

```bash
sampleworks-guidance --model boltz2 --guidance-type pure_guidance \
    --protein 1VME --structure 1vme.cif \
    --reward-type structure_factor --mtzfile 1vme.mtz --bulk-solvent combined
```

To combine rewards, or to keep reward settings in version control, describe them in a file
(YAML, JSON, or TOML) and pass `--reward-config rewards.yaml`:

```yaml
real_space_density:
  weight: 0.4
  reward_options:
    density: 1vme.ccp4
    resolution: 1.8
structure_factor:
  weight: 0.6
  reward_options:
    mtzfile: 1vme.mtz
    bulk_solvent: combined
```

Weights default to `1/N` when omitted, so combining rewards leaves the meaning of the
guidance step size intact.



## Grid Search

`run_grid_search.py` sweeps a model across scalers, ensemble sizes, and gradient weights:

```bash
pixi run -e boltz python run_grid_search.py \
    --proteins proteins.csv \
    --models boltz2 \                # options: boltz1, boltz2, protenix, rf3 (make sure env aligns!)
    --methods "X-RAY DIFFRACTION" \  # only useful for Boltz-2, ignored otherwise
    --scalers pure_guidance \        # options: pure_guidance, fk_steering, or both as space-separated list
    --ensemble-sizes "1 4" \
    --gradient-weights "0.1 0.2" \
    --output-dir grid_search_results \
    --gradient-normalization \       # normalize guidance update magnitude to diffusion update magnitude
    --augmentation \                 # apply random rotations and translations at each step (defaults for inference with AF3-like models)
    --align-to-input                 # align to input structure at each step (required for density guidance to work since it is not rotation/translation invariant)
```

**`proteins.csv` format**

Required columns and format. Supported density map formats: `.ccp4`, `.mrc`, `.map` (not MTZ or SF-CIF yet).
```csv
name,structure,density,resolution
1abc,/data/structures/1abc.cif,/data/maps/1abc.ccp4,2.0
2xyz,/data/structures/2xyz.cif,/data/maps/2xyz.mrc,1.8
```

**Key arguments:**

| Argument | Description | Default |
|---|---|---|
| `--proteins` | CSV with structure/density/resolution columns | required |
| `--models` | Model to run. One of `boltz1`, `boltz2`, `protenix`, `rf3` | required |
| `--scalers` | Guidance method(s) to sweep | `pure_guidance fk_steering` |
| `--ensemble-sizes` | Space-separated values, e.g. `"1 4"` | `"1 2 4 8"` |
| `--gradient-weights` | Space-separated values, e.g. `"0.1 0.2"` | `"0.01 0.1 0.2"` |
| `--methods` | Boltz-2 sampling method (required for boltz2) | `X-RAY DIFFRACTION` |
| `--max-parallel` | Parallel workers (default: number of GPUs) | `auto` |
| `--dry-run` | Print jobs without running them | off |
| `--force-all` | Re-run including already-successful jobs | off |
| `--only-failed` | Re-run only failed jobs | off |
| `--only-missing` | Run only jobs not yet started | off |

Output layout: `grid_search_results/<protein>/<model>[_<method>]/<scaler>/ens<N>_gw<W>/`

> **Note**: Jobs are skipped if a `refined.cif` file already exists in the output directory. Some flags (e.g., `--use-tweedie`, `--gradient-normalization`) are not reflected in the directory structure, so changing them alone won't trigger a re-run. Use `--force-all` to re-run all jobs regardless. This is under active development and will likely change soon.

Evaluation and metrics scripts can be run through `run_analysis`; see the ACTL
section below and `scripts/eval/EVALUATION.md`.


## Running preset experiments on ACTL (`run_experiments`)

This section is Astera-specific: it assumes access to ACTL, the internal Harbor
image registry, and the `diffuse-shared` PVC. External users can run the same
TOML presets with `sampleworks-runs` or `python -m sampleworks.runs.cli` after
setting equivalent local paths for `DATA_DIR`, `PROTEINS_CSV`, `RESULTS_DIR`,
`MSA_CACHE_DIR`, and model checkpoints.

Start an 8-GPU ACTL machine named `sampleworks` with the private Astera
`sampleworks` image alias and the shared data volume mounted:

```bash
actl pod up sampleworks --profile 8x --image sampleworks --storage shared --pvc-size 200Gi --mount diffuse-shared --yes
```

Keep that terminal open; it maintains sync and SSH. From another terminal:

```bash
actl pod status sampleworks
# copy the `ssh:` line, then run it, for example:
ssh workspace.actl-ws-<user>-sampleworks.devspace
cd /home/dev/workspace
```

The main command is `run_experiments`. It reads TOML presets and launches the
right `run_grid_search.py` jobs, pixi environments, GPU assignments, logs,
results directory, and MSA cache.

```bash
export DATA_DIR=/mnt/diffuse-shared/raw/sampleworks/initial_dataset_40_occ_sweeps
export PROTEINS_CSV="$DATA_DIR/proteins.csv"
export SAMPLEWORKS_ACTL_RUN_NAME="$(hostname -s)"

run_experiments --list        # show available presets (does not require DATA_DIR)
run_experiments --show rf3    # inspect what will run
run_experiments --dry-run rf3 # print commands without running
run_experiments rf3           # run the standalone RF3 preset
run_experiments boltz         # run Boltz2 X-ray + Boltz2 MD
run_experiments boltz1        # run standalone Boltz1
run_experiments protenix      # run the standalone Protenix preset
run_experiments full_8gpu     # run the full 8-GPU comparison preset
```

The default `full_8gpu` preset runs Boltz2 XRD, Boltz2 MD, RF3, and Protenix in
parallel. Run a subset with:

```bash
run_experiments full_8gpu --jobs rf3,protenix
```

Standalone presets are available for each model/model family: `boltz`,
`boltz1`, `boltz2`, `boltz2_xrd`, `boltz2_md`, `rf3`, and `protenix`.
Additional comparison presets include `protenix_dual`, `rf3_protenix`, and RF3
variants. Single-job presets default to `gpu_count = 8`, so on an 8-GPU pod
they use the whole machine.

Presets live in `experiments/*.toml` in your local checkout and on the pod at
`/home/dev/workspace/experiments/*.toml`. To modify an experiment, edit or copy
a preset locally, let ACTL sync it, then run it by name or path:

```bash
cp experiments/rf3_partial.toml experiments/my_rf3.toml
# edit experiments/my_rf3.toml locally
run_experiments --preset my_rf3
```

For one-off changes, use `--set` instead of editing TOML:

```bash
run_experiments rf3 --set jobs.rf3.gpu_count=4
run_experiments rf3 --set jobs.rf3.args.gradient-weights="0.0 0.01 0.02"
```

Presets usually declare `gpu_count = N`, not fixed GPU IDs. The runner assigns
visible GPUs automatically in job order, so the same preset works on different
pod sizes and fails fast if the pod has fewer visible GPUs than requested. Use
explicit `gpus = "0,1"` only when you need to pin a job to specific devices; the
runner validates those IDs before launching jobs.

Set `DATA_DIR` and `PROTEINS_CSV` explicitly for each run so they are captured in
the shell history and launch logs. Checkpoints default to
`/mnt/diffuse-shared/raw/checkpoints` when those files exist, results go to
`/mnt/diffuse-shared/results/sampleworks/<pod>/<target>/`, and MSA caches go to
`/mnt/diffuse-shared/cache/sampleworks/msa`. Override with `RESULTS_DIR`,
`MSA_CACHE_DIR`, or model-specific checkpoint variables before running.

The ACTL image contains baked pixi environments under `/app/.pixi`. If your
synced branch changes `pyproject.toml` or `pixi.lock`, `run_experiments` stops
with a clear error instead of mutating the baked environment. For dependency
debugging only, opt into an on-pod pixi update with
`RUNTIME_PIXI=1 run_experiments ...`; reproducible scientist runs should use a
rebuilt `pixi-with-checkpoints:sampleworks` image instead.


## Running preset analyses on ACTL (`run_analysis`)

`run_analysis` uses the same TOML runner as `run_experiments`, but loads presets
from `analyses/*.toml` and runs the scripts under `scripts/eval/`.
The `analyze_grid_search`, `all`, and `external_tools` presets first run a sequential
`patch_outputs` pre-job, which creates `refined-patched.cif` files from each
`refined.cif` before the evaluation jobs start.

```bash
export GRID_SEARCH_RESULTS_DIR=/mnt/diffuse-shared/results/sampleworks/<pod>/full_8gpu
export GRID_SEARCH_INPUTS_DIR=/mnt/diffuse-shared/raw/sampleworks/initial_dataset_40_occ_sweeps
export PROTEIN_CONFIGS_CSV="$GRID_SEARCH_INPUTS_DIR/protein_analysis_config.csv"

run_analysis --list
run_analysis --dry-run analyze_grid_search --jobs rscc
run_analysis analyze_grid_search --jobs rscc,lddt
run_analysis altloc_find
run_analysis altloc_classify
run_analysis all  # includes tortoize and phenix.clashscore jobs
```

Use `--set` for one-off changes, for example
`run_analysis analyze_grid_search --jobs rscc --set jobs.rscc.gpus=0`. If your
input layout differs from the default
`processed/{pdb_id}/{pdb_id}_single_001_density_input.cif`, override the patch
pre-job with `--set defaults.PATCH_INPUT_PDB_PATTERN='{pdb_id}/{pdb_id}_original.cif'`.
When patched CIFs already exist, add `--skip-pre-jobs` to rerun analyses without
repeating the patching step.
The `altloc_find` and `altloc_classify` presets are independent of grid-search
outputs; override `ALTLOC_ANALYSIS_DIR` and `ALTLOC_INPUTS_DIR` when their input
or output roots differ from the defaults.


## Docker

Sampleworks now has a two-layer image split:

1. `Dockerfile` builds the regular public `pixi-with-checkpoints` image.
2. `Dockerfile.astera` builds the private Astera overlay with EXT, Dynamic PDB
   CLI, and small workspace conveniences, using the public
   `pixi-with-checkpoints` image as its base.

Image names:

| Purpose | Image |
|---|---|
| Public Sampleworks runtime | `diffuseproject/pixi-with-checkpoints` |
| Astera/ACTL runtime | `sampleworks` alias; run `actl pod images` for the resolved, digest-pinned ref |

CI publishes these tags:

| Image | Tags |
|---|---|
| Public | `latest` on `main`, `sha-<short-sha>`, release semver tags |
| Astera/Harbor | `latest` and `sampleworks` on `main`, `sha-<short-sha>`, release semver tags |

The Astera image is always built from the exact public `sha-<short-sha>` image
produced earlier in the same workflow run, then adds EXT, Dynamic PDB CLI, and
small workspace tools on top.

CI configuration variables:

| Variable | Purpose |
|---|---|
| `SAMPLEWORKS_PUBLIC_REGISTRY` | Public registry host; defaults to `docker.io` |
| `SAMPLEWORKS_PUBLIC_IMAGE` | Public image path; defaults to `diffuseproject/pixi-with-checkpoints` |
| `SAMPLEWORKS_CHECKPOINTS_DOCKERHUB_IMAGE` | Optional public Docker Hub checkpoint mirror destination tag; defaults to `docker.io/diffuseproject/sampleworks-checkpoints:latest` |
| `SAMPLEWORKS_CUDA_BASE_IMAGE` | Optional digest-pinned CUDA base override |
| `SAMPLEWORKS_CHECKPOINTS_SOURCE_PATH` | **Required.** Digest-pinned path of the private checkpoint image CI mirrors to Docker Hub, without the registry host (e.g. `library/foo@sha256:...`) |

One CI secret, `ASTERA_REGISTRY`, holds the internal registry host. It is a
secret rather than a variable because this repo is public, which makes its
Actions logs public, and only secrets are masked there. Log masking is
substring-based, so keeping the secret to the bare host masks it inside every
longer image ref while leaving paths and digests readable when a build fails.
The CI jobs fail fast when it is unset.

Build the public image locally:

```bash
docker build --platform linux/amd64 \
  --build-arg CHECKPOINTS_IMAGE=<checkpoint-image-ref> \
  -t diffuseproject/pixi-with-checkpoints:local \
  .
```

Build the Astera overlay locally after a public image is available. Set
`ASTERA_REGISTRY` to the internal registry host (same value as the CI
repository secret):

```bash
docker build --platform linux/amd64 \
  -f Dockerfile.astera \
  --build-arg PIXI_WITH_CHECKPOINTS_IMAGE=diffuseproject/pixi-with-checkpoints:local \
  -t "${ASTERA_REGISTRY}/library/pixi-with-checkpoints:local" \
  .
```

For local public builds, pass `CHECKPOINTS_IMAGE` as a public, digest-pinned
checkpoint image ref. In CI, the Docker workflow first mirrors the private Harbor
checkpoint source to Docker Hub, verifies the digest, and passes the resulting
digest-pinned Docker Hub ref to the public build so that build never needs Harbor
credentials.


## Development

For lightweight source-checkout development, use [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
uv run sampleworks-guidance --help
uv run pytest tests -m 'not slow'
```

Additional uv dependency groups are available for development and analysis:

```bash
uv sync --group dev
uv sync --group analysis
uv sync --group test
```

The existing GPU model environments are split by model, e.g. `boltz-dev`,
`protenix-dev`, `rf3-dev`. To install dev dependencies and run tests there:

```bash
pixi install -e [model]-dev    # add pytest, ruff, ty
pixi run -e [model]-dev all-tests  # run tests
pixi run test-all            # run all tests across all environments
```

**Prek hooks** (various formatting, ruff + ty type checking):

```bash
pixi run -e [model]-dev prek install
pixi run -e [model]-dev prek install --hook-type commit-msg
pixi run -e [model]-dev prek run --all-files
```

See [`tests/README.md`](tests/README.md) for full testing instructions.


## macOS (experimental)

On macOS, use uv for source-checkout development and avoid Linux/CUDA-only model
extras unless you know they are supported on your machine:

```bash
brew install uv
uv sync --group dev
uv run pytest tests -m 'not slow'
```

`protenix` currently requires `triton`/NVIDIA GPU support and is not expected to
work on macOS. Some RF3/Boltz workflows may also require Linux/CUDA packages.


## Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/) to automate versioning and changelog generation. Format:

```
<type>(<scope>): <summary>
```

Common types: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`, `perf`. A commitizen pre-commit hook validates messages at commit time. See [AGENTS.md](AGENTS.md#release-process) for full details.
