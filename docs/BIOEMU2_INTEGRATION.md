# BioEmu2 in sampleworks

[BioEmu2](https://github.com/microsoft/bioemu2) (Microsoft Research, MIT) is an all-atom EDM
diffusion model trained to sample equilibrium conformational ensembles. Sampleworks runs it as a
`FlowModelWrapper`: `AF3EDMSampler` drives the loop and calls BioEmu2's denoiser once per step.
BioEmu2's own samplers, steering potentials and trajectory writers are not used, so every
sampleworks loss, scaler, CLI path and eval applies unchanged.

## How it fits

| Step | Code | Runs |
| --- | --- | --- |
| Build a BioEmu2 `System` from `chain_info` | `system_from_structure` | once per structure |
| MSA (ColabFold, A3M file, or single sequence) | `resolve_msa` | once per structure |
| Trunk embeddings | `model.embed` (no grad, bf16) | once per structure |
| Static context (pair + relative position → z-conditioning) | `model.compute_static_context` | once per structure |
| Pad atoms to a multiple of 32, `denoise`, slice back | `BioEmu2Wrapper.step` | every sampler step |

BioEmu2's EDM recipe matches `AF3EDMSampler`. `bioemu2_edm_settings(model_wrapper, mode)`
(`models/bioemu2/sampler.py`) returns the `EDMSamplerConfig` keyword arguments for BioEmu2's own
`edm_sampler`: `sigma_data`, `sigma_min` and `sigma_max` from the checkpoint (the denoiser
preconditions with the same schedule), `gamma_0=0.8`, `gamma_min=1.0`, `noise_scale=1.0`,
`step_scale=1.0` (equilibrium) or `1.5` (structure). Spread them into an `EDMSamplerConfig` and pass
it to `AF3EDMSampler`, exactly as for every other model — `_run_guidance` does this on the shared
sampler path (with reverse-diffusion alignment on), and any field can be overridden at the call
site. The wrapper itself holds no sampler settings.

`step()` keeps the autograd graph. Gradients reach `x_t` (noise-space DPS), and the trunk `single`
and `pair` tensors. The static context is built from `single`, `pair` and the relative-position
encoding, so `step()` rebuilds it whenever any of them is not the tensor it was built from (an
identity check, as Boltz2 does since #442). IT-opt (`--guidance-type latent_opt`) addresses
BioEmu2's `single` / `pair` through `models/latent_adapter.py`. The trunk itself runs once under
`no_grad`, so gradients stop at its outputs.

## Atoms

- Polymer chains use `processed_entity_canonical_sequence`, so sequence overrides carry through
  and modified residues (MSE, PCA) become their canonical parents. Their extra atoms do not
  reconcile; atomizing modified residues is a gap shared by every wrapper.
- Every non-polymer residue in the input becomes a CCD ligand; remove a ligand from the input file
  to leave it out. A chain holding several hetero groups is split into one ligand per residue and
  mapped back to the input chain and residue number. BioEmu2 groups identical ligands, so
  `model_atom_array` carries a `seq_idx` annotation (the ligand's position in its input chain) and
  reconciliation pairs ligands by input position, not by topology order. Waters are dropped.
- Ligands BioEmu2 cannot build are skipped with a warning: components missing from the CCD, and
  5-character CCD codes that a PDB file cut to 3 characters (atomworks marks both with `~`).
  Supplying custom CCD definitions for them is not supported yet.
- BioEmu2 models heavy atoms and no terminal OXT. On the test structures, every input atom
  reconciles except OXT and atoms of modified residues.
- `model_atom_array` takes residue numbers from `chain_info["res_id"]` when it covers the
  sequence, so `refined.cif` keeps the input numbering. Its atoms start at the input centroid:
  reconciliation replaces that with the input coordinates of the atoms the input has, and the
  rest (missing side chains, residues added by a sequence override) start partial diffusion from
  the centroid, not the origin.

## Usage

```bash
pixi run -e bioemu2 sampleworks-guidance --model bioemu2 --guidance-type pure_guidance \
    --protein 1VME --model-checkpoint /path/to/bioemu2-preview/checkpoint.ckpt \
    --structure tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif \
    --density tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4 \
    --resolution 1.8 --ensemble-size 4 --augmentation --align-to-input
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--bioemu2-mode` | `equilibrium` | `structure` predicts a single folded state |
| `--msa-path` | ColabFold | A3M file or JSON `{chain_id: a3m}` |
| `--bioemu2-no-msa-server` | off | Single-sequence MSA when no `--msa-path` (offline) |

`run_grid_search.py --model bioemu2` takes `--bioemu2-mode` and `--bioemu2-no-msa-server`. A grid
spans several proteins, so it fetches MSAs from ColabFold rather than taking one `--msa-path`.

`BIOEMU2_MSA_CACHE` and `BIOEMU2_EMBEDS_CACHE` (BioEmu2's own caches) apply as documented upstream.
The checkpoint resolves from `--model-checkpoint`, `BIOEMU2_CHECKPOINT`,
`/checkpoints/bioemu2-preview/checkpoint.ckpt`, then
`/mnt/diffuse-shared/raw/checkpoints/bioemu2-preview/checkpoint.ckpt`.

## Environment

`bioemu2` / `bioemu2-dev` pixi environments (Python 3.12, `bioemu2==0.1.0rc0`). Linux uses
torch 2.13.0+cu126 and cu12 cuEquivariance 0.11.1; macOS gets CPU torch for the checkpoint-free
tests. Two feature-scoped overrides make it solve alongside the shared stack:

- `biotite==1.4.0`: atomworks 2.1.1 pins it; BioEmu2 asks for `>=1.5` but only uses APIs 1.4.0 has.
- `torch>=2.11,<2.14`: lifts atomworks' `torch<2.8` cap on its `ml` extra. An override replaces
  the requirement it names, including its index, and every PyPI Linux torch from 2.11 on is a
  CUDA 13 build. The feature therefore adds the cu126 index as an extra index
  (`index-strategy = "unsafe-best-match"`), so Linux resolves to `2.13.0+cu126`; that index has
  no macOS wheels, so macOS stays on PyPI.

The workspace `numpy<2.0` cap stays; BioEmu2 runs on numpy 1.26, so no other environment relocks.

## Tests

`tests/models/bioemu2/` builds a random-weight model from BioEmu2's test config
(`tests/resources/bioemu2/tiny_config.yaml`, `trimul_variant: full`), so featurization, atom
bookkeeping, batching and gradient flow are tested on CPU without the checkpoint. The shared
protocol and integration tests pick up the `bioemu2` registry entry and run with the real
checkpoint on a GPU.

## Validation (2026-10-27, one H200, `bioemu2-preview`)

- The `bioemu2` tests in `tests/models`, `test_model_wrapper_protocol.py` and `tests/integration`
  pass with the real checkpoint (98 tests).
- Unguided 1VME, 200 steps, ensemble of 8: BioEmu2's own `EDMSampler` gives 8/8 physical samples,
  median CA-RMSD to the input 2.24 Å; the same embeddings through `AF3EDMSampler` give 8/8
  physical, median 2.73 Å.
- Pure guidance on the 1VME density with `--gradient-normalization --step-size 0.1`: final loss
  0.0147 against 0.0483 unguided, median CA-RMSD 1.80 Å, but 0/8 samples pass BioEmu2's
  high-quality physicality check (all `bad_bonds_backbone`). Without gradient normalization,
  step size 0.1 does not move the loss. Guidance strength needs tuning in the occupancy sweep.

### Guidance methods (2026-10-28, same setup)

Every row was scored the same way: the ensemble density loss on `refined.cif`, BioEmu2's
high-quality physicality check, and the median CA-RMSD to the input.

| Run | Density loss | Physical | Median CA-RMSD |
| --- | --- | --- | --- |
| Unguided | 0.0556 | 8/8 | 2.58 Å |
| Pure guidance, `--gradient-normalization --step-size 0.1` | 0.0147 | 0/8 | 1.82 Å |
| IT-opt, `--which-latent pair` (defaults) | 0.0249 | 8/8 | 1.69 Å |

- IT-opt halves the loss and keeps every sample physical.
- Pure guidance fits the density best but breaks backbone bonds.

## Not done yet

- Bake the checkpoint into the checkpoints image used by `Dockerfile` (`CHECKPOINTS_IMAGE`).
- Disulfide `Bond`s and contact conditioning are not passed to BioEmu2.
- Reproducing BioEmu2's steered sampling with sampleworks' `FKSteering` is a separate, stacked PR.
