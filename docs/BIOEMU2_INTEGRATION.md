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

BioEmu2's EDM recipe matches `AF3EDMSampler`. `_run_guidance` reads `sigma_data`, `sigma_min` and
`sigma_max` from the checkpoint and uses BioEmu2's own `gamma_min=1.0`, `noise_scale=1.0`,
`step_scale=1.0` (equilibrium) or `1.5` (structure), with reverse-diffusion alignment on.

`step()` keeps the autograd graph. Gradients reach `x_t` (noise-space DPS), and the trunk `single`
and `pair` tensors; when `pair` requires grad, `step()` rebuilds the static context with gradients
so latent optimization can work on it.

## Atoms

- Polymer chains use `processed_entity_canonical_sequence`, so sequence overrides carry through
  and modified residues (MSE, PCA) become their canonical parents.
- Every non-polymer residue becomes a CCD ligand (`--bioemu2-exclude-ligands` drops them). A chain
  holding several hetero groups is split into one ligand per residue and mapped back to the input
  chain and residue number. Residues BioEmu2 cannot build (not in the CCD) are skipped with a
  warning. Waters are dropped.
- BioEmu2 models heavy atoms and no terminal OXT. On the test structures, every input atom
  reconciles except OXT and atoms of modified residues.
- `model_atom_array` takes residue numbers from `chain_info["res_id"]` when it covers the
  sequence, so `refined.cif` keeps the input numbering.

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
| `--bioemu2-exclude-ligands` | off | Do not model non-polymer chains |

`BIOEMU2_MSA_CACHE` and `BIOEMU2_EMBEDS_CACHE` (BioEmu2's own caches) apply as documented upstream.
The checkpoint resolves from `--model-checkpoint`, `BIOEMU2_CHECKPOINT`,
`/checkpoints/bioemu2-preview/checkpoint.ckpt`, then
`/mnt/diffuse-shared/raw/checkpoints/bioemu2-preview/checkpoint.ckpt`.

## Environment

`bioemu2` / `bioemu2-dev` pixi environments (Python 3.12, `bioemu2==0.1.0rc0`). Linux uses
torch 2.13.0+cu126 and cu12 cuEquivariance 0.11.1; macOS gets CPU torch for the checkpoint-free
tests. Two feature-scoped overrides make it solve alongside the shared stack:

- `biotite==1.4.0`: atomworks 2.1.1 pins it; BioEmu2 asks for `>=1.5` but only uses APIs 1.4.0 has.
- `torch>=2.11,<2.14`: lifts atomworks' `torch<2.8` cap on its `ml` extra.

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

## Not done yet

- Bake the checkpoint into the checkpoints image used by `Dockerfile` (`CHECKPOINTS_IMAGE`).
- Latent-adapter entries for IT-opt (`models/latent_adapter.py`).
- Baseline: BioEmu2's own FK/FKC steering on a sampleworks loss, through a `Potential` bridge.
- Disulfide `Bond`s and contact conditioning are not passed to BioEmu2.
