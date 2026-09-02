# Suggestion: try `lunus.sf` behind `StructureFactorRewardFunction`

A suggestion for whoever owns `dm/sf-reward`, not a plan of record.

## Why

Using structure factors as a sampleworks guidance target means backpropagating
through the forward model on every step, and `lunus.sf` may be a better fit
there than `SFC_Torch` in three ways:

- **Memory.** `structure_factors_batch` supports gradient checkpointing, making
  peak memory flat in ensemble size N rather than linear (~2.4× the time,
  bit-identical gradients). `SFcalculator` is memory-bound on its ASU-grid batch.
- **Gradient behaviour.** lunus splats with a smooth, tapered density cutoff
  (`taper_width`, default 0.1 Å) instead of a hard one, so no gradient
  discontinuity when an atom crosses the cutoff radius. Whether that helps
  convergence for this loss is untested and is the most interesting thing to find
  out.
- **Speed.** ~0.08 s per configuration steady-state on GPU. Beware benchmarking a
  single call: the first is ~3.8 s, almost all `torch.compile` warmup.

## Scope

Depends on `mew/diffuse-1` for `core/forward_models/xray/lunus_sf.py`. Nothing
from `-2` or `-3` is needed. `DiffuseBraggRewardFunction` on `-2` is a worked
example of the same pipeline.

Swap only the structure-factor and bulk-solvent computation. Keep SFcalculator
for MTZ ingestion, `Fo`, `Eo`, `Outlier`, `free_flag`, resolution bins,
`calc_Ec`, scales, and `calc_ftotal`.

## What to change

All in `core/rewards/structure_factor.py`.

**1. In `prepare()`** — build a `LunusSetup` alongside the existing
SFcalculator. `lunus_sf.build_setup` does the coordinate-independent work
(kernels, grid, symmetry ops) once, matching the two-phase split this reward
already uses.

**2. Replace `calc_fprotein_batch`** (line 554) and the body of
`_compute_ensemble_ftotal` (line 570):

```python
F_batch = lunus_sf.structure_factors(
    self._lunus_setup,
    coordinates,  # [batch, n_atoms, 3]
    occupancies,
    torch.as_tensor(self.sfc.HKL_array),  # SFcalculator's own reflection list
    solvent=self._solvent_model,  # lunus SolventModel, or None
    use_checkpoint=True,  # once N or the grid is large
)
self.sfc.Fprotein_HKL = F_batch.sum(dim=0)
self.sfc.Fmask_HKL = lunus_mask
fcalc = self.sfc.calc_ftotal()
```

Everything downstream — `calc_Ec`, `Eo`, `Fo`, `_build_reflection_mask`, the
loss — is untouched.

## Four things to know before starting

**The seam already exists.** Lines 614, 621 and 627 already assign
`Fprotein_HKL` and `Fmask_HKL` and then call `calc_ftotal()`. This is a change of
source, not of structure.

**Reflection ordering is not a problem.** `structure_factors` takes `hkl`
explicitly and returns `F` in that order — lunus indexes the FFT grid modulo its
shape, so no ASU convention is involved. Passing `sfc.HKL_array` aligns the
result positionally by construction.

**A lunus-side solvent mask is needed too.** Both current paths read
`Fprotein_asu_batch`, a side effect of `calc_fprotein_batch` (lines 618 and
625), so skipping that call means they cannot run. lunus's `SolventModel` is a
different mask construction — as Phenix, gemmi and SFcalculator already differ
from each other — and whether that difference matters for this optimisation is
not known.

**Scales were fitted against SFcalculator output.** `_set_scales` should be
re-fitted or at least re-checked once the source changes.

## Validation

1. Build the `LunusSetup` in `prepare()` and change nothing else; assert lunus
   and `calc_fprotein_batch` agree on the existing test system. This tests the
   whole risk in isolation.
2. Switch `_compute_ensemble_ftotal` with `bulk_solvent="off"`. Existing tests
   should pass unchanged.
3. Head-to-head guided run: wall-clock per step, peak memory, convergence on the
   same target. This answers the gradient question.
4. Only then move the solvent. Validate by R-factor against the synthetic MTZ —
   the two mask constructions differ, so expect agreement, not bit-parity.

Steps 1–2 are roughly a day. Step 3 says whether the rest is worth doing.

## Open question

Is there a reason `Fprotein_HKL` and `Fmask_HKL` should not be driven from
outside? The reward already does it, but if that seam is incidental rather than
intended, better to know now.
