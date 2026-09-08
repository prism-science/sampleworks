# IT-Opt — Testing, Verification, and Wiring

How to run [`LatentOptimization`](../src/sampleworks/core/scalers/latent_optimization.py), debug it
when it misbehaves, and where it is wired into the pipeline. For the architecture, read
[IT_OPT_DESIGN.md](IT_OPT_DESIGN.md) first.

Protenix is the primary test target (it is the model the reference algorithm was written for), so the
recipes below use it; Boltz1/RF3 need neither precondition in §1.

---

## 1. Protenix preconditions (read first)

**(a) The optimizable latent must reach the trunk.** Protenix's `step()` detaches the cached trunk
outputs when gradients are on (to avoid double-backward across the denoising steps) — correct for
coordinate guidance, but it would make a *latent* gradient **exactly zero**, silently. The
`detach_unless_leaf` helper in [models/protenix/wrapper.py](../src/sampleworks/models/protenix/wrapper.py)
handles both paths automatically: a cached latent detaches, but a latent IT-opt injected as an
optimizable leaf (`requires_grad=True`) is kept attached. No manual edit is needed — if the §2 Level-0
check fails, confirm the leaf actually has `requires_grad=True` and that `single_attr`/`pair_attr` are
`s_trunk`/`z_trunk`.

**(b) For `z` optimization, disable the diffusion shared-vars cache.** `pair_z`, `p_lm`, `c_l` are
cached from `z_trunk` at featurize time. Optimizing `z_trunk` with the cache on makes the diffusion
module read the **stale** `pair_z` — the `z` gradient is only partial and the forward is wrong. Build
the Protenix wrapper with `enable_diffusion_shared_vars_cache=False` (which `_run_guidance` does for
`LATENT_OPT`), or start single-only (`optimize_pair=False`), which the cache doesn't affect.

> Recommended first Protenix run: `optimize_single=True, optimize_pair=False` with the cache at its
> default. It exercises the whole loop without the cache subtlety; then turn on `z` with the cache off.

## 2. Debug ladder

The runtime lives on the Astera pod; edit locally, sync, run in the Protenix env. Work up the levels
and stop at the first that fails.

```bash
pixi run -e protenix-dev python your_it_opt_test.py            # needs the checkpoint + a map/structure
pixi run -e protenix-dev python -m pytest tests/models/test_latent_adapter.py -q   # CPU, no checkpoint
```

### Level 0 — Does the gradient reach the latent? (the #1 failure mode)

Needs no reward; confirms precondition (a). Inject a `requires_grad` leaf as `z_trunk` (or
`s_trunk`), run one differentiable denoiser forward, and check the leaf received a gradient:

```python
import torch
from sampleworks.utils.guidance_script_utils import get_model_and_device, get_reward_function_and_structure
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.models.latent_adapter import AttrLatentIO
from sampleworks.models.protocol import GenerativeModelInput

device, model = get_model_and_device("cuda:0", "<protenix.ckpt>", "protenix")
reward, structure = get_reward_function_and_structure(
    density="<map.ccp4>", device=device, em=False, loss_order=2,
    resolution=1.8, structure_path="<structure.cif>",
)

feats = model.featurize(structure)                       # trunk pass
io = AttrLatentIO(single_attr="s_trunk", pair_attr="z_trunk")

z0 = io.read_pair(feats.conditioning).detach()
z_leaf = z0.clone().requires_grad_(True)
cond = io.write_pair(feats.conditioning, z_leaf)
feats2 = GenerativeModelInput(conditioning=cond)         # conditioning-only (x_init removed in #330)

sampler = AF3EDMSampler(EDMSamplerConfig(device=str(device), augmentation=False))
schedule = sampler.compute_schedule(num_steps=200)
t_hat = schedule.t_hat[100]
x = torch.as_tensor(model.initialize_from_prior(batch_size=1, features=feats2))

with torch.enable_grad():
    x0 = model.step(x, t_hat, features=feats2)
    x0.sum().backward()

print("z_leaf.grad is None:", z_leaf.grad is None)
print("z_leaf.grad abs-sum:", None if z_leaf.grad is None else z_leaf.grad.abs().sum().item())
```

**Expect** `grad is None → False`, `abs-sum → > 0`. Repeat with `read_single`/`write_single` and
`s_trunk` to check `s`.

> With the `z` cache **on**, this still shows a non-zero grad (through the direct `z_trunk` arg) but
> the forward used the stale `pair_z`. Level 0 confirms the leaf reaches the trunk, not cache
> correctness — that is why §1(b) matters for real `z` runs.

### Level 1 — A tiny end-to-end run

```python
from sampleworks.core.scalers.latent_optimization import LatentOptimization

itopt = LatentOptimization(
    ensemble_size=1, num_steps=20, outer_steps=1,   # tiny
    learning_rate=0.05, max_grad_norm=1.0,
    optimize_single=True, optimize_pair=False,      # single-only is cache-safe
    anchor_weight_single=1.0, single_attr="s_trunk", pair_attr="z_trunk",
)
out = itopt.sample(structure, model, sampler, step_scaler=None, reward=reward)
opt = out.metadata["optimization_losses"][0]        # per-step data losses, round 0
print("first→last opt loss:", opt[0], "→", opt[-1])
```

**Expect** the optimization loss trends **down** across the round and `out.final_state` has shape
`[ensemble_size, n_atoms, 3]`.

### Level 2 — Scale up

Raise `num_steps` (→200), `outer_steps` (→2–4), `ensemble_size`; turn on `z` (`optimize_pair=True`
**with the cache disabled**). Compare the final ensemble's density fit (RSCC) to an unguided baseline.

## 3. Failure → diagnosis

| Symptom | Likely cause | Fix |
|---|---|---|
| `latent.grad is None` / zero (Level 0) | Injected latent isn't a leaf reaching the trunk (§1a); or you optimized a latent the model doesn't route to. | Confirm `requires_grad=True`; use `single_attr="s_trunk"`, `pair_attr="z_trunk"` for Protenix. |
| `z` grad non-zero but structures don't respond | Stale `pair_z` cache (§1b). | `enable_diffusion_shared_vars_cache=False`, or optimize single-only first. |
| Optimization loss is flat | LR too small; anchor too strong (latents pinned); or gradient fully clipped. | Raise `learning_rate`; lower `anchor_weight_*`; raise `max_grad_norm`. Log the pre-clip grad norm. |
| Loss drops but structures degrade | Latent drifted off-manifold. | Raise `anchor_weight_*`; fewer steps/rounds; raise `bond_length_weight`. |
| NaNs | Augmentation/churn stochasticity, or bf16 latents. | `EDMSamplerConfig(augmentation=False)`; keep latents float32; lower LR. |
| `ValueError: no optimizable latent on the conditioning` | Wrong attr names (Boltz `"s"`/`"z"` on Protenix). | Use `s_trunk`/`z_trunk`. |
| `ValueError: State atom count != reward_inputs atom count` | Reward built from a different atom set than the model's. | Use the same structure/density the CLI takes. |

Cheap instrumentation (one line each in `_latent_adam_step`): the pre-clip grad norm (is the clip
biting?) and the anchor's `mean((Δs)²)`, `mean((Δz)²)` (how far the latents drifted).

## 4. CLI and wiring

`GuidanceType.LATENT_OPT` routes to `LatentOptimization` in `_run_guidance`
([utils/guidance_script_utils.py](../src/sampleworks/utils/guidance_script_utils.py)):

```bash
sampleworks-guidance --model protenix --guidance-type latent_opt …
```

The same seam resolves the model name to `single_attr`/`pair_attr` (raising a clear `ValueError` for
an unsupported model) and, for Protenix, disables the diffusion shared-vars cache so the `z` gradient
survives. The direct-script route in §2 is the better *debugging* surface — it isolates the scaler
from the grid-search / save machinery.

**Production footprint** (each tagged with a greppable `IT-opt wiring` comment):
- [utils/guidance_constants.py](../src/sampleworks/utils/guidance_constants.py) — `GuidanceType.LATENT_OPT`.
- [utils/guidance_script_arguments.py](../src/sampleworks/utils/guidance_script_arguments.py) —
  `add_latent_opt_args` (`--which-latent`, `--learning-rate`, `--outer-steps`, `--anchor-weight`,
  `--max-grad-norm`, `--bond-length-weight`); the six names are also in `_DYNAMIC_ATTRS` so
  `GuidanceConfig.from_cli` copies them onto the config.
- [utils/guidance_script_utils.py](../src/sampleworks/utils/guidance_script_utils.py) — the
  `LATENT_OPT` dispatch, the Protenix cache-off decision, and the `save_trajectory` case.
- [core/scalers/latent_optimization.py](../src/sampleworks/core/scalers/latent_optimization.py) —
  per-latent grad clips and the `latent_drift` diagnostic.

## 5. Open problems

1. **Density fit was measured with a local scorer**, not `scripts/eval/rscc_grid_search_script.py`, so
   no number so far is repo-exact. A repo-exact run needs a depth-4 trial-dir tree
   (`{PROTEIN}_native_occ/{model}_MD/{scaler}/ens{N}_gw{W}/refined.cif`); the generated ensembles are
   flat, so symlinks suffice.
2. **Only native-occupancy rows were scored.** Enough to compare conditions on identical rows;
   absolute fractions need a wider population.
3. **Five proteins are excluded.** 6RP1, 7Z0E, 4OLE, 8Z76, 2I6H raise "No common atoms found"
   (chain/residue-naming mismatch). Fix with `scripts/patch_output_cif_files.py` (needs network for
   `rcsb.fetch`) or sequence-based atom matching.
