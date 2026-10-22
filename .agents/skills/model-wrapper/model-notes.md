# Per-model FlowModelWrapper notes

Read the section closest to the upstream model being integrated. These are context
and source pointers, not requirements to reproduce every existing implementation.
Paths are relative to the repository root; verify current code before reusing it.

## Initialization and configuration patterns

Inspect the configuration dataclasses and constructors together. Existing
`BoltzConfig`, `ProtenixConfig`, `RF3Config`, and `ProtpardelleConfig` primarily
configure featurization; do not mistake them for complete constructor configs.
For a new wrapper, use a dataclass for relevant initial model options and resolve
them in `__init__`, delegating to the original package and standard PyTorch
operations. Keep structure-specific featurization separate.

- Boltz constructors use upstream parameter dataclasses, `asdict()`, and
  `load_from_checkpoint()`, followed by `.to(device).eval()`.
- Protenix's constructor uses upstream configuration parsing/model configuration,
  model construction, and PyTorch weight loading/evaluation setup.
- RF3's constructor initializes the upstream inference engine; model reuse also
  needs the associated engine/Fabric runtime, not merely a parameter module.
- Protpardelle's constructor delegates to upstream `load_model()` or moves an
  injected model to the requested device and evaluation mode.

These are reuse examples, not a mandate to reproduce every constructor workaround.

## Boltz-1 and Boltz-2

Source: `src/sampleworks/models/boltz/wrapper.py`.
Tests: `tests/models/boltz/test_boltz_wrapper.py`.

- Featurization uses upstream input processing/data loading and caches trunk
  conditioning. It creates input YAML and processed files; use distinct directories
  where different structures, configurations, or model versions could collide.
- `step()` pads coordinates for the upstream network and removes padding from the
  returned clean prediction. Atom masks are not token masks.
- Boltz-2 calls `preconditioned_network_forward()`, whose result already includes
  EDM preconditioning. Do not apply another clean-sample conversion.
- Build model topology from the upstream processed representation, not from the
  deposited input's atom count/order.

## Protenix

Source: `src/sampleworks/models/protenix/wrapper.py` and
`src/sampleworks/models/protenix/structure_processing.py`.
Tests: `tests/models/protenix/`.

- Featurization produces upstream features and cached trunk conditioning; trace
  terminal OXT, residue indexing, and missing-sequence handling.
- The denoiser has an extra singleton output axis. Remove that particular axis,
  not all singleton axes, so batch=1 is preserved.
- If ITO is required, `s_trunk`/`z_trunk` must stay live.
  Cached `pair_z`/`p_lm`/`c_l` depend on `z_trunk`; stale values can yield nonzero
  but incomplete gradients and incorrect forwards. Consult `docs/IT_OPT_TESTING.md`
  and `docs/IT_OPT_DESIGN.md` before choosing caching behavior.

## RF3

Source: `src/sampleworks/models/rf3/wrapper.py`.
Tests: `tests/models/rf3/`.

- Reuse the upstream Atomworks transform pipeline and retain the model's topology,
  including generated/unresolved atoms and their reference-coordinate treatment.
- Inspect chiral features and their gradient behavior rather than copying a
  disable/diagnostic setting into another model.
- Validate explicit/nondefault device selection and the actual precision/kernel
  combination. See `tests/models/rf3/test_rf3_device_selection.py`.

## Protpardelle

Source: `src/sampleworks/models/protpardelle/wrapper.py`.
Tests: `tests/models/protpardelle/test_protpardelle_wrapper.py`.

- Upstream consumes atom37 coordinates; sampleworks uses flat atoms. Use the
  explicit residue/atom-slot maps for scatter and gather in both directions.
  A boolean-mask gather emits atom37-slot order, not necessarily input atom order.
- `step()` mutates `conditioning.x_self_conditioning`. Reset/clone history for
  independent runs, single-evaluation comparisons, and finite differences.
- For a requested FK follow-up, establish how resampling transfers that history
  to the correct particle/member; do not assume a shared mutable tensor works.

## ITO compatibility when required

Read `src/sampleworks/models/latent_adapter.py`, `docs/IT_OPT_DESIGN.md`, and
`docs/IT_OPT_TESTING.md` during initial exploration if ITO is required. Confirm
that the chosen latent attributes actually exist and can be replaced through the
adapter; do not invent a single/pair representation for a model without one.
Static cached conditioning and optimizable conditioning have different graph/cache
lifecycles: do not detach injected trainable leaves or reuse conditioning derived
from an old latent. Model-agnostic optimization remains in the scaler.
