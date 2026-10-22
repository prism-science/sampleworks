# FlowModelWrapper gotchas

Use these as investigation prompts and acceptance-test cases, not permission to
copy model-specific workarounds into every wrapper. Paths below are relative to
the repository root; inspect their current implementations.

## Atom identity is a stronger contract than shape

- **Model topology is not deposited topology.** Models may remove hydrogens or
  unsupported molecules, add OXT, complete side chains, or generate missing
  residues. Expose the actual model-order `model_atom_array` on conditioning when
  the topology differs, even if the total atom count happens to match. Its row
  order must match prior and step tensors exactly. Reference: conditioning classes
  in `src/sampleworks/models/*/wrapper.py` and
  `process_structure_to_trajectory_input()` in
  `src/sampleworks/utils/structure_utils.py`.
- **Padding is internal.** Pad to the network's mask, then remove only padded
  atoms. Do not reshape away real atoms, flatten different batch members
  together, or assume token masks and atom masks are interchangeable.
- **Gather/scatter must preserve identity.** Atom37 slot order differs from PDB
  atom order; a boolean-mask gather can silently permute coordinates. Keep
  explicit residue/slot maps in both directions and test with distinct per-atom
  values and a real-coordinate round trip.
- **Sequence and structure disagree legitimately.** Missing terminal/internal
  residues, noncontiguous numbering, insertion codes, modified residues, alternate
  locations, and repeated sequences in different chains need deliberate mapping.
  Preserve chain identity and model output residue assignments; do not remap by
  residue number alone. Use existing sequence/array helpers, not bespoke alignment.
  Reference: `src/sampleworks/utils/{sequence,atom_array_utils,atom_reconciler}.py`;
  full-sequence/serialization tests in
  `tests/models/test_model_wrapper_protocol.py`.
- **Reference validity and generated occupancy differ.** Invalid deposited atoms
  may be filtered from alignment, while generated model-only atoms still need
  finite template coordinates, appropriate elements, and meaningful positive
  occupancy to survive output writing. Do not carry placeholder zero occupancy
  into a completed model. Reference: model template construction in the wrappers
  and `save_trajectory()` in `src/sampleworks/utils/guidance_script_utils.py`.
- **Experimental annotations follow model order.** Use
  `SampleworksProcessedStructure.to_loss_inputs()` to reconcile common-atom
  B-factors and reference coordinates. Two-phase losses receive those same inputs;
  `LossInputs.to_atom_array()` reconstructs topology with reconciled values.
  Do not write experimental values back into the model topology.

## Match the network parameterization, not its method names

- **`t` may mean sigma.** The protocol's normalized-time language is not an
  enforced range. `AF3EDMSampler` supplies schedule-derived noise levels, including
  values >1. Trace the exact values into the native denoiser; convert only where
  required. Reference: `src/sampleworks/core/samplers/edm.py`.
- **Clean prediction is not raw network output.** Locate upstream EDM
  preconditioning or epsilon/velocity conversion. Check signs, scaling, units,
  and low-noise behavior against upstream. Do not precondition twice when the
  called entry point already returns clean coordinates.
- **Prior and schedule must agree.** Existing wrappers return unit Gaussian noise.
  Trace initial and partial-diffusion scaling in the chosen trajectory scaler and
  sampler before changing it; establish who applies every noise scale and when.
  Neither a shape-correct prior nor a familiar schedule proves compatibility.
- **Batch axes have different meanings.** Conditioning may describe one molecular
  system while `x_t` contains many ensemble members. Annotate meaningful axes with
  jaxtyping (e.g. `batch atoms 3` versus `batch tokens channels`) and prefer `einx`
  for wrapper tensor operations to make their meaning explicit. Broadcast features
  deliberately and preserve per-member noise levels. Unrestricted `squeeze()`
  loses batch=1. Reference: each wrapper's `step()` and
  `src/sampleworks/utils/framework_utils.py`.
- **Augmentation is part of the computation.** Trace native centering, rotation,
  alignment, and unit conversion so the sampler and model do not apply conflicting
  transforms. Evaluate density in the experimental frame through the existing
  sampler/reconciler path, including gradients through transforms.

## Autograd can fail silently

- **Inference helpers may disable gradients.** Inspect decorators, global inference
  modes, fused/compiled kernels, in-place updates, tensor reconstruction, and
  NumPy round trips in the entire denoising path. `.eval()` is appropriate for
  inference behavior but does not imply `no_grad()`. Freeze model parameters when
  appropriate without disabling gradients to inputs.
- **Cache constants, not old computation graphs.** Compute static trunk features
  once and reuse graph-free conditioning for coordinate guidance. Preserve the
  live denoiser path to `x_t`; verify two successive backward passes. Do not use
  `retain_graph=True` to hide incorrectly retained trunk graphs.
- **Latents change the cache rule.** An optimized single/pair representation is
  live, not constant. Recompute derived conditioning or invalidate it when the
  latent changes. Decide ITO requirements before implementing caching: coordinate
  gradients alone do not establish latent compatibility. When required, preserve
  injected optimizable leaves and test every requested latent path and derived
  cache; see [model-notes.md](model-notes.md).
- **Device and precision are not cosmetic.** Check parameters, coordinates, masks,
  indices, times, and caches together, including a nondefault CUDA device when
  supported. Test the actual kernel/dtype/backward combination; CPU success or a
  finite mixed-precision forward does not establish GPU gradient support.

## Floating point and reproducible native predictions

- **Seeds do not fix the execution environment.** Record upstream/PyTorch and
  dependency versions, hardware/device, model/input dtype, autocast, TF32/matmul
  precision, deterministic settings, and fused/compiled kernels for both runs.
  Operation order and nondeterministic kernels can change rounding, and small
  differences may accumulate over a trajectory.
- **The same seed need not produce the same random draws.** Featurization,
  augmentation, dropout settings, batch size, self-conditioning, and scheduling
  can change RNG consumption. Trace which draws generate the initial state and
  each stochastic update; match them as closely as the native API permits.
- **Close final coordinates are the acceptance target.** Compare to a saved
  prediction from the installed model's CLI with exact reproduction metadata.
  Require low RMSD with explicit correspondence, atom selection, coverage, and
  rigid-frame handling. Use stronger numerical assertions when settings match.
  Document unavoidable differences; do not insist on bitwise equality or widen
  tolerances after observing a failure without agreement.
- **One precision mode does not validate another.** Test representative compatible
  checkpoints in their supported precision modes. A CPU/full-precision result
  cannot certify a GPU/mixed-precision path. If exact replay is unavailable,
  identify what is reproducible before claiming a failed comparison is harmless.

## Lifecycle and operational traps

- **Self-conditioning is mutable state.** Identify whether repeated evaluations
  consume or advance history; reset/clone it for parity and finite-difference checks.
  Detach previous-step history when appropriate, not the current output.
  Establish how history is copied, selected, and reset for ensemble members,
  independent trajectories, and, in a requested follow-up, FK resampling.
- **Re-featurization must not reuse the wrong system.** Wrapper instances are
  reused across queued jobs. In-memory trunk caches, preprocessing files, and
  self-conditioning must reflect the current structure/configuration. Use separate
  work directories where upstream caches would collide. Reference:
  `run_guidance_job_queue()` in
  `src/sampleworks/utils/guidance_script_utils.py`.
- **MSAs and component dictionaries are external inputs.** Identify required CCD
  files, templates, MSA pairing, caches, network services, and preprocessing side
  effects. Document offline behavior and unsupported molecules. Do not silently
  send sequences/private structures to external services or add downloads just to
  make tests pass.
- **Checkpoint/configuration mismatch can look plausible.** Verify architecture,
  preprocessing configuration, checkpoint provenance, loaded keys, device, and
  inference settings against the native run. Derive dimensions/settings through
  upstream APIs so compatible checkpoints remain plug and play. Test representative
  variants and mark others unverified; do not suppress load mismatches or claim
  incompatible architectures work.
- **A protocol check only finds attributes.** Runtime protocol membership cannot
  prove signatures, atom correspondence, gradients, or scientific fidelity.
  Lightweight stand-ins can test framework plumbing, never native-model parity.
- **Green tests can mean no model ran.** Check optional imports, registry entries,
  checkpoint fixtures, platform constraints, markers, and collected case counts.
  CLI support additionally needs model selection, annotation, checkpoint lookup,
  sampler configuration, and serialization wiring, not just an exported class.

## Updating this skill with new experience

When the user supplies a new gotcha, capture the triggering conditions, observable
failure, responsible layer, and a test or diagnostic that distinguishes it from
similar failures. Label unverified reports as hypotheses. Add a concise rule here
and link its regression test when available; do not turn one model's workaround
into a universal constraint.
