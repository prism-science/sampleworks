# Holton weighted energy as a sampleworks loss

**Status:** Draft design for discussion; not an approved implementation specification.  
**Tracking:** [#454 — add UNTANGLE challenge wE as a loss function](https://github.com/prism-science/sampleworks/issues/454).  
**Deliverable:** A Python implementation of geometry-only weighted energy (`wE`), usable for reference-compatible scoring and differentiable guidance.  
**Repository baseline:** Refreshed against `2575754c06b115073b4c85ffb759129e2015af41`; implementation names and source links below use the current loss API.  
**Base dependency:** Build on the issue-71 loss-rename branch, `kc/loss-rename` (observed ancestor `913a0e67`), rather than proposing a second rename or assuming the loss API is already on every target branch.  
**Workflow:** Use `jj` for version-control work in sampleworks workspaces.  
**Unresolved decisions:** See [Open questions and decision gates](#7-open-questions-and-decision-gates). An open decision blocks only the work that depends on it.

## 1. Motivation and scientific definition

Experimental agreement alone can favor chemically implausible structures. Holton's `wE` combines geometry and validation observations into a score that emphasizes statistically concerning deviations without letting a single enormous outlier overwhelm every other consideration.

The goal is to make that score available alongside sampleworks' existing experimental losses. It is not to replace the generative model's prior, introduce a general molecular force field, or solve the UNTANGLE challenge.

### Reference and authority

The primary reference is [James Holton's UNTANGLE scoring description](https://github.com/jmholton/UnTangle/tree/973585629b7a33efe42a68843356c0e953f66c41#score), together with the executable [scoring script at revision `973585629b7a33efe42a68843356c0e953f66c41`][holton-script].

Use the active script as the initial numerical reference, rather than reconstructing its behavior from introductory comments or displayed equations alone. It delegates geometry and validation to Phenix/MolProbity, so the script revision is necessary but insufficient: reference tool versions, restraint libraries, options, input chemistry, and hydrogen treatment must also be pinned.

Phenix is available through the project's actl Pod environment. The [Astera image integration][astera-sbgrid] initializes the SBGrid collection mounted at `/programs`; the programs are not necessarily embedded in the container image or available in every image/namespace. Initial work is to select and verify that existing environment, not assume Phenix must be acquired or installed locally.

Use `actl pod images` to discover the current image catalog and `actl pod status` to inspect an existing workspace before reference generation. Record the selected image digest, SBGrid mount and software selection, actual Phenix/MolProbity versions, and availability/versions of `gnuplot` and `tcsh`. An image digest alone does not pin the separately mounted software. Edit source/fixture definitions locally, let actl sync carry them, and run reference calculations on the Pod. No Pod provisioning or reference calculation is part of this document edit.

Values generated manually in that verified environment will be checked into the repository with provenance; ordinary tests will not require those installations or cluster access.

### Score structure

The eleven primary categories are bonds, nonbonded interactions, MolProbity clashes, angles, torsions, chirality, planarity, peptide omega, Ramachandran, rotamers, and Cβ deviation.

For ordinary geometry, the conceptual energy is a squared normalized deviation:

$$
E_i = \left(\frac{v_i-v_{0,i}}{\sigma_i}\right)^2.
$$

This is not permission to substitute arbitrary ideal values or sigmas. Periodicity, restraint conventions, atom selection, and observation counts are part of the score.

For each category, the reference combines an average-energy term with a worst-outlier term. A schematic representation is:

$$
wE_c = W_{\mathrm{avg}}(\bar E_c,N_c)\,C(\bar E_c)
      + W_{\mathrm{worst}}(E_{\max,c},N_c)\,C(E_{\max,c}).
$$

Here `C` is the reference's energy compression:

$$
C(E)=
\begin{cases}
0 & E \le 0,\\
E & 0 < E \le 10,\\
10+\log(E)-\log(10) & E > 10.
\end{cases}
$$

The expression is a conceptual guide, not a complete executable specification: clash adjustments, category filtering, probability transforms, and weight-input caps have additional rules.

Important distinctions to retain during specification:

| Aspect | Evidence and design consequence |
|---|---|
| Average weight | The active script uses gnuplot `igamma(N/2, N·mean(E)/2)` below `N=3200`, and a square-root/`erf` approximation above that boundary. Verify the special-function convention; do not substitute a chi-square survival probability by name alone. |
| Worst weight | The conceptual Gaussian weight is `erf(abs(d)/sqrt(2))^N`. The active scorer uses `softPnna`, fitted coefficients, `log10(N)`, and a variable exponent instead. Preserve active coefficients unless an explicit deviation is approved. |
| Weight-input cap | Energies supplied to weight calculation are clamped to `[0, 400]`; final energy compression is a separate operation. |
| Nonbonded energy | A shifted Lennard–Jones 6–12 expression uses geometry-derived reference distances and a shift at 6 Å. The scored positive-energy subset has its own count. |
| Full nonbonded average | The script also feeds `avg_full_nonbond` into the weighting/summation pipeline. It often contributes zero after negative-value handling, but must not be assumed to be purely reporting. Establish its behavior in the reference study. |
| Clash contribution | Clash parsing uses its own reference distance; the worst record incorporates clash count, and the final pass adds the record's second field **after** energy compression. It is not simply the ordinary nonbonded category again. |
| Plane observations | The script emits individual plane-atom deviations; replacing these with one observation per plane changes counts and weighting. |
| Omega | The active expression uses sine/cosine terms, a fixed sigma near 4°, and an adjustment based on the count of neighboring prolines. |
| Rama/rotamer energy | The active script evaluates `inverf(1 - probability_percent/100 - 1e-16)^2`, with a special fallback to `99`. Do not add a factor of two from a Gaussian-deviate derivation without reconciling it against the script. |
| Hydrogen selection | Selection differs by category. Heavy-atom-only geometry and hydrogen-influenced contact validation cannot be treated as one global filter. |

These details come from the [observation extraction][holton-extraction], [validation transforms][holton-transforms], and [aggregation/weighting][holton-weights] blocks. Their reproducibility must be established through fixtures, not asserted from inspection alone.

### Geometry-only wE versus challenge score

The overall UNTANGLE challenge score additionally includes:

$$
\mathrm{challenge\ score} = wE + \left(\frac{R_{\mathrm{free}}-2}{2}\right)^2,
$$

where `R_free` is expressed in percent.

That term is **not part of this loss**. Real-space density, structure-factor, and combined Bragg/diffuse agreement remain separate experimental losses. Challenge-score reporting may be follow-up evaluation work; `R_free` must not silently become a guidance target.

## 2. Goals and anti-goals

### Goals

1. Provide both reference-compatible scoring and a coordinate-differentiable guidance path, implemented in Python without a Phenix/MolProbity runtime dependency.
2. Compare against pinned external implementations at observation, category, and total-score levels using stored reference values.
3. Initially support standard protein residues, disulfides, and waters. Resolve hydrogen and symmetry policies before claiming their corresponding reference coverage.
4. Preserve reference multi-conformer semantics as the intended baseline; investigate how they map onto sampleworks' ensemble tensors before choosing a reduction.
5. Compose with existing experimental losses through the existing scalar, minimized-value contract.
6. Make the smallest demonstrated contract refinements necessary for geometry losses.
7. Deliver focused PRs with explicit category coverage and acceptance criteria.

### Anti-goals

- Solving UNTANGLE or guaranteeing escape from tangled local minima.
- Replacing the model prior with a general force field. Reusing suitable `tmol` components to implement wE is allowed and explicitly evaluated below; substituting its generic total energy for wE is not.
- Adding occupancy/B-factor refinement, conformer-label swaps, or topology-changing moves.
- Supporting arbitrary ligands, modified residues, nucleic acids, and metals in the first release.
- Requiring Phenix/MolProbity or shell subprocesses during guidance. External installations are for offline reference generation only.
- An unrelated sampler/scaler rewrite.

Full guided-sampling experiments, guidance-weight tuning, and evidence that geometry improvement preserves experimental fit are follow-up work. Initial integration acceptance covers composition and gradient plumbing, not a scientific claim about sampling performance.

## 3. Current architecture and loss-contract requirements

### Changes since the first draft

The reward-to-loss rename is already present in the attached code through the [#71][issue-71] base branch, `kc/loss-rename`. This design targets `core/losses`, `tests/losses`, `--loss-type`, `--loss-config`, and `loss_options`; it does not propose another naming migration or legacy reward aliases. Preserve that branch dependency when stacking implementation PRs, or rebase onto the destination containing #71 once it lands; no branch manipulation is requested here.

The loss registry now has three experimental objectives: real-space density, structure factors, and combined Bragg/diffuse scattering. The new diffuse objective is especially relevant evidence that a scalar loss may have its own ensemble reduction and property-use rules; it does not resolve the correct reduction for wE.

Preparation, weighted composition, declarative construction, and their regression tests already exist. The remaining framework work is to clarify which contract assertions actually apply to a geometry loss, not to implement those mechanisms again.

### Existing sources of truth

| Existing component | Relevant behavior |
|---|---|
| [LossFunctionProtocol][loss-contract] | Accepts coordinates and per-atom properties; returns a scalar that is **minimized**. The leading axis is used for an ensemble, not independent training examples. |
| [LossInputs][loss-inputs] | Keeps model-order topology alongside authoritative tensor values. Pipeline occupancies are generally uniform `1/ensemble_size`; the topology's annotations are not a substitute for those tensors. |
| [PreparableLossFunctionProtocol and prepare_loss_if_needed][loss-prepare] | Bind a loss to the inputs/topology it will score. Preparation may be repeated and must not mutate those inputs. |
| [SampleworksProcessedStructure.to_loss_inputs][processed-inputs] | Builds inputs from `loss_atom_array` in model order and carries reconciled reference coordinates and B-factors. |
| [CompositeLoss][composite] and [build_loss][loss-build] | Compose scalar terms and forward preparation. Explicit non-negative weights are used **as supplied**, not renormalized; omitted weights default to `1/N`. A single configured term at a non-unit weight is also scaled through composition. |
| [RealSpaceLossFunction.__call__][density-call] and [StructureFactorLossFunction.__call__][sf-call] | Sum conformer density or complex structure factors before evaluating experimental mismatch. Their ensemble reduction is not automatically appropriate for geometry. |
| [DiffuseBraggLossFunction][diffuse-call] | Computes mean structure factors and diffuse variance across configurations. It multiplies pipeline occupancies by ensemble size before taking those moments, currently assumes uniform configuration weights, and ignores runtime elements/B-factors baked into preparation. Do not copy those rules into wE without evidence. |
| [StepParams.with_loss][step-loss], [DataSpaceDPSScaler][data-dps], and [NoiseSpaceDPSScaler][noise-dps] | Carry `loss_function`/`loss_inputs` and backpropagate the scalar through coordinates, or through the denoiser to noisy coordinates. |
| [FKSteering per-particle guidance][fk-particles] | Gives each particle's ensemble its own guidance call. Independent particles must not be pooled into one geometry score. |
| [LatentOptimization's loss assembly][latent-loss] | Already accepts the composed coordinate loss; a separate geometry penalty is currently added outside it. wE need not introduce another special-case scaler argument. |
| [LossSpec registry][registry], [option schemas][options], and [Losses enum][loss-enum] | Single declaration path for loss construction, CLI/config validation, and serialization. Registration also requires the enum entry. |

The existing [BondGeometryLoss][bond-geometry] is still a coordinate-only, radius-based hinge regularizer, not Holton's statistical geometry model or a conforming `LossFunctionProtocol` implementation. The rename did not complete the composability work in #363. Reuse useful topology/array patterns where justified, but do not equate its ideal distances or clash logic with reference `wE`. Rewriting or replacing it is not a prerequisite.

The [preparation-hook tests][prepare-tests], [loss-topology tests][topology-tests], and [composition tests][composite-tests] are existing regression coverage to extend rather than duplicate. The [shared contract tests][contract-tests] now include diffuse/Bragg in Bragg-only mode on CPU; real-space and structure-factor cases are GPU-marked individually. That does not validate the multi-member diffuse term, and wE's small CPU tests should not inherit a module-wide GPU requirement.

### Minimal contract refinement

Keep the current calling signature and scalar return unless a documented requirement prevents that. In particular:

- Lower values remain better. Implement the current `LossFunctionProtocol` and use loss terminology throughout new code, configuration, tests, and documentation.
- A call receives one ensemble and returns one scalar. Document the selected ensemble semantics after the reference investigation.
- Coordinate differentiability is required for the guidance path on its supported domain. A property that a loss does not use need not have a gradient.
- Elements, B-factors, occupancies, and density-specific precomputation arguments can remain accepted even if some are irrelevant to geometry.
- Whether occupancies influence wE selection/counting is unresolved. Do not promise either occupancy gradients or occupancy independence before choosing that policy.
- Preparation must use model atom identities/order, not deposited-structure indices or cached placeholder coordinates.
- No output dictionaries or diagnostic objects are introduced into the scalar calling contract.
- Distinguish external composition weights, per-atom occupancies, and ensemble population weights. Preserve existing supplied composition weights; do not use their normalization to choose wE's statistical counts or ensemble reduction.

The [shared contract tests][contract-tests] still require occupancy gradients, experimental-target perturbation behavior, and occupancy-split ensemble invariance. Those are not universal geometry-loss laws. Separate genuinely shared requirements from loss-specific assertions, preserving existing real-space, structure-factor, and diffuse/Bragg checks in their applicable groups.

Do not weaken tests merely to make wE pass, add dummy occupancy dependencies to manufacture gradients, or require every geometry term to increase monotonically under arbitrary coordinate perturbations.

The scalar contract still has no diffusion-time argument. Coordinate-dependent chemistry parameters are not diffusion-time schedules; wE does not require adding timestep plumbing or adopting diffuse/Bragg's optional caller-managed `current_time` mechanism.

## 4. Proposed design, alternatives, and tradeoffs

### Recommended component boundaries

Prefer one scientific tensor implementation usable in two execution paths:

```text
configuration + model-order LossInputs
    -> prepare chemistry/topology/reference tables

coordinates + applicable runtime properties
    -> geometry observations
    -> category energies, counts, and statistical weighting
    -> minimized scalar wE
```

- **Reference-compatible scoring:** evaluate the Python implementation without needing autograd, and compare it to externally generated golden values.
- **Differentiable guidance:** evaluate the same scientific operations with an intact coordinate graph.

Two paths do not require duplicate chemistry implementations or a configurable backend framework. Share exact operations wherever practical. If a demonstrated gradient problem requires a different expression, make that deviation explicit and test it against the reference path and external values. Comparing two paths that share code does not replace independent golden-value tests.

### Preparation and runtime work

Preparation should establish stable atom identities, restraint incidence lists, fixed ideal values/sigmas, applicable residue classes, disulfide connectivity, contact exclusions, reference tables, and device tensors. Classify restraint parameters as fixed or coordinate-dependent rather than assuming all targets and sigmas can be cached.

The [reference commands][holton-commands] rebuild Phenix geometry restraints for each candidate. The reference study must capture effective settings, especially whether the conformation-dependent library (CDL) is active. If targets or sigmas depend on backbone conformation, recompute them from current coordinates and retain their applicable gradient dependence; freezing initial values would require an explicitly approved fidelity deviation.

Use structural identity rather than PDB-column parsing or residue-number modulo keys. Include chain, residue number/insertion code, atom name, and applicable conformer/model identity. Missing atoms, chain breaks, and terminal atoms need a documented policy.

The current [AtomReconciler][reconciler] handles label/chemistry differences as well as count/order differences. [Output residue relabeling][output-labels] is a later serialization step, not a replacement topology for loss preparation. Keep prepared indices and diagnostic identities tied to `LossInputs.atom_array`; map to deposited/output labels separately where useful. Matching normalizations such as MET/MSE equivalence do not imply that modified residues are now in wE's initial supported chemistry.

Do not infer new covalent connectivity from transient noisy coordinates during guidance. Existing bonds, trusted residue templates, and reconciled reference topology are preferable sources; disulfide identification needs an explicit decision.

Runtime evaluation must recompute coordinate-dependent quantities. In particular, upstream positive-nonbond and clash selections can change with coordinates. Freezing all category counts or a neighbor list during preparation could change the scientific score or miss newly formed contacts. Determine which memberships are fixed chemistry and which require runtime updates.

Avoid retaining coordinate computation graphs in prepared state between calls. Re-preparation for a different atom set or device must replace stale indices/tensors, not reuse incompatible state.

### Chemistry and reference-data sources

Evaluate **Atomworks 3.0+**, Biotite, and Gemmi explicitly before introducing bespoke tables. The current [project dependency declaration][project-dependencies] pins `atomworks[ml]==2.1.1`; [Atomworks 3.0.4][atomworks-3] is published from RosettaCommons. Its availability does not mean the current environments have been upgraded. Determine whether 3.0+ supplies useful reference data/APIs and whether adopting them requires a focused, separately reviewable dependency upgrade.

| Candidate | Initial assessment and required investigation |
|---|---|
| Atomworks 3.0+ | Evaluate CCD utilities, reference-molecule representations, geometry utilities, and hydrogen/protonation facilities. Inventory actual table fields and provenance rather than assuming that reference coordinates include restraint sigmas or MolProbity probabilities. Check compatibility with every affected model environment before recommending an upgrade. |
| [Biotite chemical information][biotite-info] | Provides CCD access, residue/bond definitions, and radii. Evaluate it for topology and available reference fields; bond connectivity or an idealized residue alone does not establish Phenix-compatible targets, uncertainties, or validation distributions. |
| [Gemmi chemistry/monomer-library support][gemmi-chemistry] | Provides chemical-component and monomer-restraint representations, including reading restraint dictionaries. Evaluate actual available dictionaries, links, targets/sigmas, and redistribution rights separately from the library API. Check the pinned Gemmi version (`0.6.7`) rather than assuming every feature in current development documentation is installed. |
| [TMol][tmol-source] | Offers differentiable coordinate scoring and chemical/parameter machinery. Evaluate reusable components for the initial integration, subject to the observation-level and compatibility gates below; do not adopt its total Rosetta energy as wE. |

For every candidate, record a category-by-category coverage matrix: topology, targets/sigmas, CDL, probability tables, contact conventions, differentiability, licensing, and gaps. Evaluate CCTBX and MolProbity-related data where these candidates leave gaps; no candidate is yet a selected dependency or a demonstrated numerical equivalent.

The source decision must cover:

1. Ideal values, sigmas, periodicities, chirality, and plane definitions, including effective CDL settings and coordinate-dependent parameter evaluation.
2. Atom typing, exclusions, hydrogen bonds, water contacts, and disulfides.
3. Ramachandran/rotamer probability tables, class selection, and interpolation.
4. Cβ ideal-position construction.
5. Redistribution rights, attribution, installation requirements, and model-environment compatibility.

The reference also enables `flip_symmetric_amino_acids=True` in its MolProbity call. Characterize orientation-sensitive cases such as Asn/Gln/His before claiming clash parity. Decide whether Python scoring reproduces a virtual orientation choice or scores the supplied labeling as a documented deviation. A scoring-time virtual choice must not mutate sampled coordinates, topology, or labels: it is distinct from the conformer-swap/topology-changing moves excluded by the anti-goals. Its gradients and behavior at orientation-switch boundaries remain an open decision.

A small set of Phenix-derived test outputs is not a reusable runtime restraint library. Per-protein exported restraints are not an acceptable substitute for general support of the agreed chemistry.

The [Holton repository license][holton-license] is MIT. Preserve attribution when reusing substantial material. External software, chemistry libraries, and probability tables have separate licensing questions; verify permissions before importing or redistributing them.

### Can TMol support the initial integration?

**Assessment: potentially, at the component level; compatibility and score parity are not yet established.** The inspected [TMol revision][tmol-source] exposes Cartesian bonded subgraphs/parameters and has backbone-torsion, Dunbrack, disulfide, and contact terms. These are plausible reuse points for preparation and differentiable geometry; their Rosetta parameterization and statistical meaning are not automatically Holton/Phenix/MolProbity equivalents.

The initial feasibility work should test one small standard-protein case and a water/disulfide case, checking:

1. Atom order, residue identity, missing atoms, hydrogens, and waters survive the adapter. The inspected Cartesian-bonded implementation notes that its readers drop water; an unexamined convenience reader cannot satisfy the agreed water scope.
2. Individual observations, counts, and required parameters can be obtained before category reduction. A whole-pose or per-residue reduced energy cannot recover Holton's maximum, observation count, or squared-sigma distribution.
3. Coordinate gradients survive conversion, and parameters/energies can match pinned reference values or have explicitly approved differences. Evaluate reusable machinery separately from default force-field parameters.
4. A suitable TMol release works with the repository's model environments, CPU/CUDA requirements, and dependency policy. The inspected upstream revision declares `torch>=2.11`, while the [Protpardelle environment][model-torch] requires `<2.8`; do not silently upgrade all models. Check a compatible release or conclude that this route is unsuitable for the initial delivery.
5. Code and bundled-data licenses permit the intended use and distribution.

**Recommendation:** make a bounded TMol reuse assessment part of WE-1, not a prerequisite to rewrite the framework. Use it for the initial scientific implementation only where the coverage/parity/gradient checks justify it. If its public outputs are too reduced, chemistry coverage is unsuitable, or integration requires a disproportionate model-stack upgrade, use the other evaluated data sources with minimal PyTorch operations. Record that decision rather than calling a generic TMol score “partial wE.”

### Ensemble semantics: explicitly unresolved

The following alternatives are materially different:

| Alternative | Consequence |
|---|---|
| Score each member and average | Preserves per-member categories/counts; may not match the reference multi-conformer score. |
| Pool applicable observations before weighting | Changes statistical counts and worst-outlier competition as ensemble size changes. |
| Reproduce the reference's explicit alternate-conformer model | Closest reference representation, but blank/shared atoms, incomplete alternates, occupancy selection, and model-stack conversion need investigation. |

Do not assume an occupancy-weighted mean, pooled maximum, or existing density-mixture rule is correct. Evaluate controlled A/B structures, atom stacks, shared atoms, duplicated identical members, and changed occupancies against the pinned reference.

The outcome must explain both the scalar score and how each member receives gradients. Mutually exclusive conformers must not be treated as simultaneous molecules that clash with one another merely because coordinates share an ensemble tensor.

### Outlier weighting and per-residue wE: decision required

Keep three separate questions explicit: Holton's within-category statistical weights, any additional outlier/category emphasis for guidance, and residue-level reporting or objectives. None is determined by the external `CompositeLoss` weight.

| Candidate behavior | Tradeoff to evaluate |
|---|---|
| Canonical global wE | Retain reference category counts, worst-outlier competition, and compression. This remains the baseline for reference agreement. |
| Attribute the global result to residues | Useful local diagnostics, but define how to assign bonds/angles spanning residues and pair contacts without double counting. Attribution does not imply that each residue has an independently computed wE. |
| Compute wE independently per residue, then aggregate | Local counts and maxima change the statistical weights; sum/mean/max or selected-residue emphasis can produce a genuinely different objective from global wE. Define boundary interactions and ensemble reduction explicitly. |
| Apply additional outlier/category/residue weights for guidance | May target locally concerning geometry, but changes scale, rankings, and gradient allocation. Justify it with controlled cases, and name/version it separately from canonical reference wE. |

WE-1 should compare isolated severe outliers, many moderate deviations, density-supported unusual conformations, protein-size changes, and inter-residue interactions. Record whether per-residue values are diagnostic attribution, independently recomputed local scores, or guidance terms; whether contributions should sum to the global score; and how extra weights are chosen. Leave the choice open (D12). Do not silently change canonical wE or imply that per-residue scores are additive under nonlinear weighting.

This decision does not presume a new configurable objective in the first release. Canonical global wE can proceed if additional weighting/local objectives are explicitly deferred; implement a variant only after its scientific meaning and scope are approved.

### Differentiability and numerical policy

Preserve reference formulas by default. `max`, piecewise compression, and other operations can have useful derivatives away from ties/boundaries; smooth replacements are not mandatory.

Specifically investigate:

- Periodic angle residuals and probability interpolation across angular boundaries.
- Gradients of the chi-square weighting on the actual pinned PyTorch stack.
- Zero-distance Lennard–Jones behavior, degenerate angles/planes, and probability endpoints.
- Coordinate-dependent restraint targets/sigmas and discrete scoring-time orientation choices.
- Changes in contact membership or positive-energy observation counts.
- Max ties, the `E=10` compression boundary, the weight-input cap, and the `N=3200` branch.

Energy compression is continuous at `E=10`, but its derivative is not continuous. Tests must acknowledge such boundaries rather than claim the reference is globally smooth.

Do not quietly stabilize an expression by changing its scientific definition. For invalid/singular input, choose and document supported behavior: a validated exclusion, explicit rejection, reference-preserving limit, or justified guidance-only approximation. Ordinary denoised model structures must remain usable.

### Coverage and diagnostics

Permit incremental category PRs, but distinguish **supported**, **inapplicable**, and **unsupported** categories. Unsupported terms must not silently become zero contributions to a score advertised as full `wE`. Empty legitimate categories, such as no rotamer observations for a glycine-only input, need separate treatment from missing implementations.

As a minor implementation item, provide separate diagnostics for counts, average/worst energies, weights, contributions, coverage, and relevant atom/residue identities. The scalar loss stays scalar; no general observability framework is proposed.

### Alternatives not selected

- **Wrap the external scoring script at runtime:** conflicts with dependency independence and gradient guidance.
- **Substitute an existing generic force field or hinge regularizer:** potentially useful separately, but does not establish Holton-score compatibility.
- **Invent broad loss/backend protocols first:** unsupported by demonstrated needs; use existing preparation and composition.
- **Smooth all terms in advance:** introduces avoidable numerical divergence; require evidence for each substitution.

## 5. Validation strategy and acceptance criteria

### Independent, stored reference fixtures

The initial reference/implementation PR may include manually generated expected values. "Manually generated" means running and inspecting the pinned reference and recording its outputs; it does not mean calculating expectations with the new implementation.

Each fixture should contain or identify:

- Input coordinates/topology, stable atom identities, and a checksum.
- Script revision, Phenix/MolProbity and relevant gnuplot versions, effective command settings (including CDL and flip preprocessing), chemistry/table versions, and hydrogen/conformer/symmetry policy.
- Raw or suitably reduced reference observations, counts, energies, weights, contributions, and total.
- How the fixture was generated, applicable categories, known reference quirks, units, and precision.
- Attribution/redistribution information and justified comparison tolerances.

Normal tests load checked-in data. No external executables, network downloads, licensed-tool discovery, or installed Phenix data directories may be required. Reference regeneration is a documented manual procedure, not a prerequisite for CI.

Published README totals are useful smoke-test context, not precision golden values: some displayed values differ across examples, and tool versions/options are not fully specified.

### Required test layers

| Layer | Evidence required |
|---|---|
| Statistical aggregation | Analytic cases and independent reference values for average/worst weights, coefficients, caps, compression ordering, clash adjustments, empty categories, and count boundaries. |
| Outliers and residue weighting | Controlled cases for severe versus moderate outliers, size-dependent counts, and cross-residue interactions. Verify attribution/no-double-counting rules and, if a local or additionally weighted objective is selected, compare values, rankings, and gradients with canonical global wE without requiring false equivalence. |
| Geometry observations | Compare selected atoms, ideal values/sigmas, observed quantities, energies, and counts per category—not only totals that could conceal compensating errors. Include changed backbone conformations to detect coordinate-dependent restraint parameters. |
| Whole structures | Store pinned-reference outputs for representative UNTANGLE inputs plus controlled standard-protein, disulfide, water, alternate-conformer, and flip-sensitive cases. Select small redistributable fixtures; do not import the entire upstream repository. |
| Gradients | Numerical derivative/`gradcheck` comparisons on small double-precision examples away from discontinuities, plus explicit boundary/degeneracy tests. Require finite gradients on the supported domain; zero gradients at valid minima are allowed. |
| Representation and preparation | Extend existing preparation/topology regression coverage with wE-specific atom-order permutations, same-count/different-identity cases, chain/insertion-code identity, missing/terminal atoms, repeat preparation, input non-mutation, and device changes. Serialization relabeling must not change which atoms are scored. |
| Ensembles | Selected reference semantics, all members participating as specified, no accidental cross-conformer clashes, and independent FK particle handling. Keep unresolved reductions out of claimed acceptance results. |
| Composition and plumbing | Extend existing weighted value/gradient tests with wE: explicit weights remain as supplied, omitted weights remain uniform, and a non-unit-weight single loss is scaled. Verify sign consistency, actual coordinate-scaler backward calls, and differentiation through a small deterministic denoiser/latent computation. Use real computations rather than internal-call mocks. |
| Registration/configuration | Library construction, `--loss-type`/`--loss-config` help/options, `loss_options` validation, serialization, lazy imports, and preparation via the existing registry/composite path. Preserve all three existing experimental registrations. |

Keep small formula, fixture, and gradient tests CPU-capable. CUDA checks validate device behavior when available and are marked appropriately. Full model/checkpoint sampling experiments are not a completion gate for this change.

If symmetry contacts are included, rigid transforms must also transform the crystal reference consistently; do not impose isolated-molecule SE(3) invariance on a fixed crystal-frame contact model.

### Acceptance gates

1. **Reference feasibility:** repeatable external values, licensed/runtime-independent data sources, and an agreed fidelity specification.
2. **Category acceptance:** justified observation/count/value tolerances, required gradient checks, and declared coverage.
3. **Full wE acceptance:** all required applicable categories present and reference-compatible on the declared domain; deviations are enumerated, not hidden by a broad total-score tolerance.
4. **Integration acceptance:** composition and gradient plumbing work through existing loss interfaces, with existing experimental-loss behavior and explicit-weight semantics preserved.

Tolerances, fixture versions, and performance envelopes are intentionally not invented here. Measure them in the reference/feasibility stage. Separate float64 reference comparisons from deployment-dtype tolerances.

## 6. Delivery roadmap: issues, PRs, and implementation steps

Use [#454][issue-454] as the umbrella tracking issue. The IDs below are local roadmap labels, **not filed GitHub issue numbers**. No issues or PRs have been created by this document.

Coordinate with [#363][issue-363] for composable geometry/latent regularizers. Stack the wE implementation on the [#71][issue-71] loss-rename base (or its landed descendant); the rename itself is not a new roadmap task. TMol reuse is assessed directly in WE-1, not tracked through a separate physical-plausibility issue. Do not duplicate unrelated scope or make complete regularizer migration a prerequisite.

### Proposed child issues

| Label and proposed issue title | Concrete scope and completion evidence | Dependencies |
|---|---|---|
| WE-1: Establish pinned wE references and chemistry feasibility | Verify/pin the existing actl/SBGrid Phenix environment and generate independent stored values. Evaluate Atomworks 3.0+, Biotite, Gemmi, and bounded TMol reuse; record data/compatibility gaps, CDL/flip settings, outlier/per-residue weighting choices, hydrogen/symmetry decisions, and ensemble experiments. May land with an initial implementation PR. | Issue-71 loss-rename base for repository-facing implementation |
| WE-2: Clarify geometry loss contract and test applicability | Refine the existing loss API's semantics; separate universal assertions from experimental-only assumptions while preserving all three experimental loss cases and existing preparation/composition behavior. Coordinate the relevant part of #363; no rename work. | WE-1 findings needed for affected semantics |
| WE-3: Implement wE weighting and prepared covalent geometry | Implement reference aggregation and bonds/angles/torsions/chirality/planes in Python/PyTorch, reusing accepted library components. Test counts, reference values, gradients, disulfides, topology lifecycle, and selected residue attribution/weighting behavior without changing canonical global wE implicitly. | Relevant WE-1 source/TMol/weighting decisions; WE-2 only where interface/test changes are needed |
| WE-4: Implement backbone and side-chain validation categories | Omega, Cβ, Ramachandran, and rotamers; justified reference tables/classification/interpolation; independent values and gradients. | Table/chemistry decisions from WE-1; aggregation/preparation from WE-3 |
| WE-5: Implement reference contact and clash categories | Nonbonded and MolProbity-clash semantics, hydrogen/exclusion/symmetry policies, dynamic membership/counts, full-nonbond average behavior, and numerical limits. | Relevant WE-1 decisions; aggregation/preparation from WE-3 |
| WE-6: Complete wE loss integration and acceptance | Full coverage, existing loss registry/options/enum/builder path, `loss_options` configuration, composition and scaler/latent gradient plumbing, cross-environment regression checks, and user/API documentation. | WE-2 through WE-5; ensemble decision resolved |

Scientific work is deliberately split into category families rather than eleven tiny PRs or one opaque all-category PR. WE-4 and WE-5 can proceed independently once shared preparation/aggregation and their own scientific decisions are settled.

### Proposed PR sequence

| PR | Suggested Conventional Commit title | Review boundary and acceptance |
|---|---|---|
| 0 | `docs(losses): design Holton weighted energy integration` | This editable design and decision/roadmap review. No implementation changes. |
| 1 | `test(losses): add pinned Holton weighted energy references` | WE-1 actl-generated fixtures/provenance, reference specification, library coverage/TMol feasibility, and outlier/per-residue weighting findings. Include small initial implementation only if it makes reference validation reviewable; use an appropriate `feat` title if included. No external tools or cluster access in normal tests. |
| 2, if needed | `refactor(losses): clarify geometry loss contract` | WE-2 narrowly scoped contract/test changes on the already-renamed API. Preserve the calling interface and existing experimental assertions. Fold documentation-only clarification into a scientific PR if a separate change is unnecessary. |
| 3 | `feat(losses): add wE weighting and covalent geometry` | WE-3 preparation, aggregation, covalent categories, reference tests, and differentiability. Clearly partial scientific coverage; not a registered full wE loss. |
| 4a | `feat(losses): add wE conformational validation terms` | WE-4 residue validation and gradients. |
| 4b | `feat(losses): add wE nonbonded and clash terms` | WE-5 contact logic, category counts, reference comparison, and gradients. Can run alongside 4a. |
| 5 | `feat(losses): integrate complete Holton weighted energy loss` | WE-6 full scalar loss, minor separate diagnostics, configuration integration, composition/backprop tests, and supported-domain documentation. |

Do not land a PR that depends on an unresolved scientific decision affecting its outputs. Unrelated decisions do not block progress. Human reviewers approve the reference/fidelity policy and any scientific deviations; green tests alone do not settle those choices.

### Expected implementation touchpoints

These are proposed locations, not committed API names or a requirement to create every file:

| Location | Proposed change |
|---|---|
| `tests/resources/weighted_energy/` | Small reference structures, expected observations/results, provenance, and coverage metadata. |
| `tests/losses/test_weighted_energy.py` | Formula, category, whole-score, preparation, and CPU gradient tests; split by category only when size warrants it. |
| `src/sampleworks/core/forward_models/geometry/` | Prepared chemistry and coordinate-to-geometry/validation operations, if separating them from aggregation is justified. Avoid a generic plugin framework. |
| `src/sampleworks/core/losses/weighted_energy.py` | Statistical aggregation and the eventual topology-preparable scalar loss; minor diagnostic reporting and builder. Reuse the same scientific core for scoring and gradients. |
| [protocol.py][loss-contract] and [shared contract tests][contract-tests] | Only demonstrated contract clarification and applicability changes. |
| [options.py][options], [registry.py][registry], [guidance_constants.py][loss-enum] | Final loss options, lazy builder registration, and enum member. Use `--loss-type`, `--loss-config`, and `loss_options`; do not hand-edit argparse dispatch for a normally declared option. |
| Existing [preparation-hook][prepare-tests], [loss-topology][topology-tests], and [composition][composite-tests] tests; loss registry/config/build integration tests | Extend with wE-specific cases; preserve all experimental registrations, supplied weight scaling, topology binding, serialization, and lazy dependency behavior. |
| Integration tests for scalers/composition | Exercise actual scalar backward paths without introducing new sampler mechanics or full-model experiments. |
| [Sphinx/MyST design-note index][docs-index] and [AutoAPI configuration][docs-config] | At integration, publish supported chemistry, ensemble meaning, reference provenance, limitations, and composition examples through the existing documentation site; document new public objects with NumPy-style docstrings. Keep this editable draft in its current blueprint location for now. |

When implementation exists, run the project's Pixi tasks on the selected actl Pod, initially targeted tests and then applicable environment suites. Keep source edits local and verify sync before execution:

```bash
pixi run -e boltz-dev tests "-k weighted_energy"
pixi run test-fast
```

The current `test-fast` task covers Boltz, Protenix, RF3, and Protpardelle development environments. Use [parameter-level CPU/GPU marking][contract-tests] so small wE reference and gradient tests participate in ordinary fast runs.

Optional CUDA-specific validation uses `pixi run -e <model>-dev gpu-tests` on a suitable GPU machine. These are **future implementation checks**, not checks already run for this document. Use `jj` for changes and review; do not create issues, open PRs, commit, or publish without the corresponding request.

## 7. Open questions and decision gates

The approved direction does not settle the following scientific/API details. Record the chosen answer, evidence, and consequences here as work proceeds.

| ID | Open decision | Evidence needed / recommended starting point | Blocks |
|---|---|---|---|
| D1 | Which existing actl/SBGrid reference environment is pinned? | Phenix is available through the Pod environment; verify the selected image digest, mounted software/versions, Phenix/MolProbity, chemistry tables, gnuplot/tcsh, and effective settings. Demonstrate repeatability on a small fixture. Reference-generation owner is not assigned. | Trustworthy golden values and numerical acceptance |
| D2 | Which restraint and validation data sources can ship independently, and which parameters depend on conformation? | Evaluate Atomworks 3.0+ (including upgrade compatibility from the current 2.1.1 pin), Biotite, and Gemmi before bespoke tables. Capture effective CDL settings; compare targets/sigmas across conformations; classify fixed/runtime parameters and gradients; verify source rights and operation without Phenix/MolProbity. | Covalent preparation/runtime evaluation; Rama/rotamer/Cβ/contact implementations |
| D3 | How do reference multi-conformer semantics map to ensemble tensors? | Controlled A/B, shared-atom, stack, identical-member, and occupancy experiments. Compare pooled versus per-member aggregation and gradient allocation. | Ensemble weighting/counts; final integration |
| D4 | Are hydrogens supplied, reconstructed, or excluded by category? | Compare explicit/absent-H inputs and reference hydrogen preparation. If reconstruction is needed during guidance, it must track changing heavy-atom coordinates and propagate applicable gradients. | Hydrogen-dependent contacts and full reference coverage |
| D5 | Are crystal-symmetry contacts in scope, and where does crystal metadata enter? | Determine reference behavior with symmetry mates/waters. `LossInputs` still has no explicit cell/space-group fields; examine existing crystallographic-loss and [metadata-resolution][crystal-metadata] patterns before proposing a minimal configuration source. Do not couple geometry construction to the batch synthetic API merely to obtain metadata. | Symmetry-aware contacts and invariance tests |
| D6 | How are reference quirks distinguished from unintended parser limitations? | Characterize active formulas, clash ordering, full-nonbond averages, probability caps, chain filters, and modulo residue handling. Recommend stable identities in Python; document where reference parser limitations make comparison unsuitable. | Exact fidelity specification and affected goldens |
| D7 | What are contact membership, exclusions, and category-count semantics? | Determine reference neighbor ranges, positive-energy subset changes, mutually exclusive conformers, hydrogen bonds, and occupancy filtering. Do not freeze dynamic memberships without evidence. | Contact scores/gradients and ensemble acceptance |
| D8 | How are chain breaks, missing/terminal atoms, and disulfides represented? | Reference cases and available topology annotations/templates. Respect existing model-space reconciliation and separate output relabeling; choose deterministic preparation and distinguish inapplicable categories from unsupported chemistry. | Covalent preparation and supported-domain claims |
| D9 | Are any guidance-only numerical substitutions necessary? | Finite-difference and boundary tests on actual tensor operations; measure any change against reference values. Default to preserving the reference, including piecewise behavior. | Affected differentiable categories |
| D10 | What tolerances and practical performance envelopes are justified? | Measure per-observation/category/total differences and CPU/CUDA dtype effects; assess contact memory/runtime on representative supported sizes. Do not mandate a hardware budget without evidence. | Final acceptance and documented practical limits |
| D11 | How is MolProbity's discrete orientation preprocessing reproduced without mutating the sampled structure? | Capture `flip_symmetric_amino_acids=True` behavior on flip-sensitive fixtures; compare virtual orientation selection against supplied labeling, and specify gradients at switch boundaries. Record any deviation rather than silently disabling preprocessing. | Clash parity and differentiable contact evaluation |
| D12 | Should outliers receive additional guidance weight, and what does per-residue wE mean? | Compare canonical global scoring, residue attribution, independently recomputed local scores, and extra outlier/category/residue weighting. Specify counts/maxima, boundary interactions, additivity, scale, ensemble reduction, and gradient allocation; name any changed objective separately from reference wE. | Scientific weighting specification and any public local-score/guidance API |
| D13 | Can TMol supply components for the initial wE integration? | Prototype observation-level access and autograd on a protein plus water/disulfide cases; compare parameters/counts with the reference. Check topology preservation, a compatible TMol/PyTorch release, CPU/CUDA behavior, and code/data licensing. Choose component reuse or a documented alternative; do not substitute Rosetta total energy. | Selected preparation/geometry implementation approach, not the entire framework |

### Decision record

Use the following short fields when resolving a decision; no separate tracking system is required:

- **Decision ID and status:**
- **Chosen behavior and alternatives rejected:**
- **Evidence / fixture / reference versions:**
- **Effect on fidelity, gradients, supported scope, and PR dependencies:**
- **Reviewer / unresolved follow-up:**

### Current confidence

Confidence is high (about 90%) that the current loss preparation, scalar composition, and registry mechanisms are sufficient without a broad interface rewrite. This refresh is based on source and test inspection, not a new runtime test run. Full wE is not present in the inspected source; confidence in numerical parity and the correct ensemble/contact policy remains unquantified until the pinned reference study. The upstream scientific questions, including CDL and discrete flip preprocessing, remain open.

[issue-454]: https://github.com/prism-science/sampleworks/issues/454
[issue-363]: https://github.com/prism-science/sampleworks/issues/363
[issue-71]: https://github.com/prism-science/sampleworks/issues/71
[atomworks-3]: https://github.com/RosettaCommons/atomworks/tree/v3.0.4
[biotite-info]: https://www.biotite-python.org/latest/apidoc/biotite.structure.info.html
[gemmi-chemistry]: https://gemmi.readthedocs.io/en/latest/chemistry.html#monomer-library
[tmol-source]: https://github.com/uw-ipd/tmol/tree/8a42677a7bfd88230dbf6f8f1fc5df501ea2f911
[project-dependencies]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/pyproject.toml#L42-L53
[model-torch]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/pyproject.toml#L202-L209
[astera-sbgrid]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/Dockerfile.astera#L100-L107
[holton-script]: https://github.com/jmholton/UnTangle/blob/973585629b7a33efe42a68843356c0e953f66c41/untangle_score.csh
[holton-commands]: https://github.com/jmholton/UnTangle/blob/973585629b7a33efe42a68843356c0e953f66c41/untangle_score.csh#L278-L291
[holton-extraction]: https://github.com/jmholton/UnTangle/blob/973585629b7a33efe42a68843356c0e953f66c41/untangle_score.csh#L299-L383
[holton-transforms]: https://github.com/jmholton/UnTangle/blob/973585629b7a33efe42a68843356c0e953f66c41/untangle_score.csh#L395-L461
[holton-weights]: https://github.com/jmholton/UnTangle/blob/973585629b7a33efe42a68843356c0e953f66c41/untangle_score.csh#L670-L830
[holton-license]: https://github.com/jmholton/UnTangle/blob/973585629b7a33efe42a68843356c0e953f66c41/LICENSE
[loss-contract]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/losses/protocol.py#L222-L271
[loss-inputs]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/losses/protocol.py#L18-L43
[loss-prepare]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/losses/protocol.py#L274-L332
[processed-inputs]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/utils/structure_utils.py#L58-L122
[composite]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/losses/composite.py#L18-L132
[loss-build]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/losses/config.py#L425-L460
[density-call]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/losses/real_space_density.py#L356-L397
[sf-call]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/losses/structure_factor.py#L473-L578
[diffuse-call]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/losses/diffuse_bragg.py#L671-L737
[step-loss]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/samplers/protocol.py#L64-L117
[data-dps]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/scalers/step_scalers.py#L51-L97
[noise-dps]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/scalers/step_scalers.py#L100-L165
[fk-particles]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/scalers/fk_steering.py#L347-L387
[latent-loss]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/scalers/latent_optimization.py#L586-L636
[registry]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/losses/registry.py#L53-L129
[options]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/losses/options.py#L69-L168
[loss-enum]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/utils/guidance_constants.py#L59-L64
[bond-geometry]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/core/losses/geometry.py#L50-L66
[contract-tests]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/tests/losses/test_loss_function_contract.py#L1-L140
[prepare-tests]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/tests/losses/test_prepare_hook.py#L20-L56
[topology-tests]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/tests/utils/test_loss_atom_array.py#L48-L79
[composite-tests]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/tests/losses/test_composite.py#L57-L85
[reconciler]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/utils/atom_reconciler.py#L85-L136
[output-labels]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/utils/guidance_script_utils.py#L178-L256
[crystal-metadata]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/src/sampleworks/synthetic/synthetic_utils.py#L719-L744
[docs-index]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/docs/index.md#L7-L20
[docs-config]: worktree://eb7256af-33c5-457e-80c1-c3707845ac22/docs/conf.py#L6-L32
