"""TorchRef-backed reward functions for reciprocal-space fitting.

Provides :class:`TorchRefXrayRewardFunction`, which scores coordinates against
experimental structure factors using torchref's scaling and maximum-likelihood stack:
per-resolution-bin scale, anisotropic scale tensor, refined bulk solvent, and a
sigma_A target carrying a model-error term.

A ``torchref.model.ModelFT`` is built directly rather than through ``Model.load()``.
Cell and space group come from the MTZ, a hand-built ``pdb`` DataFrame carries the
topology, and coordinates and occupancies are caller-owned tensors held in
:class:`_TensorSlot`, so gradients flow from ``target()`` back to the caller's tensor.

Structure factors are linear over atoms::

    F(h) = sum_j f_j occ_j exp(2 pi i h . x_j)

so ``C`` conformers at occupancy ``1/C`` form a single structure-factor calculation
over a ``C * n_atoms`` stack rather than ``C`` separate ones. The stack is
conformer-major, matching the ``[batch, n]`` layout of the reward protocol.

Scale, bulk solvent, sigma_A and ADPs are nuisance parameters refit periodically by
:meth:`TorchRefXrayRewardFunction.refresh_nuisance_parameters`. The objective changes
discontinuously at each refresh.

Usage constraints
-----------------
- Returns an unnormalized summed negative log-likelihood, O(1e5) for a real dataset,
  not sign-constrained. Guidance step sizes do not transfer from MSE-valued rewards.
- sigma_A is fitted on the free set, so ``R_free`` is a diagnostic here rather than an
  independent validation statistic.
- The bulk-solvent mask is built at the current coordinates under ``no_grad``. Use
  ``bulk_solvent=False`` when guidance starts from near-noise coordinates.
- ADPs are refined during maintenance, one per asymmetric-unit atom shared across
  conformers. ``adp_weight=0`` freezes them at ``b_factor``.
- Geometry restraints are off by default (``geometry_weight=0``). A zero weight means
  the target is never constructed.
"""

from __future__ import annotations

import functools
from collections import Counter
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from loguru import logger
from sampleworks.utils.elements import elements_to_scattering_indices


if TYPE_CHECKING:
    from biotite.structure import AtomArray


# torchref restricts the scale-fit objective to these two: an alpha-centred mode is
# degenerate with the scale being fitted and drives it to absorb 1/alpha.
_SCALE_TARGETS = ("nll", "ml_noalpha")

# sigma_A maximum-likelihood rows carry a model-error term; "nll"/"ls" do not but are
# stateless (their maintenance() is a no-op).
_TARGET_MODES = ("ml", "ml_noalpha", "ml_full", "nll_beta", "nll", "ls", "ls_wunit_k1")

DEFAULT_B_FACTOR = 20.0
"""Initial isotropic ADP (Å²) assigned to every atom."""


class _TensorSlot:
    """Callable holder for a caller-owned tensor, read by ``ModelFT`` as a wrapper.

    Returns the held tensor verbatim rather than detaching it, so autograd reaches the
    caller's leaf. Implements the three members ``ModelFT`` requires: the call itself
    (``Model.get_iso``), ``refinable_params`` (``ModelFT._check_forward_dtype``
    float-dtype probe) and ``fixed_values`` (``Model.get_aniso`` placeholder sizing).

    Not an ``nn.Module``, so the held tensors stay out of ``model.parameters()`` and
    cannot be picked up by an optimizer.

    Parameters
    ----------
    t : torch.Tensor
        The held tensor. Reassign :attr:`t` to rebind; nothing is copied.
    """

    __slots__ = ("t",)

    def __init__(self, t: torch.Tensor):
        self.t = t

    def __call__(self) -> torch.Tensor:
        return self.t

    @property
    def refinable_params(self) -> torch.Tensor:
        """Float-dtype probe for ``ModelFT._check_forward_dtype``."""
        return self.t

    @property
    def fixed_values(self) -> torch.Tensor:
        """Dtype/device/shape source for ``Model.get_aniso``'s empty placeholders."""
        return self.t


@functools.lru_cache(maxsize=1)
def _shared_adp_cls() -> type:
    """Build (once) the shared-ADP wrapper. Lazy so importing this module needs no torchref.

    Holds one refinable B per asymmetric-unit atom and expands it across every conformer
    in the stack, so the same atom carries one B-factor rather than C independent ones.

    Subclasses ``PositiveMixedTensor`` so the refinable leaf keeps torchref's ADP
    parameterisation (log-space, positive by construction) -- the form
    ``parameters_of_types(("adp",))`` and the ADP restraint targets expect.
    """
    from torchref.model.parameter_wrappers import PositiveMixedTensor

    class _SharedADP(PositiveMixedTensor):
        """``(n_asu,)`` refinable log-B, expanded to ``(n_conformers * n_asu,)``."""

        def __init__(self, *args: Any, n_conformers: int = 1, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._n_conformers = int(n_conformers)

        def forward(self) -> torch.Tensor:
            """Return the shared B-factors expanded to ``(n_conformers * n_asu,)``."""
            # repeat, not repeat_interleave: the stack is conformer-major. Gradients from
            # all C copies sum onto the shared leaf.
            #
            # Side effect: the ADP restraint targets read model.adp() and so see the
            # expanded (C*n_asu,) array, evaluating each restraint C times over identical
            # values. The effective ADP weight is C x its nominal value, and adp_weight is
            # therefore not comparable across conformer counts.
            per_asu = super().forward()
            if self._n_conformers == 1:
                return per_asu
            return per_asu.repeat(self._n_conformers)

    return _SharedADP


@functools.lru_cache(maxsize=1)
def _external_model_cls() -> type:
    """Build (once) the ``ModelFT`` subclass this module drives.

    Lazy so importing this module needs no torchref. Adds coordinate/occupancy setters
    and disables ``CachedForwardMixin``'s forward cache. That cache fingerprints only
    ``parameters()`` and ``buffers()``; caller-owned coordinates are neither, so it would
    never invalidate and every call after the first would return a stale F_calc. Each
    call now builds a fresh graph, which also removes the mixin's ``retain_graph``
    requirement on a second backward.
    """
    from torchref.model import ModelFT

    class _ExternalModelFT(ModelFT):
        """``ModelFT`` driven by caller-owned tensors, with the forward cache off."""

        def set_coordinates(self, xyz: torch.Tensor) -> None:
            """Bind ``xyz``. The single place coordinates are set."""
            self.xyz.t = xyz

        def set_occupancies(self, occ: torch.Tensor) -> None:
            """Bind ``occ``. The single place occupancies are set."""
            self.occupancy.t = occ

        def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            kwargs.pop("recalc", None)  # the cache is off; recalc is meaningless
            return self.forward(*args, **kwargs)

    return _ExternalModelFT


def _conformer_tag(i: int) -> str:
    """Single-character conformer label: A..Z then a..z, wrapping past 52."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return alphabet[i % len(alphabet)]


def _resolve_structure(structure: AtomArray | str | Path) -> AtomArray:
    """Return an ``AtomArray``, loading from a PDB/mmCIF path if given one.

    Parameters
    ----------
    structure
        An ``AtomArray``, or a path to a structure file.

    Returns
    -------
    AtomArray
        The atom array to take topology from.
    """
    if isinstance(structure, (str, Path)):
        from sampleworks.utils.atom_array_utils import load_structure_with_altlocs

        logger.info(f"Loading topology from {structure}")
        return load_structure_with_altlocs(Path(structure))
    return structure


class TorchRefXrayRewardFunction:
    """Reciprocal-space reward using torchref's scaling and likelihood stack.

    Scores an ensemble of conformers against experimental amplitudes from an MTZ.
    Construction is two-phase: ``__init__`` reads the reflection data, :meth:`prepare`
    takes the topology, and the torchref model is built on first call once the ensemble
    size is known.

    Parameters
    ----------
    mtzfile
        Path to the MTZ holding the target amplitudes. The MTZ is the sole authority for
        the unit cell and space group; there are no arguments to override them.
    structure
        Optional topology (``AtomArray`` or path to PDB/mmCIF). When given,
        :meth:`prepare` is called immediately. Omit it to defer to an explicit
        :meth:`prepare`.
    device
        Device to place the reflection data and model on. Only used when
        ``structure`` is given; otherwise pass it to :meth:`prepare`.
    resolution
        High-resolution cutoff (dmin, Å). ``None`` (default) keeps the MTZ's own range.
        Also sets the density grid spacing.
    target_mode
        torchref X-ray target. ``"ml"`` (default) is the sigma_A maximum-likelihood row
        and carries a model-error term. ``"nll"`` and ``"ls"`` are stateless and cheaper
        but do not penalise overfitting.
    bulk_solvent
        Fit and apply a bulk-solvent contribution. Default True. Costs a real-space mask
        build plus an FFT on every refresh, and is non-differentiable with respect to
        coordinates.
    b_factor
        Initial isotropic ADP (Å²) shared by every atom.
    nbins
        Resolution bins for the scale. torchref may lower this for sparse data.
    refresh_interval
        Refit the nuisance parameters every this many calls; the first call always
        refreshes.
    scale_target
        Objective for the scale fit; must be one of ``("nll", "ml_noalpha")``.
    use_set
        Reflection subset the loss is summed over: ``"work"`` (default), ``"free"``
        or ``"val"``.
    french_wilson
        Derive amplitudes from intensities via French-Wilson. Forwarded to
        ``ReflectionData.load_mtz``.
    column_names
        Explicit MTZ column mapping, e.g. ``{"F": "Fprotein", "SIGF": "SIGFprotein"}``.
        Required when the MTZ carries more than one amplitude set.

    Raises
    ------
    ValueError
        For an unknown ``target_mode``/``scale_target``/``use_set``, a
        non-positive ``refresh_interval`` or ``b_factor``, or an MTZ without a
        usable cell or space group.
    """

    def __init__(
        self,
        mtzfile: str | Path,
        *,
        structure: AtomArray | str | Path | None = None,
        device: torch.device | str | None = None,
        resolution: float | None = None,
        target_mode: str = "ml",
        bulk_solvent: bool = True,
        b_factor: float = DEFAULT_B_FACTOR,
        nbins: int = 20,
        refresh_interval: int = 10,
        scale_target: str = "nll",
        adp_weight: float = 0.02,
        geometry_weight: float = 0.0,
        use_set: str = "work",
        french_wilson: bool = True,
        column_names: dict | None = None,
    ):
        if target_mode not in _TARGET_MODES:
            raise ValueError(f"target_mode must be one of {_TARGET_MODES}, got {target_mode!r}.")
        if scale_target not in _SCALE_TARGETS:
            raise ValueError(
                f"scale_target must be one of {_SCALE_TARGETS}, got {scale_target!r}. An "
                "alpha-centred mode is degenerate with the scale being fitted."
            )
        if use_set not in ("work", "free", "val"):
            raise ValueError(f"use_set must be 'work', 'free' or 'val'; got {use_set!r}.")
        if refresh_interval < 1:
            raise ValueError(f"refresh_interval must be >= 1, got {refresh_interval}.")
        if b_factor <= 0:
            raise ValueError(f"b_factor must be positive, got {b_factor}.")
        if adp_weight < 0 or geometry_weight < 0:
            raise ValueError(
                f"restraint weights must be >= 0; got adp_weight={adp_weight}, "
                f"geometry_weight={geometry_weight}. Use 0 to disable a group entirely."
            )

        self.mtzfile = str(mtzfile)
        self.target_mode = target_mode
        self.bulk_solvent = bulk_solvent
        self.b_factor = float(b_factor)
        self.nbins = nbins
        self.refresh_interval = refresh_interval
        self.scale_target = scale_target
        self.use_set = use_set
        self.adp_weight = float(adp_weight)
        self.geometry_weight = float(geometry_weight)
        # ADPs are refinable only when something regularises them; on the x-ray term
        # alone they would absorb model error.
        self.refine_adp = self.adp_weight > 0

        self._load_reflection_data(resolution, french_wilson, column_names)

        # Populated by prepare().
        self._prepared = False
        self._element_symbols: list[str] | None = None
        self._expected_codes: torch.Tensor | None = None
        self.n_atoms: int | None = None
        self.device: torch.device | None = None
        # Keyed on ensemble size; see _stack_for. The call counter is per-stack because
        # nuisance parameters belong to a triple, so a newly built one is cold-fitted on
        # its first use regardless of any other stack's count.
        self._stacks: dict[int, tuple[Any, Any, Any]] = {}
        self._calls: dict[int, int] = {}
        self._states: dict[int, Any] = {}

        if structure is not None:
            self.prepare(structure, device=device or "cpu")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _load_reflection_data(
        self, resolution: float | None, french_wilson: bool, column_names: dict | None
    ) -> None:
        """Read the MTZ into a ``ReflectionData`` and cache its crystal metadata.

        ``ReflectionData.__post_init__`` runs ``setup_scale()`` and ``setup_anisotropy()``,
        so ``get_corrected_data()``, which the sigma_A rows require, works from here on.
        """
        from torchref import read_mtz

        data = read_mtz(
            self.mtzfile, verbose=0, french_wilson=french_wilson, column_names=column_names
        )
        if resolution is not None:
            data = data.filter_by_resolution(d_min=resolution)
        if data.cell is None:
            raise ValueError(f"{self.mtzfile} carries no unit cell; cannot compute d-spacings.")
        if data.spacegroup is None:
            raise ValueError(f"{self.mtzfile} carries no space group gemmi can recognise.")

        self._data = data
        self.unit_cell = data.cell.data.detach().cpu().tolist()
        self.space_group = data.spacegroup.hm
        # dmin drives the density grid spacing as well as the resolution range, so
        # ModelFT's 1.0 A default would build a needlessly fine grid for lower-res data.
        self.resolution = float(resolution if resolution is not None else data.d_min)

        n_free = int((data.rfree_flags == 0).sum())
        logger.info(
            f"Loaded {self.mtzfile}: n_reflections={len(data.hkl)}, "
            f"dmin={self.resolution:.2f}A, cell={[round(c, 2) for c in self.unit_cell]}, "
            f"space_group={self.space_group}, n_free={n_free}"
        )
        if self.target_mode in ("ml", "ml_noalpha", "ml_full", "nll_beta") and n_free < 100:
            logger.warning(
                f"Only {n_free} free reflections. sigma_A is fitted on the free set; below "
                "~100 the per-shell fit degenerates to a single conservative value, which "
                "silently removes the model-error term's resolution dependence."
            )

    def prepare(
        self, structure: AtomArray | str | Path, *, device: torch.device | str = "cpu"
    ) -> None:
        """Take the topology and place the reflection data on ``device``.

        Caches the element symbols, their scattering-table codes, the per-atom
        annotations the restraint builder needs, and a coordinate snapshot.

        Idempotent: clears the model cache, so it is also how a built reward is moved to
        another device.

        Parameters
        ----------
        structure
            ``AtomArray`` or path to a PDB/mmCIF. Its atom order defines the column
            order of the coordinate tensor passed to :meth:`__call__`.
        device
            Device for the reflection data and every model built from here.

        Raises
        ------
        ValueError
            If the structure has no atoms.
        """
        atom_array = _resolve_structure(structure)
        if len(atom_array) == 0:
            raise ValueError("Structure has no atoms.")

        # Resolve to a concrete device so the per-call check compares like with like:
        # torch.device("cuda") != torch.device("cuda:0") even though a tensor placed on
        # the former reports the latter.
        self.device = torch.zeros(0, device=torch.device(device)).device
        self._element_symbols = [str(e).strip() for e in atom_array.element]
        self.n_atoms = len(self._element_symbols)

        codes = elements_to_scattering_indices(self._element_symbols)
        self._expected_codes = torch.tensor(codes, dtype=torch.long, device=self.device)

        # Per-atom annotations the restraint builder needs, cached once. Biotite always
        # provides these on a parsed structure; `ins_code` may be absent on arrays built
        # by hand, so default it to blank rather than failing.
        self._topology = {
            "atom_name": [str(v).strip() for v in atom_array.atom_name],
            "res_name": [str(v).strip() for v in atom_array.res_name],
            "res_id": [int(v) for v in atom_array.res_id],
            "chain_id": [str(v).strip() for v in atom_array.chain_id],
            "ins_code": [
                str(v).strip() for v in getattr(atom_array, "ins_code", [""] * self.n_atoms)
            ],
        }
        # Snapshot coordinates: the restraint build reads pdb[["x","y","z"]] to count
        # each parent's heavy neighbours for the riding-hydrogen topology.
        self._input_coords = np.ascontiguousarray(np.asarray(atom_array.coord, dtype=float))

        self._data = self._data.to(self.device)

        # Invalidate anything built for a previous topology or device.
        self._stacks = {}
        self._calls = {}
        self._prepared = True

        # Index 0 is the zero-scattering '?' row; those atoms contribute nothing to F_calc.
        n_unknown = int(self._expected_codes.eq(0).sum())
        if n_unknown:
            unknown = sorted({s for s, c in zip(self._element_symbols, codes) if c == 0})
            logger.warning(
                f"{n_unknown} atoms have no scattering factors (elements {unknown}) and will "
                "contribute zero density."
            )
        composition = ", ".join(
            f"{el}:{n}" for el, n in sorted(Counter(self._element_symbols).items())
        )
        logger.info(
            f"Prepared TorchRefXrayRewardFunction: n_atoms={self.n_atoms}, "
            f"device={self.device}, target_mode={self.target_mode}, "
            f"bulk_solvent={self.bulk_solvent}, b_factor={self.b_factor}, "
            f"composition=({composition})"
        )

    # ------------------------------------------------------------------
    # Model / scaler / target construction
    # ------------------------------------------------------------------
    def _build_pdb_dataframe(self, n_conformers: int) -> pd.DataFrame:
        """Assemble the atom table restraint building reads, conformer-major.

        Carries only the columns torchref reads on these paths:
        ``Model._build_restraints`` needs ``name, index, chainid, resseq, resname, icode,
        x, y, z, element, ATOM, altloc``; ``Model.Z`` and ``get_vdw_radii`` need
        ``element``.

        Each conformer gets its own chain id and altloc letter. Distinct chain ids keep
        residue grouping and peptide links within a conformer, since residues are found by
        contiguous run over a ``chainid_resseq`` key and links by ``resseq + 1``. The
        altloc letter suppresses van der Waals restraints between conformers, which are
        otherwise built for every near-coincident duplicated atom pair.

        ``index`` is an explicit column of positional indices, used by the restraint
        builders to index the coordinate tensor directly. ``x, y, z`` hold the
        prepare-time coordinate snapshot, read by ``build_hydrogen_topology`` to count
        each parent's heavy neighbours.
        """
        n = self.n_atoms
        base = self._topology  # per-ASU-atom annotation arrays, from prepare()
        # Chain ids: keep the original for a single conformer so the common case reads
        # naturally; suffix per conformer only when there is a stack to disambiguate.
        chain_ids, altlocs = [], []
        for c in range(n_conformers):
            if n_conformers == 1:
                chain_ids.extend(base["chain_id"])
                altlocs.extend([""] * n)
            else:
                tag = _conformer_tag(c)
                chain_ids.extend(f"{cid}{tag}" for cid in base["chain_id"])
                altlocs.extend([tag] * n)

        coords = self._input_coords  # (n, 3) float, prepare-time snapshot
        xyz = np.tile(coords, (n_conformers, 1))
        df = pd.DataFrame(
            {
                "ATOM": ["ATOM"] * (n * n_conformers),
                "name": list(base["atom_name"]) * n_conformers,
                "altloc": altlocs,
                "resname": list(base["res_name"]) * n_conformers,
                "chainid": chain_ids,
                "resseq": list(base["res_id"]) * n_conformers,
                "icode": list(base["ins_code"]) * n_conformers,
                "x": xyz[:, 0],
                "y": xyz[:, 1],
                "z": xyz[:, 2],
                "element": self._element_symbols * n_conformers,
            }
        )
        # Positional, NOT the pandas index -- see the docstring.
        df["index"] = np.arange(len(df), dtype=int)
        return df

    def _build_model(self, n_conformers: int) -> Any:
        """Build a ``ModelFT`` over a ``n_conformers * n_atoms`` conformer stack.

        Sets up only what the structure-factor and scaling paths read. ``Model.load()`` is
        bypassed because its hydrogen stripping and NaN-row dropping would change the atom
        count and de-align the caller's coordinate tensor.

        Parameters
        ----------
        n_conformers
            Conformers sharing the topology. Occupancy defaults to ``1 / n_conformers``
            per atom, which is what makes the stack equal the ensemble mean.

        Returns
        -------
        ModelFT
            An initialised model whose coordinate and occupancy slots are
            :class:`_TensorSlot` holders.
        """
        from torchref.symmetry import Cell

        n_total = n_conformers * self.n_atoms
        dev = self.device

        # wavelength=None: ModelFT defaults to 1.0, which applies the dispersive f'
        # correction on every forward. Nothing here wants that.
        model = _external_model_cls()(
            verbose=0, wavelength=None, max_res=self.resolution, device=dev
        )
        model.cell = Cell(self.unit_cell, dtype=model.dtype_float, device=dev)
        # Setter builds the SfFFT submodule once cell and space group are both set.
        model.spacegroup = self.space_group

        model.pdb = self._build_pdb_dataframe(n_conformers)
        model.initialized = True  # gates Z / _build_parametrization

        model.register_buffer("aniso_flag", torch.zeros(n_total, dtype=torch.bool, device=dev))
        model._rebuild_sf_indices()  # _iso_indices / _iso_covers_all / _aniso_is_empty

        # torchref's symbol -> Z map differs from the scattering table used above, so an
        # ionic form resolved there can still land on Z=0 here. Checked once per model.
        n_unknown_z = int(model.Z.eq(0).sum())
        if n_unknown_z:
            missing = sorted({s for s, z in zip(model.pdb["element"], model.Z.tolist()) if z == 0})
            logger.warning(
                f"{n_unknown_z} atoms have no atomic number in torchref's scattering table "
                f"(elements {missing}) and will contribute zero density to F_calc."
            )

        full = functools.partial(torch.full, (n_total,), dtype=model.dtype_float, device=dev)
        if self.refine_adp:
            # One refinable B per ASU atom, broadcast across the stack. Constructed with
            # n_atoms values -- not n_total -- which is what makes the leaf shared.
            model.adp = _shared_adp_cls()(
                torch.full((self.n_atoms,), self.b_factor, dtype=model.dtype_float),
                name="adp",
                device=dev,
                n_conformers=n_conformers,
            )
        else:
            model.adp = _TensorSlot(full(self.b_factor))
        model.occupancy = _TensorSlot(full(1.0 / n_conformers))
        model.xyz = _TensorSlot(torch.zeros(n_total, 3, dtype=model.dtype_float, device=dev))

        # Called explicitly: _late_symmetry_compatible starts as None and is set only in
        # setup_grid(), but compute_structure_factors reads it before build_density_map
        # lazily calls setup_grid. Without this the first call takes the early-symmetry
        # path and later calls take late symmetry -- same answer, ~5x the cost.
        model.setup_grid()
        return model

    def _stack_for(self, n_conformers: int) -> tuple[Any, Any, Any]:
        """Return the cached ``(model, scaler, target)`` for this ensemble size.

        The triple is cached as a unit so each set stays internally consistent; sharing
        one scaler across models would require re-pointing ``Scaler.model``, the
        ``SolventModel``'s model and the target's ``_model`` submodule, and rebuilding the
        solvent mask. With a constant ensemble size only one triple is ever built.
        """
        if n_conformers in self._stacks:
            return self._stacks[n_conformers]

        from torchref.refinement.targets import create_xray_target
        from torchref.scaling import Scaler

        logger.info(f"Building torchref model stack for {n_conformers} conformer(s)")
        model = self._build_model(n_conformers)
        scaler = Scaler(model, self._data, nbins=self.nbins, verbose=0, device=self.device)
        # The model is attached to the target as well; the solvent mask reads it directly.
        target = create_xray_target(
            data=self._data,
            model=model,
            scaler=scaler,
            mode=self.target_mode,
            use_set=self.use_set,
            verbose=0,
            device=self.device,
        )
        self._stacks[n_conformers] = (model, scaler, target)
        self._states[n_conformers] = self._build_loss_state(model, target)
        return self._stacks[n_conformers]

    def _build_loss_state(self, model: Any, xray_target: Any) -> Any:
        """Assemble the weighted ``LossState`` over x-ray and optional restraint groups.

        Follows ``base_refinement._create_loss_state`` so naming, weighting and
        maintenance semantics match torchref's.

        Restraint groups are registered via ``register_targets`` without a name: the leaf
        targets self-name ``geometry/bond``, ``adp/simu`` and so on, and passing a name
        would double-prefix them, which stops ``"geometry/ramachandran": 0.0`` matching
        and re-enables Ramachandran at the group weight.

        ``set_weights`` is required. ``LossState.weights`` defaults to ``{}`` with a
        per-name lookup default of 1.0, which would run the ADP prior 50x too strong.
        Weights compose multiplicatively down the ``/``-separated path.

        Groups are gated on their weight at registration rather than evaluation, because
        ``register_target`` probes each target once and that first call is what builds the
        restraint graph, riding-hydrogen topology and van der Waals pair list.
        """
        from torchref.refinement.loss_state import LossState
        from torchref.refinement.targets import TotalADPTarget, TotalGeometryTarget

        state = LossState(device=self.device)
        state.register_target("xray", xray_target)

        if self.adp_weight > 0:
            state.register_targets(TotalADPTarget(model, verbose=0))
        if self.geometry_weight > 0:
            state.register_targets(TotalGeometryTarget(model, verbose=0))

        state.set_weights(
            {
                "xray": 1.0,
                "adp": self.adp_weight,
                "geometry": self.geometry_weight,
                # torchref disables Ramachandran by default; keep that.
                "geometry/ramachandran": 0.0,
            }
        )

        groups = sorted(state.targets)
        logger.info(
            f"Loss groups: {groups} "
            f"(xray=1.0, adp={self.adp_weight}, geometry={self.geometry_weight})"
        )
        if self.adp_weight > 0 or self.geometry_weight > 0:
            logger.info(f"Restraint counts: {self._restraint_counts(model)}")
        return state

    @staticmethod
    def _restraint_counts(model: Any) -> dict[str, int]:
        """Per-type restraint counts, for logging.

        A zero count means no restraints were built for that type, which happens silently
        when a residue is absent from the monomer library.

        ``model.restraints.restraints`` is a ``_RestraintsAccessor`` rather than a dict --
        it has ``get``/``__getitem__``/``__contains__`` but no ``items()`` -- so this uses
        the nested-``get`` idiom from ``RestraintsNew.summary``.
        """
        r = model.restraints.restraints
        # "vdw" and "chiral" are torchref's _FLAT_TYPES: stored as restraints[t]["indices"]
        # with no origin level, unlike the nested restraints[t][origin]["indices"].
        flat = {"vdw", "chiral"}
        counts: dict[str, int] = {}
        for rtype in ("bond", "angle", "torsion", "plane", "chiral", "vdw"):
            total = 0
            try:
                group = r.get(rtype, {})
                if rtype in flat:
                    idx = group.get("indices")
                    total = 0 if idx is None else int(idx.shape[0])
                else:
                    for origin in group.keys() if hasattr(group, "keys") else ():
                        idx = group.get(origin, {}).get("indices")
                        if idx is not None:
                            total += int(idx.shape[0])
            except Exception:  # noqa: BLE001 - logging must not break the build
                total = -1
            counts[rtype] = total
        return counts

    # ------------------------------------------------------------------
    # Nuisance parameters
    # ------------------------------------------------------------------
    def refresh_nuisance_parameters(
        self, n_conformers: int = 1, coordinates: torch.Tensor | None = None
    ) -> dict[str, float]:
        """Refit the scale, bulk solvent and sigma_A, and report R-factors.

        Called automatically on the first call and every ``refresh_interval`` calls after
        it; exposed so a caller can drive the cadence itself.

        Coordinates are swapped for a ``.detach().clone()`` for the duration, giving the
        refit its own storage and version counter. A bare ``detach()`` would share the
        version counter, so an in-place write during the refit would fail autograd's
        version check on the caller's later ``backward()``. Cost is an ``(N, 3)`` copy.

        A warm failure is logged and swallowed, keeping the previous scale and sigma_A. A
        cold-start failure raises, since there is no previous state and every subsequent
        loss would be scored against an unfitted scale.

        Parameters
        ----------
        n_conformers
            Which cached stack to refresh.
        coordinates
            Coordinates to fit against, ``(C, n, 3)`` or ``(C*n, 3)``. Defaults to
            whatever is currently bound.

        Returns
        -------
        dict of str to float
            ``{"r_work": ..., "r_free": ...}``, or empty if a warm refit failed.

        Raises
        ------
        RuntimeError
            If the very first fit fails; see above.
        """
        self._require_prepared()
        model, scaler, target = self._stack_for(n_conformers)

        if coordinates is not None:
            model.set_coordinates(coordinates.reshape(-1, 3))
        live = model.xyz()

        cold = not hasattr(scaler, "log_scale")
        model.set_coordinates(live.detach().clone())
        try:
            if cold:
                # Cold start. Scaler.initialize is the canonical order --
                # calc_initial_scale -> setup_solvent -> setup_anisotropy_correction --
                # but it always sets up solvent, so do the pair by hand when it is off.
                if self.bulk_solvent:
                    scaler.initialize()
                else:
                    scaler.calc_initial_scale()
                    scaler.setup_anisotropy_correction()
            elif self.bulk_solvent:
                # Warm: rebuild the mask at the current coordinates.
                scaler.solvent.update_solvent()
                # forward() only rebuilds the raw solvent structure factors when this
                # cache is None, so without it the new mask never reaches F_calc.
                scaler._f_sol_raw = None

            scaler.refine_lbfgs(scale_target=self.scale_target)
            target.maintenance()  # drops the sigma_A cache; nothing else invalidates it

            if self.refine_adp:
                self._refine_adp(n_conformers)

            with torch.no_grad():
                r_work, r_free = target.get_rfactor()
            logger.info(
                f"Refreshed nuisance parameters (C={n_conformers}, "
                f"call {self._calls.get(n_conformers, 0)}): "
                f"R_work={r_work:.4f}, R_free={r_free:.4f}"
            )
            return {"r_work": float(r_work), "r_free": float(r_free)}
        except Exception as exc:  # noqa: BLE001 - a failed refit must not kill sampling
            if cold:
                # No previous state to fall back on, so every loss from here would be
                # scored against an unfitted scale.
                raise RuntimeError(
                    "The initial nuisance-parameter fit failed, so the reward has no "
                    "usable scale. Check that the MTZ and the structure describe the "
                    f"same crystal and that the free set is non-empty. Cause: {exc}"
                ) from exc
            logger.warning(
                f"Nuisance-parameter refresh failed ({type(exc).__name__}: {exc}); "
                "keeping the previous scale and sigma_A."
            )
            return {}
        finally:
            # Reattach even if the refit raised; a bound detached clone would return a
            # gradient-free loss for every subsequent call.
            model.set_coordinates(live)

    def _refine_adp(self, n_conformers: int) -> None:
        """LBFGS over the shared B-factors, coordinates frozen.

        Called from :meth:`refresh_nuisance_parameters`, where the caller's coordinates
        are already swapped for a detached clone.

        Refines ``("adp",)`` only, unlike torchref's ``refine_adp`` which also passes
        ``("u", "occupancy")``: every atom here is isotropic so ``u`` is a zero-element
        leaf, and occupancy is caller-owned. Coordinates need no explicit freeze, since
        ``LossState.run`` diffs its loss leaves against the optimizer's parameters and
        disables the rest for the duration.
        """
        model, _scaler, _target = self._stacks[n_conformers]
        state = self._states[n_conformers]

        params = model.parameters_of_types(("adp",))
        if not params:
            logger.warning("No refinable ADP leaf found; skipping the ADP refit.")
            return

        # The k-NN list behind the ADP locality restraint is cached and is NOT covered by
        # maintenance(), so it goes stale as the caller moves coordinates. The simu bond
        # pairs are topology-only and stay valid.
        locality = getattr(state, "targets", {}).get("adp/locality")
        if locality is not None:
            try:
                locality(recompute_neighbors=True)
            except TypeError:
                pass  # older signature without the kwarg; the stale list still works

        before = float(state.aggregate())
        b_before = model.adp().detach()
        optimizer = torch.optim.LBFGS(
            params, lr=1.0, max_iter=20, history_size=100, line_search_fn="strong_wolfe"
        )
        state.step(optimizer, context="torchref_reward.refine_adp")
        after = float(state.aggregate())

        b_now = model.adp().detach()
        logger.info(
            f"Refined {params[0].numel()} shared B-factors: loss {before:.4g} -> {after:.4g}, "
            f"B {b_before.min():.1f}-{b_before.max():.1f} -> "
            f"{b_now.min():.1f}-{b_now.max():.1f} A^2"
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _require_prepared(self) -> None:
        if not self._prepared:
            raise RuntimeError(
                "TorchRefXrayRewardFunction.prepare() must be called with the model atom "
                "array (or `structure=` passed to __init__) before the reward is evaluated."
            )

    def _validate(
        self,
        coordinates: torch.Tensor,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
        occupancies: torch.Tensor,
    ) -> None:
        """Check the incoming batch against the cached topology.

        Atom count is checked before element identity so the common case reports a clear
        message rather than a shape error. A mismatch in either is raised rather than
        adapted to: element identity and atom ordering are fixed for a given experiment.
        """
        for name, n in (
            ("coordinates", coordinates.shape[-2]),
            ("elements", elements.shape[-1]),
            ("b_factors", b_factors.shape[-1]),
            ("occupancies", occupancies.shape[-1]),
        ):
            if n != self.n_atoms:
                raise ValueError(
                    f"{name} has {n} atoms but prepare() cached a topology of "
                    f"{self.n_atoms}. Call prepare() with the same atom array the "
                    "sampled coordinates correspond to (model atom space)."
                )
        expected = self._expected_codes.expand_as(elements)
        if not torch.equal(elements, expected):
            raise ValueError(
                "elements do not match the topology cached by prepare(). Element identity "
                "and atom ordering are fixed for a given experiment, so this means the "
                "coordinate columns no longer correspond to the cached atoms."
            )

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Int[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
        occupancies: Float[torch.Tensor, "batch n_atoms"],
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Compute the X-ray likelihood for an ensemble of conformers.

        Call ``.backward()`` on the result for gradients with respect to ``coordinates``
        and ``occupancies``.

        The batch dimension is the ensemble: ``C`` conformers form a single
        ``C * n_atoms`` stack evaluated in one structure-factor calculation. The reshapes
        are already in conformer-major order, so no tiling is needed and both tensors stay
        in the autograd graph.

        Nuisance parameters are refreshed before the loss is computed, so the returned
        value is scored with a scale at most ``refresh_interval`` calls stale, and the
        cold-start fit happens before the reward first returns.

        Parameters
        ----------
        coordinates
            ``[batch, n_atoms, 3]`` Cartesian coordinates (Å) in the crystal frame. Not
            SE(3)-invariant: a rigid translation changes every phase, so these must arrive
            aligned to the MTZ's frame.
        elements
            ``[batch, n_atoms]`` scattering-table codes. Checked against the cached
            topology rather than used; see :meth:`_validate`.
        b_factors
            ``[batch, n_atoms]``. Ignored. ADPs come from the model's own slot, either
            held at ``b_factor`` or refined during maintenance when ``adp_weight > 0``.
        occupancies
            ``[batch, n_atoms]``. Used directly and differentiable. The ``1/batch_size``
            weighting makes the stacked structure factor the multi-conformer total.
        unique_combinations, inverse_indices
            Accepted for protocol compatibility and unused; torchref's kernels do not
            vmap.

        Returns
        -------
        torch.Tensor
            Scalar summed negative log-likelihood, unnormalised and not sign-constrained.

        Raises
        ------
        RuntimeError
            If :meth:`prepare` has not been called, or if the cold-start nuisance
            fit on the first call fails — see
            :meth:`refresh_nuisance_parameters`.
        ValueError
            On an atom-count or element mismatch against the cached topology, or if
            the inputs are on a different device than :meth:`prepare` was given.
        """
        self._require_prepared()
        if coordinates.ndim != 3:
            raise ValueError(
                f"coordinates must be [batch, n_atoms, 3]; got shape {tuple(coordinates.shape)}."
            )
        if coordinates.device != self.device:
            raise ValueError(
                f"coordinates are on {coordinates.device} but this reward was prepared on "
                f"{self.device}. Re-run prepare(device=...) to move it."
            )
        self._validate(coordinates, elements, b_factors, occupancies)

        n_conformers = coordinates.shape[0]
        model, _scaler, target = self._stack_for(n_conformers)

        live = coordinates.reshape(-1, 3)
        model.set_coordinates(live)
        model.set_occupancies(occupancies.reshape(-1))

        # Per-stack count: 0 means this triple has never been fitted, so a new ensemble
        # size cold-starts rather than inheriting another stack's refresh position.
        if self._calls.get(n_conformers, 0) % self.refresh_interval == 0:
            self.refresh_nuisance_parameters(n_conformers)
        self._calls[n_conformers] = self._calls.get(n_conformers, 0) + 1

        # No fcalc argument: torchref computes F_calc from the model, applies the scaler
        # and evaluates the likelihood.
        #
        # aggregate() is the weighted sum over registered groups; with both restraint
        # weights at 0 only "xray" is registered and this reduces to target().
        return self._states[n_conformers].aggregate()

    def __repr__(self) -> str:
        if not self._prepared:
            return f"TorchRefXrayRewardFunction({self.mtzfile!r}, unprepared)"
        return (
            f"TorchRefXrayRewardFunction({self.mtzfile!r}, n_atoms={self.n_atoms}, "
            f"mode={self.target_mode!r}, device={self.device}, calls={sum(self._calls.values())})"
        )
