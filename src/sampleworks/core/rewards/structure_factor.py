"""Reciprocal-space reward function for structure-factor amplitudes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import reciprocalspaceship as rs
import torch
from jaxtyping import Bool, Complex, Float, Int
from loguru import logger
from sampleworks.core.rewards.options import StructureFactorOptions
from sampleworks.core.rewards.registry import RewardBuildContext
from sampleworks.synthetic.synthetic_utils import atomarray_to_gemmi, resolve_mtz_column
from SFC_Torch import SFcalculator
from SFC_Torch.io import PDBParser


if TYPE_CHECKING:
    from sampleworks.core.rewards.protocol import RewardInputs


# Loss callable: maps (|Fcalc|, |Fobs|) over the masked reflections to a scalar.
AmplitudeLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Bulk-solvent treatment for the scored amplitude. "off": |Fprotein|. "combined":
# |Ftotal| with one mask from the combined density. "per_conformer": |Ftotal| with
# the mean of per-conformer masks (see the class docstring / _compute_ensemble_ftotal).
_BULK_SOLVENT_MODES = ("off", "combined", "per_conformer")

# SFcalculator init kwargs owned by this class (derived from other constructor
# arguments or injected in prepare()). They must not be overridden via
# sfcalculator_kwargs, which is an escape hatch for the remaining, unmanaged kwargs.
_RESERVED_SFC_KWARGS = frozenset(
    {"dmin", "mode", "anomalous", "mtzdata", "pdbmodel", "expcolumns", "device", "set_experiment"}
)

# Two thresholds for warnings reflection set too small after dropping outliers and R-free test
# set; see _build_reflection_mask.
# The min fraction is intended to catch an R-free convention mismatch (specify the wrong test
# set value for example). R-free test sets are conventionally 5-10% and outlier should be under
# 1%, so 75% is a safe threshold.
# The min absolute number is intended to catch a reflection set too small to guide the structure
# generation. We should have on the order of 1e4 reflections for 2A resolution macromolecular
# data, so 5e3 is a safe threshold.
_MIN_RETAINED_REFLECTION_FRACTION = 0.75
_MIN_RETAINED_REFLECTIONS = 5_000


def _resolve_expcolumns(expcolumns: list[str] | None, ds: rs.DataSet) -> tuple[str, str]:
    """Resolve the ``[amplitude, sigma]`` columns to read from ``ds``, logging the choice.

    Auto-detection requires the MTZ to hold exactly one amplitude column and one sigma
    column. A multi-set MTZ (e.g. ``Fprotein`` + ``Ftotal``) is ambiguous and forces the
    caller to pass ``expcolumns`` explicitly.

    Parameters
    ----------
    expcolumns
        Caller-provided ``[amplitude, sigma]`` to use verbatim, or ``None`` to auto-detect.
    ds
        The parsed MTZ whose amplitude and sigma columns are selected.

    Returns
    -------
    tuple of str
        The resolved ``(amplitude, sigma)`` pair, in that order.

    Raises
    ------
    ValueError
        If a provided ``expcolumns`` is not a length-2 ``[amplitude, sigma]`` pair, or names
        a column absent from the MTZ's amplitude/sigma columns; or, when auto-detecting, if
        the MTZ has zero or more than one amplitude (or sigma) column.
    """
    amplitude_dtype, sigma_dtype = rs.StructureFactorAmplitudeDtype(), rs.StandardDeviationDtype()
    if expcolumns is not None:
        # A bare string of length 2 could pass the length check and fail in not so obvious ways
        # ("FP" -> amplitude "F", sigma "P"), so reject it explicitly.
        if isinstance(expcolumns, str) or len(expcolumns) != 2:
            raise ValueError(f"expcolumns must be a [amplitude, sigma] pair; got {expcolumns!r}.")
        amplitude = resolve_mtz_column(ds, amplitude_dtype, column=expcolumns[0])
        sigma = resolve_mtz_column(ds, sigma_dtype, column=expcolumns[1])
        return amplitude, sigma

    amplitude = resolve_mtz_column(ds, amplitude_dtype)
    sigma = resolve_mtz_column(ds, sigma_dtype)
    logger.info(
        f"No expcolumns provided; auto-detected SFC columns: "
        f"amplitude='{amplitude}', sigma='{sigma}'."
    )
    return amplitude, sigma


class StructureFactorRewardFunction:
    def __init__(
        self,
        mtzfile: str | Path,
        *,
        expcolumns: list[str] | None = None,
        resolution: float | None = None,
        scattering_factor_mode: str = "xray",
        bulk_solvent: str = "off",
        loss: AmplitudeLoss | None = None,
        normalize_amplitude: bool = False,
        exclude_free_reflections: bool = False,
        batch_partition: int = 10,
        sfcalculator_kwargs: dict | None = None,
    ):
        """Reward for fitting structure-factor amplitudes via SFcalculator.

        Scores structures against experimental (or synthetic) structure-factor
        amplitudes ``|Fobs|`` using ``SFcalculator`` from ``SFC_Torch``. This is the
        reciprocal-space counterpart to :class:`RealSpaceRewardFunction` in
        ``real_space_density.py``.

        The reward compares the model amplitudes ``|Fcalc|`` against a target
        ``|Fobs|`` loaded from an MTZ. For an ensemble (batch dimension) of
        conformers indexed by ``b``, the per-conformer *protein* structure
        factors are combined by a *complex* sum in the reciprocal space
        (``|sum F| != sum |F|``): ``Fprotein(h) = sum_b F_b(h)``. The atomic
        occupancy is accounted for in the calculation of the per-conformer
        *protein* structure factor F_b(h). ``|Fcalc|`` is then ``|Fprotein|``
        (``bulk_solvent="off"``) or ``|Ftotal|`` if bulk solvent is folded in.

        ``bulk_solvent`` can take one of the following values:
            * ``"off"`` (default): ``|Fprotein|`` — no solvent.
            * ``"combined"``: ``|Ftotal|`` with one bulk-solvent mask from the combined
              protein density, ``mask(<rho>)`` — matches the altloc single-structure
              ``Ftotal`` that ``generate_synthetic_sf`` writes to the MTZ.
            * ``"per_conformer"``: ``|Ftotal|`` with the mean of per-conformer solvent
              mask ``<mask(rho)>`` — the ensemble-averaged bulk solvent. Each conformer
              contributes a solvent mask at 1/batch_size weight.

        ``"combined"`` and ``"per_conformer"`` differ only for a real ensemble
        (batch > 1); the mask operator is nonlinear, so ``mask(<rho>) != <mask(rho)>``.
        Both use the default, *unrefined* scales ``kiso=1``, ``kmask=0.35``, small
        ``uaniso``; refining them during sampling is left for a later revision.
        ``normalize_amplitude`` (``|F|`` vs ``|E|``) is orthogonal and composes with
        any ``bulk_solvent`` choice.

        The crystal metadata is taken from the MTZ, which is the only self-consistent
        source: ``SFcalculator``'s ``init_mtz`` overwrites the cell and space group of
        the structure by the MTZ's whenever they disagree. Naively overriding the cell
        and/or the space group of MTZ can be dangerous because the reflections' indexing
        could become wrong, so there are no arguments here to override them.

        Unlike the real-space reward function, ``SFcalculator`` needs the full topology
        (``PDBParser`` -> ``gemmi.Structure``: atom names/elements, unit cell, space group,
        ``resolution`` -> the HKL set) at construction. That information is only available
        with the model atom array, i.e. after ``process_structure_to_trajectory_input``
        runs inside ``sample()``. Therefore, construction is two-phase:
            * ``__init__`` stores only the up-front config (target MTZ, ``resolution``,
              ``scattering_factor_mode``, loss) and resolves the crystal metadata against
              the MTZ; it does *not* build ``SFcalculator``.
            * :meth:`prepare` builds ``SFcalculator`` from the model atom array, on the
              ``device`` given there. The caller (a step scaler) is responsible for
              invoking it before the first ``__call__``.

        Parameters
        ----------
        mtzfile
            Path to the MTZ holding the target amplitudes (and a sigma column).
            Loaded with ``set_experiment=True`` so ``sfc.Fo`` / HKL set / bins are
            populated and the calculation is aligned to the target reflections.
        expcolumns
            Column names ``[amplitude, sigma]`` in the MTZ. If ``None`` (default), the
            amplitude and sigma columns are auto-detected, which requires the MTZ to hold
            exactly one of each; a multi-set MTZ (e.g. ``Fprotein`` + ``Ftotal``) is
            ambiguous and raises, forcing an explicit ``[amplitude, sigma]``. If provided,
            a ``ValueError`` is raised when it is not a length-2 pair or names a column
            absent from the MTZ.
        resolution
            High-resolution limit (dmin) in Angstrom, or ``None`` (default) to use
            the MTZ's own resolution. The HKL set comes from the ``mtzfile``; when
            given, ``resolution`` further truncates it.
        scattering_factor_mode
            SFcalculator scattering mode: ``"xray"`` or ``"cryoem"``.
        bulk_solvent
            Bulk-solvent treatment, one of ``"off"`` (default; score ``|Fprotein|``),
            ``"combined"`` (``|Ftotal|`` with one mask from the combined density), or
            ``"per_conformer"`` (``|Ftotal|`` with the plain, unweighted mean of the
            per-conformer masks). The per-conformer mean is *not* occupancy-weighted,
            meaning each conformer contributes a solvent mask at 1/batch_size weight.
            For a single conformer, ``"combined"`` and ``"per_conformer"`` coincide.
            For multiple conformers, ``"per_conformer"`` should be a more faithful model.
        loss
            Callable ``(|Fcalc|, |Fobs|) -> scalar`` over masked reflections.
            Defaults to mean-squared error on amplitudes. Pass any callable
            (L1, R-factor-style, resolution-weighted, ...) to customize.
        normalize_amplitude
            If True, score normalized structure factors (E-values) instead of ``|F|``.
        exclude_free_reflections
            If True, drop the R-free test-set reflections from the loss (use only
            the working set). Default False: use all non-outlier reflections.
        batch_partition
            Ensemble chunk size forwarded to ``SFcalculator.calc_fprotein_batch`` as its
            ``PARTITION`` parameter. SFC's own default (20) can still lead to OOM.
            Must be a positive integer.
        sfcalculator_kwargs
            Extra keyword arguments forwarded verbatim to ``SFcalculator(...)`` in
            :meth:`prepare` (e.g. ``n_bins``, ``freeflag``, ``testset_value``). Reserved
            keys in this class (listed in :data:`_RESERVED_SFC_KWARGS`) cannot be overridden.
        """
        self.mtzfile = str(mtzfile)
        self.resolution = resolution
        self.scattering_factor_mode = scattering_factor_mode
        if bulk_solvent not in _BULK_SOLVENT_MODES:
            raise ValueError(
                f"bulk_solvent must be one of {_BULK_SOLVENT_MODES}, got {bulk_solvent!r}."
            )
        self.bulk_solvent = bulk_solvent
        self.exclude_free_reflections = exclude_free_reflections
        # bool will pass isinstance(..., int) check but not the type check
        if type(batch_partition) is not int or batch_partition <= 0:
            raise ValueError(
                f"batch_partition must be a positive integer, got {batch_partition!r}."
            )
        self.batch_partition = batch_partition
        self.loss: AmplitudeLoss = loss if loss is not None else torch.nn.MSELoss()
        self.normalize_amplitude = normalize_amplitude  # |F| vs resolution-bin normalized |E|

        # Resolve crystal metadata / column names against the MTZ.
        self._resolve_mtz_metadata(expcolumns)

        # All SFcalculator init kwargs are known except `pdbmodel` (needs the model
        # atom array), `mtzdata` (a copy of the parsed dataset), and `device` (resolved
        # at prepare() time), all injected in prepare().
        self._sfc_kwargs = dict(
            dmin=self.resolution,
            mode=self.scattering_factor_mode,
            anomalous=False,
            set_experiment=True,
            expcolumns=self.expcolumns,
        )
        if sfcalculator_kwargs:
            reserved = _RESERVED_SFC_KWARGS & sfcalculator_kwargs.keys()
            if reserved:
                raise ValueError(
                    f"sfcalculator_kwargs may not override reserved keys {sorted(reserved)}; "
                    "these are managed by this class (via other constructor arguments or "
                    "injected in prepare())."
                )
            self._sfc_kwargs.update(sfcalculator_kwargs)

        # Populated by prepare(); None until then.
        self.sfc: SFcalculator | None = None
        self._reflection_mask: torch.Tensor | None = None

    def _resolve_mtz_metadata(self, expcolumns: list[str] | None) -> None:
        """Set ``self.unit_cell`` / ``space_group`` / ``expcolumns`` from the MTZ.

        The MTZ is the only source: it must be consistent with its own reflections, and
        ``SFcalculator`` enforces that by overwriting the cell / space group of the
        structure handed to it with the MTZ's (``init_mtz``) before deriving symmetry
        operators or the reciprocal cell. The values resolved here are therefore what the
        calculation uses; they are stamped onto the gemmi structure in :meth:`prepare`
        only to keep it self-consistent. The dataset is parsed once here and retained on
        ``self._mtz_dataset`` so :meth:`prepare` can build ``SFcalculator`` from it
        directly instead of re-reading the file. ``expcolumns``, unlike the crystal
        metadata, is caller-supplied and validated against the MTZ's columns; see
        :func:`_resolve_expcolumns`.

        Parameters
        ----------
        expcolumns
            Caller-supplied ``[amplitude, sigma]`` column names to use verbatim, or
            ``None`` to auto-detect from the MTZ; forwarded to :func:`_resolve_expcolumns`.

        Raises
        ------
        ValueError
            If the MTZ carries no space group gemmi can recognize, or no unit cell (a
            missing ``CELL`` header leaves gemmi's placeholder, which would make every
            d-spacing meaningless). Also propagated from :func:`_resolve_expcolumns` for a
            malformed or unknown ``expcolumns``.
        """
        self._mtz_dataset = rs.read_mtz(self.mtzfile)
        if self._mtz_dataset.spacegroup is None:
            raise ValueError(
                f"{self.mtzfile} carries no space group gemmi can recognize (missing or "
                "unrecognized SYMINF header)."
            )
        # gemmi's own check on if cell_a is exactly 1A (the dummy value gemmi writes)
        if not self._mtz_dataset.cell.is_crystal():
            raise ValueError(
                f"{self.mtzfile} carries no unit cell (missing CELL header, leaving gemmi's "
                f"placeholder {self._mtz_dataset.cell.parameters}). Please re-generate the MTZ."
            )
        self.unit_cell = self._mtz_dataset.cell
        self.space_group = self._mtz_dataset.spacegroup.hm

        # Stored (and passed to SFcalculator) as a list: SFC_Torch annotates the kwarg
        # `List[str]` and swallows any failure into a misleading "columns not in the mtz" error.
        self.expcolumns = list(_resolve_expcolumns(expcolumns, self._mtz_dataset))

    def prepare(self, reward_inputs: RewardInputs, *, device: torch.device | str = "cpu") -> None:
        """Build the SFcalculator from the reward inputs, on ``device``.

        Constructing the ``SFcalculator`` consumes the MTZ dataset parsed at
        ``__init__`` (no second file read) and populates the observed structure
        factor amplitudes ``sfc.Fo``, the observed HKL set in the ASU, the resolution
        bins, the outlier mask ``sfc.Outlier``, the R-free flags ``sfc.free_flag``,
        and the normalized ``|Eo|`` in ``sfc.Eo``.

        Must be called once before the first ``__call__``, with the inputs built
        for the sampled coordinates (model atom space); they are the same inputs
        ``__call__`` is fed. The atom ordering of ``reward_inputs.atom_array``
        defines the column order of the coordinate tensor passed to ``__call__``.
        The gemmi structure comes from :meth:`RewardInputs.to_atom_array`: the
        model topology with the reconciled reference coordinates and B-factors,
        not the model template's placeholders. That matters here because
        ``inspect_data`` estimates the solvent fraction from the atom positions.

        This is also where the torch device is set, so pass the device the sampled
        coordinates will live on: every tensor built here (the SFcalculator internals and
        the reflection mask) is allocated on it, and SFcalculator bakes it in at
        construction with no way to migrate afterwards. Re-running ``prepare`` on a new
        device is therefore the way to move a built reward — safe to do, since the parsed
        MTZ dataset on ``self._mtz_dataset`` is kept pristine for exactly that reason.

        Per-atom B-factors / occupancy are set per ``__call__`` (not here), leaving
        the door open to refining them during sampling.

        Parameters
        ----------
        reward_inputs
            Inputs for the atoms the model operates on, as returned by
            ``SampleworksProcessedStructure.to_reward_inputs``. Their ``atom_array``
            needs ``chain_id``, ``res_id``, ``res_name``, ``atom_name``, ``element``
            annotations; the B-factors and occupancies are baked as defaults but
            overridden each ``__call__``. A missing ``altloc_id`` is defaulted to
            blank inside ``atomarray_to_gemmi``.
        device
            Torch device to build on. Defaults to CPU rather than auto-selecting a GPU:
            the same behavior as how the RewardInputs' device is resolved.

        Raises
        ------
        RuntimeError
            If ``normalize_amplitude`` is True but SFcalculator's normalization did not
            yield finite ``sfc.Eo`` from the supplied MTZ (absent or non-finite). When
            ``normalize_amplitude`` is False the same condition only logs a warning,
            since it also implies no reflection was flagged as an outlier.
        ValueError
            If ``reward_inputs`` carry no atom array to take the topology from, or
            if no reflection survives the mask; see :meth:`_build_reflection_mask`.
        """
        self.device = torch.device(device)

        atom_array = reward_inputs.to_atom_array()
        gemmi_structure = atomarray_to_gemmi(
            atom_array,
            unit_cell=self.unit_cell,
            space_group=self.space_group,
        )
        # SFcalculator mutates its mtzdata in place (dropna / hkl_to_asu on the reference),
        # so hand it a copy of the once-parsed dataset; self._mtz_dataset stays pristine and
        # prepare() remains safely re-runnable.
        sfc_kwargs = {
            "mtzdata": self._mtz_dataset.copy(),
            "device": self.device,
            **self._sfc_kwargs,
        }
        self.sfc = SFcalculator(pdbmodel=PDBParser(gemmi_structure), **sfc_kwargs)
        # inspect_data estimates solvent percentage and grid size from atom positions
        # and vdW radii, independent of occupancy / B-factor.
        self.sfc.inspect_data()

        # Ftotal uses the default (unrefined) scales kiso=1, kmask=0.35, small uaniso,
        # matching generate_synthetic_sf's Ftotal. They don't depend on coordinates, so
        # set them once here. _set_scales only needs atom_pos_frac (set at construction)
        # and n_bins (set by the experiment init) for dtype/device and bin count.
        if self.bulk_solvent != "off":
            self.sfc._set_scales(requires_grad=False)

        # |Eo|, epsilon, and Outlier all come from one bare try/except inside SFC's experiment
        # init. A missing Eo means that block raised; a non-finite Eo means a resolution bin
        # had zero mean intensity. Either way |Eo| is unusable and the sfc.Outlier mask would
        # not flag the outlier reflections based on Eo.
        Eo = getattr(self.sfc, "Eo", None)
        if Eo is None or not torch.isfinite(Eo).all():
            if self.normalize_amplitude:
                raise RuntimeError(
                    "normalize_amplitude=True requires finite sfc.Eo, but SFcalculator's "
                    "normalization block did not yield them from this MTZ."
                )
            logger.warning(
                "SFcalculator did not yield finite sfc.Eo from this MTZ; its outlier detection "
                "will only flag non-positive amplitudes as outliers instead of also using Eo."
            )

        # True where a reflection contributes to the reward (non-outlier, and not in the
        # R-free test set if exclude_free_reflections is True). Shape is [n_hkl].
        mask_np = self._build_reflection_mask(
            outlier=self.sfc.Outlier, free_flag=self.sfc.free_flag
        )
        self._reflection_mask = torch.from_numpy(mask_np).to(self.device)

        logger.info(
            f"Prepared StructureFactorRewardFunction: n_atoms={len(self.sfc.atom_pos_orth)}, "
            f"n_reflections={len(self.sfc.Fo)}, n_used={int(mask_np.sum())}, "
            f"cell={self.sfc.unit_cell}, space_group={self.sfc.space_group.hm}, "
            f"solventpct={self.sfc.solventpct}, gridsize={self.sfc.gridsize}"
        )

    def _build_reflection_mask(
        self,
        *,
        outlier: Bool[np.ndarray, "n_hkl"],  # noqa: F821, UP037
        free_flag: Bool[np.ndarray, "n_hkl"],  # noqa: F821, UP037
    ) -> Bool[np.ndarray, "n_hkl"]:  # noqa: F821, UP037
        """Build a mask of valid training set reflections for reward computation and
        sanity check the mask size.

        Outliers are always dropped; the R-free test set is also dropped when user sets
        ``exclude_free_reflections``. Raise error when the number of reflections remain
        is 0, and log warning when it is too small (below ``_MIN_RETAINED_REFLECTION_FRACTION``
        or ``_MIN_RETAINED_REFLECTIONS``).

        Parameters
        ----------
        outlier
            ``sfc.Outlier`` ``[n_hkl]``: True where SFcalculator flagged the observed
            reflection as an outlier.
        free_flag
            ``sfc.free_flag`` ``[n_hkl]``: True where the reflection is in the R-free
            test set.

        Returns
        -------
        numpy.ndarray
            Boolean mask over the observed reflections ``[n_hkl]``, True where the
            reflection contributes to the loss.

        Raises
        ------
        ValueError
            If no reflection survives the mask.
        """
        mask_np = ~outlier  # allocates, so the &= below never touches SFC's own arrays
        if self.exclude_free_reflections:
            mask_np &= ~free_flag

        n_total, n_used = mask_np.size, int(mask_np.sum())
        composition = (
            f"{n_total} observed reflections, "
            f"{int(outlier.sum())} outliers, "
            f"{int(free_flag.sum())} flagged free "
            f"(exclude_free_reflections={self.exclude_free_reflections})"
        )
        convention_hint = (
            "SFcalculator treats mtz[freeflag] == testset_value as the test set; "
            "pass a matching `freeflag` / `testset_value` via sfcalculator_kwargs "
            "if the MTZ's R-free convention differs from its defaults ('FreeR_flag' / 0)."
        )
        if n_used == 0:
            raise ValueError(f"No reflections remain: {composition}. {convention_hint}")
        if n_used < _MIN_RETAINED_REFLECTIONS:
            logger.warning(
                f"Only {n_used} reflections remain: {composition}. "
                f"Check `resolution` (dmin={self.resolution} A) and the MTZ reflection range."
            )
        if n_used < _MIN_RETAINED_REFLECTION_FRACTION * n_total:
            logger.warning(
                f"Only {n_used}/{n_total} reflections remain: {composition}. {convention_hint}"
            )
        return mask_np

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Int[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
        occupancies: Float[torch.Tensor, "batch n_atoms"],
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Compute the amplitude loss for the (ensemble of) coordinates.

        Call ``.backward()`` on the result to get gradients w.r.t.
        ``coordinates``.

        ``elements`` is ignored (topology is fixed in the SFcalculator built by
        :meth:`prepare`). ``b_factors`` and ``occupancies`` are reset onto the
        SFcalculator each call (mirroring ``RealSpaceRewardFunction``), and must be
        broadcast-identical across the batch dim (enforced; SFcalculator has no
        per-conformer occupancy/B axis, so non-broadcast input raises ``ValueError``).

        Parameters
        ----------
        coordinates
            Atomic coordinates ``[batch, n_atoms, 3]`` in model atom space,
            matching the atom ordering passed to :meth:`prepare`.
        elements
            Per-atom element codes ``[batch, n_atoms]``. Ignored: the topology
            (including elements) is fixed in the SFcalculator built by
            :meth:`prepare`. Present only to satisfy the reward-function signature.
        b_factors
            Per-atom isotropic B-factors ``[batch, n_atoms]``, written to
            ``sfc.atom_b_iso`` (reconciled: real deposited where shared with the
            structure, 20.0 for model-only / NaN atoms).
        occupancies
            Per-atom occupancies ``[batch, n_atoms]`` (uniform ``1/batch_size`` from
            the pipeline), written to ``sfc.atom_occ``; the ``1/batch_size`` weighting
            makes the complex ensemble sum the multi-conformer total.
        unique_combinations
            Pre-computed unique (element, b_factor) pairs for vmap compatibility.
            Currently unused.
        inverse_indices
            Pre-computed inverse indices for vmap compatibility. Currently unused.

        Returns
        -------
        torch.Tensor
            Scalar reward (loss).

        Raises
        ------
        RuntimeError
            If :meth:`prepare` has not been called.
        ValueError
            If any input's atom count differs from the SFcalculator topology built by `prepare`
            or if ``b_factors`` / ``occupancies`` are not broadcast-identical across batch dim.
        """
        if self.sfc is None or self._reflection_mask is None:
            raise RuntimeError(
                "StructureFactorRewardFunction.prepare() must be called with the reward "
                "inputs before the reward is evaluated."
            )

        # The topology (atom count included) is fixed by prepare(); a mismatch would otherwise
        # surface as a broadcast error inside calc_fprotein_batch.
        n_topology_atoms = len(self.sfc.atom_pos_orth)
        for name, n_atoms in (
            ("coordinates", coordinates.shape[-2]),
            ("b_factors", b_factors.shape[-1]),
            ("occupancies", occupancies.shape[-1]),
        ):
            if n_atoms != n_topology_atoms:
                raise ValueError(
                    f"{name} has {n_atoms} atoms but the SFcalculator topology built by "
                    f"prepare() has {n_topology_atoms}. Call prepare() with the reward inputs "
                    "built for the sampled coordinates (model atom space)."
                )

        # SFcalculator has no per-conformer (batch) occupancy/B axis, so these must be shared
        # across the ensemble; row 0 is used. Reject non-broadcast input as a guard.
        for name, tensor in (("occupancy", occupancies), ("B-factor", b_factors)):
            if not torch.equal(tensor, tensor[:1].expand_as(tensor)):
                raise ValueError(
                    f"StructureFactorRewardFunction requires {name} identical across the "
                    "batch dim (SFcalculator has no per-conformer occupancy/B axis); got "
                    "per-conformer values."
                )
        self.sfc.atom_b_iso = b_factors[0]
        self.sfc.atom_occ = occupancies[0]

        # Multi-conformer combination: complex sum over the ensemble [batch, n_hkl].
        # occ = 1/batch_size (set per call) makes the summed |F| the multi-conformer total.
        Fprotein_batch = self.sfc.calc_fprotein_batch(
            coordinates, Return=True, PARTITION=self.batch_partition
        )
        Fprotein = Fprotein_batch.sum(dim=0)

        fcalc = Fprotein if self.bulk_solvent == "off" else self._compute_ensemble_ftotal(Fprotein)

        mask = self._reflection_mask
        if self.normalize_amplitude:
            calc = self.sfc.calc_Ec(fcalc).abs()
            obs = self.sfc.Eo
        else:
            calc = torch.abs(fcalc)
            obs = self.sfc.Fo
        return self.loss(calc[mask], obs[mask])

    def _compute_ensemble_ftotal(
        self,
        Fprotein_HKL: Complex[torch.Tensor, "n_hkl"],  # noqa: F821, UP037
    ) -> Complex[torch.Tensor, "n_hkl"]:  # noqa: F821, UP037
        """Add default-scaled bulk solvent to the ensemble Fprotein to form Ftotal.

        ``Ftotal(h) = kiso * aniso(h) * (Fprotein(h) + kmask * Fmask(h))`` with the
        default (unrefined) scales set in :meth:`prepare`, evaluated on the
        experimental HKL set. ``Fprotein`` is the ensemble complex sum; the
        bulk-solvent ``Fmask`` is combined per :attr:`bulk_solvent`. The two
        ``|Ftotal|`` modes are identical for a single conformer.

        ``bulk_solvent="combined"`` (``mask(<rho>)``)
            One mask built from the combined protein density. ``calc_fsolvent``
            builds the mask by FFT over the full ASU set, so the combined
            ``Fprotein_asu`` (not just the HKL subset) is fed to it. Matches the
            altloc single-structure Ftotal in the synthetic MTZ.

        ``bulk_solvent="per_conformer"`` (``<mask(rho)>``)
            Average of the per-conformer bulk solvent masks. ``rsgrid2realmask``
            normalizes the protein density and cuts at a quantile, so each conformer
            contributes ``Fmask`` in a scale-invariant way. All conformers are assumed
            to contribute the solvent mask equally (1/batch_size weight).

            Two caveats from ``rsgrid2realmask``'s batch path (SFC_Torch 0.3.3):

            - It is NOT permutation-invariant over the ensemble. The quantile ``CUTOFF``
              for bulk solvent mask comes from ``real_grid_norm[0]`` (batch element 0)
              and is then used for every conformer.
            - Above 5M voxels per conformer the cutoff is a quantile of an unseeded
              ``torch.randperm`` subsample, making the mask (and its gradients)
              nondeterministic run-to-run.

        Parameters
        ----------
        Fprotein_HKL
            Ensemble complex-sum Fprotein on the experimental HKL set ``[n_hkl]``.

        Returns
        -------
        torch.Tensor
            Complex Ftotal on the experimental HKL set ``[n_hkl]``.
        """
        assert self.sfc is not None  # prepare() built it; __call__ guards before dispatching here
        self.sfc.Fprotein_HKL = Fprotein_HKL  # drives calc_ftotal on the HKL set
        if self.bulk_solvent == "per_conformer":
            # calc_fsolvent_batch masks each conformer (from Fprotein_asu_batch, set by
            # calc_fprotein_batch); the mean applies the 1/batch_size weight -> <mask(rho)>.
            Fmask_HKL_batch = self.sfc.calc_fsolvent_batch(
                Return=True, PARTITION=self.batch_partition
            )
            self.sfc.Fmask_HKL = Fmask_HKL_batch.mean(dim=0)
        else:
            # One mask from the combined density -> mask(<rho>). SFC calc_solvent() requires
            # having the Fprotein_asu set, but batch mode only sets Fprotein_asu_batch.
            self.sfc.Fprotein_asu = self.sfc.Fprotein_asu_batch.sum(dim=0)
            self.sfc.calc_fsolvent()  # sets Fmask_HKL
        return self.sfc.calc_ftotal()  # default scales set in prepare()


def build_structure_factor_reward(
    options: StructureFactorOptions, context: RewardBuildContext
) -> StructureFactorRewardFunction:
    """Build the structure-factor reward from its configured options.

    The returned reward is not yet usable: like every two-phase reward it binds to
    the model topology in :meth:`StructureFactorRewardFunction.prepare`, which the
    trajectory scaler calls once sampling knows the model atom array. ``context``
    is therefore unused -- neither the input structure nor the device is settled at
    build time.

    Parameters
    ----------
    options
        Structure-factor reward options; see
        :class:`~sampleworks.core.rewards.options.StructureFactorOptions`.
    context
        Run-level inputs. Unused here, kept for a uniform builder signature.

    Returns
    -------
    StructureFactorRewardFunction
        Reward scoring the model amplitudes against the MTZ target.

    Raises
    ------
    ValueError
        If no MTZ was given.
    """
    del context  # topology and device arrive in prepare()

    if options.mtzfile is None:
        raise ValueError(
            "The structure_factor reward needs a target MTZ. Pass --mtzfile, or set "
            "reward_options.mtzfile in a --reward-config file."
        )

    return StructureFactorRewardFunction(
        options.mtzfile,
        expcolumns=options.expcolumns,
        resolution=options.resolution,
        scattering_factor_mode=options.scattering_factor_mode,
        bulk_solvent=options.bulk_solvent,
        normalize_amplitude=options.normalize_amplitude,
        exclude_free_reflections=options.exclude_free_reflections,
        batch_partition=options.batch_partition,
        sfcalculator_kwargs=options.sfcalculator_kwargs,
    )
