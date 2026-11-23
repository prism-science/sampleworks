"""Combined Bragg + diffuse loss, scored through the lunus.sf engine.

One forward pass gives both observables: ``structure_factors_batch`` returns
``F`` per configuration, and ``mean_and_diffuse`` reduces that to ``<F>`` (Bragg)
and ``<|F|²> − |<F>|²`` (diffuse). Scoring them in one loss rather than two
avoids paying for the grid evaluation and FFT twice.

Each term keeps its own reflection list. Bragg data are indexed on the crystal
cell's integer reciprocal lattice. A diffuse map may be too, or may be sampled
between the Bragg peaks, in which case it is indexed on an integer supercell of
the crystal cell (index ``h`` on a cell with a doubled ``a`` axis is ``h/2`` of the
crystal cell, so its odd indices are the crystal cell's half-integer points).
Both are valid diffuse targets; today only the first is evaluated, since one FFT
grid on the crystal cell serves both terms and each gathers its own reflections
from it. A supercell-indexed target needs the model's transform on a supercell
grid without folding, a second setup for the diffuse term (#493). A supercell
*model* is a different matter and already handled by lunus, which folds its
density into the crystal cell before extracting F(hkl), exact at integer hkl.

The two terms are combined convexly, ``bragg_weight`` running from 1.0 (pure
Bragg) to 0.0 (pure diffuse). Each term is ``||s·calc − obs||² / ||obs||²`` with
``s`` the least-squares scale (``fit_scale``) and the denominator the
normalization (``normalize``); with both on, a term is ``1 − cos²(calc, obs)``
and lies in [0, 1]. This is the structure-factor loss's MSE divided by the
mean squared target. The division is what lets ``bragg_weight`` mix the terms:
a Bragg residual is an L2 on amplitudes and a diffuse residual an L2 on
intensities, so their raw magnitudes differ by orders of magnitude and an
unnormalized ``0.5`` would not be a half-and-half mixture. It also keeps the
total O(1) across a weight sweep, so the sweep does not double as a step-size
sweep.

A weight of exactly zero means the term is **not computed** and its target is
**not required** — not computed and multiplied by zero, which would let a
non-finite term poison the loss through ``0 * nan`` and would force a
pure-diffuse run to supply Bragg amplitudes it ignores.

Diffuse is scored on its **anisotropic component**: the radial part is dominated
by contributions a coordinate model does not describe (solvent, incoherent
background, absorption). Both sides go through the same transform, matching what
``lunus/sf/xtraj.py`` does when it correlates against experimental diffuse data.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast, NamedTuple, TYPE_CHECKING

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
from jaxtyping import Bool, Float, Int
from loguru import logger
from lunus.sf import mean_and_diffuse
from lunus.sf.symmetry_torch import supercell_factors

from sampleworks.core.forward_models.xray import lunus_sf
from sampleworks.core.losses.options import DiffuseBraggOptions
from sampleworks.core.losses.protocol import LossInputs
from sampleworks.core.losses.registry import LossBuildContext
from sampleworks.synthetic.synthetic_utils import resolve_mtz_column


if TYPE_CHECKING:
    from lunus.sf import IsotropicBackground


# Bulk-solvent treatment, named as in StructureFactorLossFunction. lunus builds
# its mask per configuration, so there is no "combined" mode here.
_BULK_SOLVENT_MODES = ("off", "per_conformer")
# Taper width for an explicit solvent cutoff, in the cutoff's own units (e/A^3; the
# mask thresholds blurred density), as in generate_synthetic_sf_lunus. The cutoff
# itself is calibrated by default; see prepare().
DEFAULT_SOLVENT_TAPER_WIDTH = 0.10
# Retained-reflection thresholds, as in StructureFactorLossFunction: the fraction
# catches an R-free convention mismatch, the count a reflection set too small to guide.
_MIN_RETAINED_REFLECTION_FRACTION = 0.75
_MIN_RETAINED_REFLECTIONS = 5_000


class BraggTarget(NamedTuple):
    """The measured Bragg amplitudes of one MTZ, row-aligned, with its R-free flags.

    Attributes
    ----------
    hkl
        int64 Miller indices ``[n_refl, 3]``, as the MTZ indexes them.
    f_obs
        float64 observed amplitudes ``[n_refl]``, all finite.
    free_mask
        True where the reflection is in the R-free test set, ``[n_refl]``. ``None``
        when the MTZ's flag column could not be resolved, which is distinct from a
        resolved column that flags nothing.
    outlier_mask
        True where the reflection should be left out of every score, working or
        free, ``[n_refl]``: those without a positive amplitude for now, simplified
        from SFC.
    unit_cell, space_group
        Crystal metadata of the MTZ.
    """

    hkl: Int[np.ndarray, "n_refl 3"]
    f_obs: Float[np.ndarray, " n_refl"]
    free_mask: Bool[np.ndarray, " n_refl"] | None
    outlier_mask: Bool[np.ndarray, " n_refl"]
    unit_cell: gemmi.UnitCell
    space_group: gemmi.SpaceGroup


def _within_resolution(
    cell: gemmi.UnitCell,
    hkl: Int[np.ndarray, "n_refl 3"],
    resolution: float,
    name: str,
) -> Bool[np.ndarray, " n_refl"]:
    """Mask of one target's reflections at or below a high-resolution cutoff.

    Parameters
    ----------
    cell
        Cell the d-spacings are computed in.
    hkl
        int64 Miller indices ``[n_refl, 3]``.
    resolution
        High-resolution cutoff in Å.
    name
        ``"Bragg"`` or ``"diffuse"``, for the error message.

    Returns
    -------
    Bool[np.ndarray, " n_refl"]
        True where ``d >= resolution``.

    Raises
    ------
    ValueError
        If no reflection survives the cutoff.
    """
    d_spacing = np.asarray(cell.calculate_d_array(hkl), dtype=np.float64)
    keep = d_spacing >= resolution
    if not keep.any():
        raise ValueError(
            f"No {name} reflections survive a {resolution} A cutoff; the "
            f"target reaches {d_spacing.min():.2f} A."
        )
    return keep


def load_bragg_target(
    mtz_path: str | Path,
    *,
    amplitude_column: str | None = None,
    free_flag_column: str | None = None,
    test_set_value: int = 1,
    resolution: float | None = None,
) -> BraggTarget:
    """Read the Bragg amplitudes and R-free flags a structure-factor model is scored on.

    Upon loading the Bragg mtz, keeps every reflection with a finite amplitude inside
    the cutoff that is not a systematic absence. Test set reflections and outliers are
    NOT dropped; both are returned as masks.
    Indices stay as the MTZ gives them, not mapped to canonical reciprocal ASU.

    Parameters
    ----------
    mtz_path
        MTZ holding the observed amplitudes.
    amplitude_column
        Amplitude column, resolved by :func:`resolve_mtz_column`: ``None`` takes the
        MTZ's sole amplitude column.
    free_flag_column
        R-free flag column; ``None`` takes the MTZ's sole integer column.
    test_set_value
        Flag value marking the test set: 1 in the Phenix convention the synthetic
        generators write, 0 in CCP4's.
    resolution
        High-resolution cutoff in Å, or ``None`` to keep every measured reflection.

    Returns
    -------
    BraggTarget
        The reflections in MTZ row order. ``free_mask`` is ``None`` when the MTZ has
        no integer column, several with no ``free_flag_column`` to choose between
        them, or no integer column of that name.

    Raises
    ------
    ValueError
        If the amplitude column cannot be resolved, or no reflection survives the
        resolution cutoff.
    """
    dataset = rs.read_mtz(str(mtz_path))
    amplitude_column = resolve_mtz_column(
        dataset, rs.StructureFactorAmplitudeDtype(), column=amplitude_column
    )
    f_obs = dataset[amplitude_column].to_numpy(dtype=np.float64)
    hkl = dataset.get_hkls().astype(np.int64)
    # A systematic absence is zero by symmetry whatever the model, so it measures
    # nothing about the structure; SFcalculator drops them from its targets too.
    keep = np.isfinite(f_obs) & ~rs.utils.is_absent(hkl, dataset.spacegroup)
    if resolution is not None:
        keep[keep] = _within_resolution(dataset.cell, hkl[keep], resolution, "Bragg")

    try:
        flag_column = resolve_mtz_column(dataset, rs.MTZIntDtype(), column=free_flag_column)
    except ValueError as unresolved:
        # Not an error here: only a caller that needs the test set can say so.
        logger.debug(f"No R-free flags read from {mtz_path}: {unresolved}")
        free_mask = None
    else:
        # MTZInt is a nullable integer dtype; as float, a missing flag is NaN and
        # compares unequal to every value, so an unflagged reflection is working set.
        flags = dataset[flag_column].to_numpy(dtype=np.float64, na_value=np.nan)
        free_mask = (flags == test_set_value)[keep]

    f_obs = f_obs[keep]
    outlier_mask = f_obs <= 0
    return BraggTarget(hkl[keep], f_obs, free_mask, outlier_mask, dataset.cell, dataset.spacegroup)


class DiffuseTarget(NamedTuple):
    """The measured diffuse intensities of one MTZ, row-aligned.

    Attributes
    ----------
    hkl
        int64 Miller indices ``[n_refl, 3]``, as the MTZ indexes them.
    intensity
        float64 diffuse intensities ``[n_refl]``, all finite.
    unit_cell, space_group
        Crystal metadata of the MTZ.
    """

    hkl: Int[np.ndarray, "n_refl 3"]
    intensity: Float[np.ndarray, " n_refl"]
    unit_cell: gemmi.UnitCell
    space_group: gemmi.SpaceGroup


def load_diffuse_target(
    mtz_path: str | Path,
    *,
    intensity_column: str | None = None,
    resolution: float | None = None,
) -> DiffuseTarget:
    """Read the diffuse intensities an ensemble is scored on.

    Keeps every reflection with a finite intensity inside the cutoff.
    Everything here is vectorized deliberately. A diffuse map from xtraj runs to
    millions of reflections -- 1.57M for the examples/xtraj cell at 1.8 A -- and
    per-reflection Python (dict keys, ``calculate_d`` per index) turns this into
    minutes.

    Parameters
    ----------
    mtz_path
        MTZ holding the diffuse intensities.
    intensity_column
        Intensity column, resolved by :func:`resolve_mtz_column`: ``None`` takes the
        MTZ's sole intensity column.
    resolution
        High-resolution cutoff in Å, or ``None`` to keep every measured reflection.

    Returns
    -------
    DiffuseTarget
        The reflections in MTZ row order.

    Raises
    ------
    ValueError
        If the intensity column cannot be resolved, or no reflection survives the
        resolution cutoff.
    """
    dataset = rs.read_mtz(str(mtz_path))
    intensity_column = resolve_mtz_column(dataset, rs.IntensityDtype(), column=intensity_column)
    intensity = dataset[intensity_column].to_numpy(dtype=np.float64)
    hkl = dataset.get_hkls().astype(np.int64)
    keep = np.isfinite(intensity)
    if resolution is not None:
        keep[keep] = _within_resolution(dataset.cell, hkl[keep], resolution, "diffuse")
    return DiffuseTarget(hkl[keep], intensity[keep], dataset.cell, dataset.spacegroup)


class DiffuseBraggLossFunction:
    """Score an ensemble against Bragg amplitudes and anisotropic diffuse data.

    Construction is two-phase, like ``StructureFactorLossFunction``: the
    targets and configuration are read in ``__init__``, but the scattering
    kernels, grid and symmetry operations need the model topology and so are
    built in :meth:`prepare` from the ``LossInputs``, which the caller must
    invoke before the first evaluation.

    Parameters
    ----------
    bragg_target
        MTZ holding target structure-factor amplitudes. Required unless
        ``bragg_weight`` is 0.
    diffuse_target
        MTZ holding target diffuse intensities — for example what
        ``lunus/sf/xtraj.py`` writes with ``diffuse=<name>.mtz``, whose column is
        named ``ID``. Required unless ``bragg_weight`` is 1.
    bragg_weight
        Convex mixing weight: 1.0 scores Bragg alone, 0.0 diffuse alone. May be a
        callable of the diffusion time ``t`` for an annealed mixture, in which
        case :attr:`current_time` must be set before each evaluation — see the
        note there, since no sampleworks caller does that yet.
    bragg_column, diffuse_column
        Column names in the respective MTZs, resolved by
        :func:`resolve_mtz_column`: ``None`` takes the sole column of the right
        MTZ dtype, and a given name is checked against those candidates rather
        than trusted.
    resolution
        High-resolution cutoff in Å applied to the reflection list, or ``None``
        to use everything the targets contain.
    normalize
        Divide each term (Bragg, diffuse) by the squared norm of its target,
        making both dimensionless and O(1). Turning this off makes ``bragg_weight``
        effectively meaningless; it exists to diagnose that.
    fit_scale
        Fit one scalar per term at every evaluation, the least-squares scale of
        calc onto obs, so each term scores shape only and is invariant to the
        overall scale of model and target. Needed for experimental data, where
        Bragg amplitudes and diffuse intensities arrive on unrelated scales. It
        discards the magnitude of disorder, which both observables carry, so turn
        it off when the targets are on the model's absolute scale, as synthetic
        ones are. The scale is detached; by the envelope theorem that gives the
        exact gradient of the minimized residual, since ``∂/∂s`` vanishes at the
        fitted ``s``.
    bulk_solvent
        ``"off"`` (default) scores ``|Fprotein|``; ``"per_conformer"`` adds a
        bulk-solvent contribution from a mask built on each configuration's own
        density, giving ``|Ftotal|``. Named as in ``StructureFactorLossFunction``;
        lunus has no single-mask ``"combined"`` mode.
    solvent_cutoff, solvent_taper_width
        Density below which a voxel is solvent, and the width of the mask's smooth
        transition. Both are in e/Å³ because the mask thresholds the blurred
        density itself, not a distance from atoms. ``None`` (default) calibrates
        the cutoff in :meth:`prepare` so the mask calls the same fraction of the
        cell solvent as gemmi's geometric mask on the reference configuration, with
        the taper set to half the cutoff, lunus's convention; the matched cutoff is
        not a constant across systems or B-factor conventions. An explicit cutoff
        uses ``solvent_taper_width``.
    exclude_free_reflections
        If True, leave the R-free test set out of the Bragg residual, so the Bragg
        term scores the working set only. The test set is still evaluated, and
        counts toward the resolution the density grid is sized for when
        ``resolution`` is ``None``. The diffuse list is not filtered: diffuse
        intensity is continuous in reciprocal space and correlated between
        neighbouring points, so removing the Bragg test-set hkl would not remove
        the test set's influence, and a cross-validation scheme for diffuse data is
        an open question. The option therefore protects R-free, nothing more.
    free_flag_column, test_set_value
        The Bragg MTZ's flag column (``None`` takes its sole integer column) and the
        flag value marking the test set. The default 1 is the Phenix convention the
        synthetic generators write; CCP4-convention files use 0.
    exclude_outliers
        If True (default), leave outlier reflections out of the Bragg residual.
        Like the test set they are still evaluated.

    Attributes
    ----------
    current_time
        Diffusion time for a callable ``bragg_weight``. ``LossFunctionProtocol``
        carries no ``t``, so nothing in the sampling loop sets this today;
        annealing the mixture needs a caller that does.

    Raises
    ------
    ValueError
        If ``bulk_solvent`` is not a known mode, ``bragg_weight`` is outside
        [0, 1], a weighted term has no target, a target cannot be read (see
        :func:`load_bragg_target`, :func:`load_diffuse_target`), the two targets'
        cells are incompatible, user chooses to exclude test set reflections but
        the Bragg target carries no R-free flags or no Bragg reflection left.
    NotImplementedError
        If the diffuse target is indexed on a supercell of the Bragg cell.
    RuntimeError
        If no target was read.
    """

    def __init__(
        self,
        bragg_target: str | Path | None = None,
        diffuse_target: str | Path | None = None,
        *,
        bragg_weight: float | Callable[[float], float] = 0.5,
        bragg_column: str | None = None,
        diffuse_column: str | None = None,
        resolution: float | None = None,
        normalize: bool = True,
        fit_scale: bool = True,
        bulk_solvent: str = "off",
        solvent_cutoff: float | None = None,
        solvent_taper_width: float = DEFAULT_SOLVENT_TAPER_WIDTH,
        exclude_free_reflections: bool = False,
        free_flag_column: str | None = None,
        test_set_value: int = 1,
        exclude_outliers: bool = True,
    ) -> None:
        # A constant weight is stored as a float so that NumPy scalars are not taken
        # for callables later.
        self.bragg_weight = bragg_weight if callable(bragg_weight) else float(bragg_weight)
        self.resolution = resolution
        self.normalize = normalize
        self.fit_scale = fit_scale
        self.exclude_free_reflections = exclude_free_reflections
        self.free_flag_column = free_flag_column
        self.test_set_value = test_set_value
        self.exclude_outliers = exclude_outliers

        if bulk_solvent not in _BULK_SOLVENT_MODES:
            raise ValueError(
                f"bulk_solvent must be one of {_BULK_SOLVENT_MODES}, got {bulk_solvent!r}."
            )
        self.bulk_solvent = bulk_solvent
        self.solvent_cutoff = solvent_cutoff
        # A given cutoff is used as is; None is calibrated in prepare(), which has
        # the grid and the reference configuration the calibration needs.
        self._solvent = None
        if bulk_solvent == "per_conformer" and solvent_cutoff is not None:
            # Inline: lunus.sf loads submodules on first access (PEP 562), so this
            # loads solvent_torch only when a mask is requested.
            from lunus.sf import SolventModel

            self._solvent = SolventModel(cutoff=solvent_cutoff, taper_width=solvent_taper_width)
        self.current_time: float | None = None

        # Validate the weight against the targets now, not on first evaluation.
        # A constant weight tells us exactly which targets are needed; a callable
        # could ask for either at any time, so both are required.
        if callable(bragg_weight):
            needs_bragg = needs_diffuse = True
        else:
            if not 0.0 <= float(bragg_weight) <= 1.0:
                raise ValueError(
                    f"bragg_weight must lie in [0, 1] to be a convex mixture; got {bragg_weight}."
                )
            needs_bragg = float(bragg_weight) > 0.0
            needs_diffuse = float(bragg_weight) < 1.0

        if needs_bragg and bragg_target is None:
            raise ValueError(
                "bragg_target is required when bragg_weight > 0 (or is callable). "
                "Pass bragg_weight=0.0 to score diffuse alone."
            )
        if needs_diffuse and diffuse_target is None:
            raise ValueError(
                "diffuse_target is required when bragg_weight < 1 (or is callable). "
                "Pass bragg_weight=1.0 to score Bragg alone."
            )

        self._scores_bragg = needs_bragg
        self._scores_diffuse = needs_diffuse

        # Each term's reflection list comes from its target, so the model is
        # evaluated exactly where there is data. lunus extracts F(hkl) by indexing
        # the FFT grid modulo its shape, so any integer hkl works and no ASU
        # convention has to be reconciled.
        self._bragg, self._diffuse = self._load_targets(
            bragg_target, diffuse_target, bragg_column, diffuse_column
        )
        # _load_targets has checked that two targets share a cell
        if self._bragg is not None:
            self.unit_cell, self.space_group = self._bragg.unit_cell, self._bragg.space_group
        elif self._diffuse is not None:
            self.unit_cell, self.space_group = self._diffuse.unit_cell, self._diffuse.space_group
        else:
            raise RuntimeError("No target was read; __init__ requires at least one.")
        # The whole Bragg target is evaluated, so the grid is sized for all of it;
        # this mask picks the reflections the Bragg residual scores.
        self._bragg_mask = None if self._bragg is None else self._scored_mask(self._bragg)
        # Both lists are integer hkl of one cell today, so one grid evaluation
        # serves both terms and each gathers its own reflections from it.
        self._hkl, self._bragg_index, self._diffuse_index = self._union_hkl()

        self.setup: lunus_sf.LunusSetup | None = None
        # lunus.sf.IsotropicBackground instance, built in prepare(); calling it
        # returns the anisotropic component of a per-reflection intensity.
        self._isotropic_background_subtractor: IsotropicBackground | None = None

    def _load_targets(
        self,
        bragg_path: str | Path | None,
        diffuse_path: str | Path | None,
        bragg_column: str | None,
        diffuse_column: str | None,
    ) -> tuple[BraggTarget | None, DiffuseTarget | None]:
        """Read each scored term's target and check they share one cell.

        Parameters
        ----------
        bragg_path, diffuse_path
            MTZ paths; a term that is not scored passes ``None`` and is skipped.
        bragg_column, diffuse_column
            Column names, or ``None`` to take the sole column of the right dtype.

        Returns
        -------
        tuple[BraggTarget | None, DiffuseTarget | None]
            Each term's target, ``None`` for a term that is not scored.

        Raises
        ------
        ValueError
            If the diffuse target's cell is not an integer supercell of the Bragg
            target's, or a resolution cutoff leaves a term with no reflections.
        NotImplementedError
            If the diffuse target is indexed on a supercell of the Bragg cell.
        """
        bragg = diffuse = None
        if self._scores_bragg and bragg_path is not None:
            bragg = load_bragg_target(
                bragg_path,
                amplitude_column=bragg_column,
                free_flag_column=self.free_flag_column,
                test_set_value=self.test_set_value,
                resolution=self.resolution,
            )
        if self._scores_diffuse and diffuse_path is not None:
            diffuse = load_diffuse_target(
                diffuse_path, intensity_column=diffuse_column, resolution=self.resolution
            )

        if bragg is not None and diffuse is not None:
            # Raises ValueError unless the diffuse cell is an integer supercell of
            # the Bragg cell (same angles, each edge an integer multiple).
            factors = supercell_factors(diffuse.unit_cell.parameters, bragg.unit_cell.parameters)
            if factors != (1, 1, 1):
                raise NotImplementedError(
                    f"The diffuse target is indexed on a {factors} supercell of the "
                    f"Bragg cell, i.e. at fractional hkl. Evaluating the model there "
                    "needs its own supercell grid, which this loss does not build "
                    "yet (#493); supply a diffuse target indexed on the Bragg cell."
                )

        logger.info(
            "Targets:"
            + (
                f" {len(bragg.hkl)} Bragg reflections, |F| "
                f"{bragg.f_obs.min():.1f}-{bragg.f_obs.max():.1f}"
                if bragg is not None
                else ""
            )
            + (
                f" {len(diffuse.hkl)} diffuse reflections, I "
                f"{diffuse.intensity.min():.3g}-{diffuse.intensity.max():.3g}"
                if diffuse is not None
                else ""
            )
        )
        return bragg, diffuse

    def _union_hkl(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """The reflections to evaluate once, and each term's gather indices into them.

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None, np.ndarray | None]
            hkl: int64 ``[n_union, 3]``, every reflection either target needs, each
            once, in lexicographic order.
            bragg_index, diffuse_index: int64 ``[n_term]``, positions of that term's
            reflections within ``hkl``, one per row of its target; ``None`` for a
            term that is not scored.
        """
        bragg_hkl = None if self._bragg is None else self._bragg.hkl
        diffuse_hkl = None if self._diffuse is None else self._diffuse.hkl
        lists = [h for h in (bragg_hkl, diffuse_hkl) if h is not None]
        stacked = np.concatenate(lists)
        hkl, inverse = np.unique(stacked, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)  # NumPy 2.0.x returns (n, 1) for axis=0
        n_bragg = 0 if bragg_hkl is None else len(bragg_hkl)
        bragg_index = inverse[:n_bragg] if bragg_hkl is not None else None
        diffuse_index = inverse[n_bragg:] if diffuse_hkl is not None else None
        return hkl, bragg_index, diffuse_index

    def _scored_mask(self, bragg: BraggTarget) -> Bool[np.ndarray, " n_refl"]:
        """Mask of the Bragg target's reflections that the Bragg residual scores.

        Outliers are left out with ``exclude_outliers`` and the R-free test set with
        ``exclude_free_reflections``. For the test set this mirrors
        ``StructureFactorLossFunction._build_reflection_mask``: warn if the retained
        fraction or count is suspiciously low, since a low fraction usually means
        ``test_set_value`` has the wrong convention.

        Parameters
        ----------
        bragg
            The Bragg target, from :func:`load_bragg_target`.

        Returns
        -------
        Bool[np.ndarray, " n_refl"]
            True where the reflection is scored.

        Raises
        ------
        ValueError
            If the test set is to be excluded and the target carries no R-free
            flags, or no reflection remains.
        """
        scored = np.ones(len(bragg.f_obs), dtype=bool)
        n_outliers = n_free = 0
        if self.exclude_outliers:
            n_outliers = int(bragg.outlier_mask.sum())
            scored &= ~bragg.outlier_mask
        if self.exclude_free_reflections:
            if bragg.free_mask is None:
                raise ValueError(
                    "exclude_free_reflections needs R-free flags, but none could be read "
                    f"from the Bragg target (free_flag_column={self.free_flag_column!r}). The "
                    "MTZ must hold exactly one integer column, or free_flag_column must name one."
                )
            n_free = int(bragg.free_mask.sum())
            scored &= ~bragg.free_mask

        n_total, n_scored = scored.size, int(scored.sum())
        composition = (
            f"{n_total} Bragg reflections, {n_outliers} excluded as outliers, "
            f"{n_free} excluded as R-free (test_set_value={self.test_set_value})"
        )
        convention_hint = (
            "Reflections with flag == test_set_value are the test set; pass test_set_value=0 "
            "for a CCP4-convention MTZ."
        )
        if n_scored == 0:
            raise ValueError(f"No reflections remain: {composition}. {convention_hint}")
        if self.exclude_free_reflections:
            if n_free == 0:
                logger.warning(
                    f"exclude_free_reflections excluded nothing: {composition}. {convention_hint}"
                )
            if n_scored < _MIN_RETAINED_REFLECTIONS:
                logger.warning(f"Only {n_scored} reflections remain: {composition}.")
            if n_scored < _MIN_RETAINED_REFLECTION_FRACTION * n_total:
                logger.warning(
                    f"Only {n_scored}/{n_total} reflections remain: {composition}. "
                    f"{convention_hint}"
                )
        return scored

    def prepare(self, loss_inputs: LossInputs, *, device: torch.device | str = "cpu") -> None:
        """Build the scattering setup and the anisotropic transform on ``device``.

        Must be called with the inputs built for the sampled coordinates — model
        atom space, so ``model_atom_array`` where the reconciler reports a
        mismatch — since their atom ordering fixes the columns of every
        coordinate tensor passed to :meth:`__call__`.

        The topology comes from :meth:`LossInputs.to_atom_array`, which carries
        the model atom identities together with the reconciled coordinates and
        B-factors ``__call__`` will see. The B-factors matter here and not only
        there: :func:`lunus_sf.build_setup` bakes them into the scattering
        kernels, which is why they cannot vary per configuration afterwards.

        Parameters
        ----------
        loss_inputs
            Inputs for the sampled coordinates, as returned by
            ``SampleworksProcessedStructure.to_loss_inputs``. Needs an
            ``atom_array`` to take the topology from.
        device
            Torch device to build on. Every tensor built here is allocated on it,
            so pass the device the sampled coordinates will live on.

        Raises
        ------
        ValueError
            If ``loss_inputs`` carries no atom array.
        """
        device = torch.device(device)
        if loss_inputs.atom_array is None:
            raise ValueError(
                "DiffuseBraggLossFunction.prepare() needs loss_inputs.atom_array "
                "for the topology required to build the scattering kernels."
            )
        # <|F|^2> - |<F>|^2 is identically zero for one configuration, so the
        # diffuse term then contributes a constant and no gradient.
        n_configs = loss_inputs.input_coords.shape[0] if loss_inputs.input_coords.ndim == 3 else 1
        if self._scores_diffuse and n_configs < 2:
            logger.warning(
                "The diffuse term is weighted but the ensemble has one configuration, "
                "whose diffuse intensity is identically zero: the term contributes no "
                "gradient. Raise the ensemble size, or set bragg_weight=1.0 to score "
                "Bragg alone."
            )
        self.setup = lunus_sf.build_setup(
            loss_inputs.to_atom_array(),
            self.unit_cell,
            self.space_group.hm,
            self.resolution or self._highest_resolution(),
            device=device,
        )
        self._hkl_t = torch.as_tensor(self._hkl, dtype=torch.long, device=device)
        self._bragg_index_t = (
            None if self._bragg_index is None else torch.as_tensor(self._bragg_index, device=device)
        )
        self._diffuse_index_t = (
            None
            if self._diffuse_index is None
            else torch.as_tensor(self._diffuse_index, device=device)
        )

        # Recalibrated on every prepare(): the matched cutoff belongs to the topology
        # and grid prepared here, and prepare() may be re-run for different inputs.
        if self.bulk_solvent == "per_conformer" and self.solvent_cutoff is None:
            reference = loss_inputs.to_atom_array()
            target = lunus_sf.geometric_solvent_fraction(
                reference, reference.coord, self.unit_cell, self.space_group, self.setup.grid_shape
            )
            self._solvent = lunus_sf.calibrate_solvent(
                self.setup,
                torch.as_tensor(reference.coord, dtype=torch.float32, device=device),
                torch.ones(reference.array_length(), device=device),
                target,
            )

        if self._bragg is not None and self._bragg_mask is not None:
            self._bragg_fobs_t = torch.as_tensor(
                self._bragg.f_obs, dtype=torch.float32, device=device
            )
            self._bragg_mask_t = torch.as_tensor(self._bragg_mask, device=device)

        if self._diffuse is not None:
            # Inline for the same reason as SolventModel above: loads aniso_torch
            # only when the diffuse term is scored.
            from lunus.sf import IsotropicBackground

            # Shell thickness is d* of (1,1,1), the convention to_aniso uses.
            shell_thickness = float(np.sqrt(self.unit_cell.calculate_1_d2((1, 1, 1))))
            d_star = np.sqrt(self.unit_cell.calculate_1_d2_array(self._diffuse.hkl))
            isotropic_background = IsotropicBackground(
                d_star, shell_thickness, device=device, dtype=torch.float32
            )
            # The spline basis is (n_refl, n_bins) and is built in float64 before
            # being cast, so peak setup memory is roughly three times what it
            # finally occupies. A diffuse map of a few million reflections puts
            # that in the hundreds of MB -- worth reporting rather than
            # discovering as an allocation failure.
            basis_mb = len(self._diffuse.hkl) * isotropic_background.n_bins * 4 / 1e6
            logger.info(
                f"Anisotropic transform: {isotropic_background.n_bins} shells of "
                f"{shell_thickness:.5f} 1/A, basis {basis_mb:.0f} MB on {device}"
            )
            # The target is constant, so transform it once. Done in torch rather
            # than through the numpy one-shot so both sides pass through exactly
            # the same operator, leaving no room for the two to disagree.
            target = torch.as_tensor(self._diffuse.intensity, dtype=torch.float32, device=device)
            self._diffuse_iobs_aniso_t = isotropic_background(target)
            self._isotropic_background_subtractor = isotropic_background

        logger.info(
            f"Prepared DiffuseBraggLossFunction: {len(self._hkl)} reflections evaluated, "
            f"bragg_weight={self.bragg_weight}, normalize={self.normalize}, "
            f"fit_scale={self.fit_scale}"
        )

    def _highest_resolution(self) -> float:
        """d_min in Å over every evaluated reflection, for sizing the density grid."""
        return float(np.asarray(self.unit_cell.calculate_d_array(self._hkl)).min())

    def _weight(self) -> float:
        """The Bragg weight for this evaluation, in [0, 1].

        Raises
        ------
        RuntimeError
            If ``bragg_weight`` is callable and :attr:`current_time` is unset.
        ValueError
            If the callable returns a weight outside [0, 1].
        """
        if not callable(self.bragg_weight):
            return float(self.bragg_weight)
        # callable() narrows to an unknown signature for ty; __init__ only accepts floats
        # or a callable of t, so this is the latter.
        weight_of_time = cast(Callable[[float], float], self.bragg_weight)
        if self.current_time is None:
            raise RuntimeError(
                "bragg_weight is a callable of the diffusion time, but current_time "
                "was never set. LossFunctionProtocol carries no t, so a caller "
                "must set loss.current_time before each evaluation."
            )
        weight = float(weight_of_time(self.current_time))
        if not 0.0 <= weight <= 1.0:
            raise ValueError(
                f"bragg_weight({self.current_time}) returned {weight}, outside [0, 1]."
            )
        return weight

    def _residual(
        self,
        calc: Float[torch.Tensor, " n_refl"],
        obs: Float[torch.Tensor, " n_refl"],
    ) -> Float[torch.Tensor, ""]:
        """Scaled, optionally normalized squared residual between calc and obs.

        Parameters
        ----------
        calc, obs
            Calculated and observed values on one term's reflection list: Bragg
            amplitudes, or the signed anisotropic component of diffuse intensities.

        Returns
        -------
        Float[torch.Tensor, ""]
            ``||s·calc − obs||²``, divided by ``||obs||²`` when ``normalize`` is on.
        """
        if self.fit_scale:
            # Least squares in the scale alone, floored at zero: the anisotropic
            # diffuse component is signed, and an unconstrained fit would flip an
            # anticorrelated calc onto the target and score it as a match. Detaching
            # is exact: at the fitted scale d(residual)/d(scale) = 0, so the gradient
            # through calc is the full gradient of the minimized residual (envelope
            # theorem).
            scale = torch.dot(calc, obs) / torch.dot(calc, calc).clamp_min(1e-30)
            calc = scale.clamp_min(0.0).detach() * calc
        residual = torch.sum((calc - obs) ** 2)
        if self.normalize:
            residual = residual / torch.sum(obs**2).clamp_min(1e-30)
        return residual

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Int[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
        occupancies: Float[torch.Tensor, "batch n_atoms"],
    ) -> Float[torch.Tensor, ""]:
        """Score one ensemble.

        ``elements`` and ``b_factors`` are ignored: both are baked into the
        kernels built by :meth:`prepare`, which is also why B-factors cannot vary
        per configuration here.

        ``occupancies`` arrive as ``1/N`` from ``LossInputs`` because the sum-based
        losses add ``F`` over configurations and need the ``1/N`` to get the
        ensemble mean. This loss takes the mean and variance itself in
        ``mean_and_diffuse``, so it needs each configuration's ``F`` at full
        occupancy and multiplies the ``1/N`` back out. Left in place it would scale
        both moments by ``1/N²`` and shrink the diffuse term toward zero, which
        looks like guidance doing nothing rather than like an error.

        TODO(#483): configuration weights are uniform. Once ``lunus.sf.mean_and_diffuse``
        takes ``weights``, pass the occupancies through as given and the weights
        explicitly, so non-uniform populations can be scored.
        """
        if self.setup is None:
            raise RuntimeError(
                "DiffuseBraggLossFunction.prepare() must be called with the model "
                "atom array before the loss is evaluated."
            )

        n_configs = coordinates.shape[0]
        occupancies = occupancies * n_configs

        # Before the grid evaluation and FFT, not after: _weight() raises for an unset
        # current_time or a callable that returns a weight outside [0, 1], and
        # there is no reason to pay for a forward pass first.
        weight = self._weight()

        f_configs = lunus_sf.structure_factors(
            self.setup, coordinates, occupancies, self._hkl_t, solvent=self._solvent
        )
        # TODO(#493): one setup and one F serve both terms. A diffuse target at
        # fractional hkl, or a supercell model, needs a setup per term and a
        # separate mean / diffuse call; the weights those calls take are #483.
        mean_f, diffuse = mean_and_diffuse(f_configs)

        total = torch.zeros((), dtype=torch.float32, device=coordinates.device)

        # Zero-weight terms are skipped rather than multiplied by zero: 0 * nan
        # is nan, and a term that cannot contribute should not be able to poison
        # the loss.
        if weight > 0.0:
            total = total + weight * self._residual(
                torch.abs(mean_f[self._bragg_index_t])[self._bragg_mask_t],
                self._bragg_fobs_t[self._bragg_mask_t],
            )
        if weight < 1.0:
            if self._isotropic_background_subtractor is None:
                raise RuntimeError("prepare() built no anisotropic transform for the diffuse term.")
            total = total + (1.0 - weight) * self._residual(
                self._isotropic_background_subtractor(diffuse[self._diffuse_index_t]),
                self._diffuse_iobs_aniso_t,
            )

        return total


def build_diffuse_bragg_loss(
    options: DiffuseBraggOptions, context: LossBuildContext
) -> DiffuseBraggLossFunction:
    """Build the combined Bragg + diffuse loss from its configured options.

    Like the structure-factor loss this one binds to the model topology in
    :meth:`DiffuseBraggLossFunction.prepare`, so ``context`` is unused.

    Parameters
    ----------
    options
        Loss options; see :class:`~sampleworks.core.losses.options.DiffuseBraggOptions`.
    context
        Run-level inputs. Unused here, kept for a uniform builder signature.

    Returns
    -------
    DiffuseBraggLossFunction
        Loss scoring the ensemble against the MTZ targets.

    Raises
    ------
    ValueError
        If ``bragg_weight`` is outside [0, 1], or a weighted term has no target.
    """
    del context  # topology and device arrive in prepare()

    if not 0.0 <= options.bragg_weight <= 1.0:
        raise ValueError(
            f"--bragg-weight must lie in [0, 1] to be a convex mixture; got {options.bragg_weight}."
        )
    if options.bragg_weight > 0.0 and options.bragg_target is None:
        raise ValueError(
            "The diffuse_bragg loss needs a Bragg target unless --bragg-weight is 0. "
            "Pass --bragg-target, or set loss_options.bragg_target in a --loss-config file."
        )
    if options.bragg_weight < 1.0 and options.diffuse_target is None:
        raise ValueError(
            "The diffuse_bragg loss needs a diffuse target unless --bragg-weight is 1. "
            "Pass --diffuse-target, or set loss_options.diffuse_target in a --loss-config file."
        )

    return DiffuseBraggLossFunction(
        bragg_target=options.bragg_target,
        diffuse_target=options.diffuse_target,
        bragg_weight=options.bragg_weight,
        bragg_column=options.bragg_column,
        diffuse_column=options.diffuse_column,
        resolution=options.resolution,
        normalize=options.normalize,
        fit_scale=options.fit_scale,
        bulk_solvent=options.bulk_solvent,
        solvent_cutoff=options.solvent_cutoff,
        solvent_taper_width=options.solvent_taper_width,
        exclude_free_reflections=options.exclude_free_reflections,
        free_flag_column=options.free_flag_column,
        test_set_value=options.test_set_value,
        exclude_outliers=options.exclude_outliers,
    )
