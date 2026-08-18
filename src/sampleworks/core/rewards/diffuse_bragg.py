"""Combined Bragg + diffuse reward, scored through the lunus.sf engine.

One forward pass gives both observables: ``structure_factors_batch`` returns
``F`` per configuration, and ``mean_and_diffuse`` reduces that to ``<F>`` (Bragg)
and ``<|F|²> − |<F>|²`` (diffuse). Scoring them in one reward rather than two
avoids paying for the splat and FFT twice.

The two terms are combined convexly, ``bragg_weight`` running from 1.0 (pure
Bragg) to 0.0 (pure diffuse). For that dial to mean anything, each term is
normalized first: a Bragg residual is an L2 on amplitudes and a diffuse residual
an L2 on intensities, so their raw magnitudes differ by orders of magnitude and
an unnormalized ``0.5`` would not be a half-and-half mixture. Normalizing also
keeps the total O(1) across a weight sweep, so the sweep does not silently
double as a step-size sweep.

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
from typing import TYPE_CHECKING

import numpy as np
import reciprocalspaceship as rs
import torch
from jaxtyping import Float, Int
from loguru import logger
from sampleworks.core.forward_models.xray import lunus_sf


if TYPE_CHECKING:
    from biotite.structure import AtomArray


class DiffuseBraggRewardFunction:
    """Score an ensemble against Bragg amplitudes and anisotropic diffuse data.

    Construction is two-phase, like ``StructureFactorRewardFunction``: the
    targets and configuration are read in ``__init__``, but the scattering
    kernels, grid and symmetry operations need the model atom array and so are
    built in :meth:`prepare`, which the caller must invoke before the first
    evaluation.

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
        Column names in the respective MTZs. ``None`` auto-detects: the single
        structure-factor-amplitude column for Bragg, and for diffuse the single
        intensity column.
    resolution
        High-resolution cutoff in Å applied to the reflection list, or ``None``
        to use everything the targets contain.
    normalize
        Divide each term by the squared norm of its target, making both
        dimensionless and O(1). Turning this off makes ``bragg_weight``
        effectively meaningless; it exists to diagnose that.
    fit_scale
        Fit a least-squares scale factor per term per call, and **detach** it.
        The model and target need not share an overall scale, and detaching
        means the fit acts as a normalizer rather than as a parameter the
        gradient can exploit by shrinking the structure.
    isotropic_background
        Factory for the anisotropic transform, called as
        ``factory(d_star, shell_thickness, device=..., dtype=...)``. Defaults to
        ``lunus.sf.IsotropicBackground``; injectable so the reward can be tested
        against a reference implementation.

    Attributes
    ----------
    current_time
        Diffusion time for a callable ``bragg_weight``. ``RewardFunctionProtocol``
        carries no ``t``, so nothing in the sampling loop sets this today;
        annealing the mixture needs a caller that does.
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
        isotropic_background: Callable | None = None,
    ) -> None:
        self.bragg_weight = bragg_weight
        self.resolution = resolution
        self.normalize = normalize
        self.fit_scale = fit_scale
        self.current_time: float | None = None

        if isotropic_background is None:
            from lunus.sf import IsotropicBackground

            isotropic_background = IsotropicBackground
        self._isotropic_background = isotropic_background

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

        # The reflection list comes from the targets, so the model is evaluated
        # exactly where there is data. lunus extracts F(hkl) by indexing the FFT
        # grid modulo its shape, so any integer hkl works and no ASU convention
        # has to be reconciled.
        self._hkl, self._bragg_obs, self._diffuse_obs, self.unit_cell, self.space_group = (
            self._load_targets(bragg_target, diffuse_target, bragg_column, diffuse_column)
        )

        self.setup: lunus_sf.LunusSetup | None = None
        self._aniso = None

    def _load_targets(self, bragg_path, diffuse_path, bragg_column, diffuse_column):
        """Read the targets and reduce them to one shared reflection list.

        Everything here is vectorized deliberately. A diffuse map from xtraj runs
        to millions of reflections -- 1.57M for the examples/xtraj cell at 1.8 A
        -- and per-reflection Python (dict keys, ``calculate_d`` per index) turns
        this into minutes.
        """
        bragg_hkl = bragg_values = None
        diffuse_hkl = diffuse_values = None
        cell = spacegroup = None

        if self._scores_bragg:
            ds = rs.read_mtz(str(bragg_path))
            column = bragg_column or self._sole_column(
                ds, rs.StructureFactorAmplitudeDtype(), bragg_path
            )
            bragg_hkl, bragg_values = self._finite_column(ds, column)
            cell, spacegroup = ds.cell, ds.spacegroup

        if self._scores_diffuse:
            ds = rs.read_mtz(str(diffuse_path))
            column = diffuse_column or self._sole_column(ds, rs.IntensityDtype(), diffuse_path)
            diffuse_hkl, diffuse_values = self._finite_column(ds, column)
            if cell is None:
                cell, spacegroup = ds.cell, ds.spacegroup

        if bragg_hkl is None:
            hkl, bragg, diffuse = diffuse_hkl, None, diffuse_values
        elif diffuse_hkl is None:
            hkl, bragg, diffuse = bragg_hkl, bragg_values, None
        else:
            shared, i_bragg, i_diffuse = np.intersect1d(
                self._pack(bragg_hkl), self._pack(diffuse_hkl), return_indices=True
            )
            if shared.size == 0:
                raise ValueError(
                    "The Bragg and diffuse targets share no reflections. Check that "
                    "they were generated on the same cell and resolution range."
                )
            hkl = bragg_hkl[i_bragg]
            bragg = bragg_values[i_bragg]
            diffuse = diffuse_values[i_diffuse]

        if self.resolution is not None:
            d_spacing = cell.calculate_d_array(hkl)
            keep = d_spacing >= self.resolution
            if not keep.any():
                raise ValueError(
                    f"No reflections survive a {self.resolution} A cutoff; the "
                    f"targets reach {d_spacing.min():.2f} A."
                )
            hkl = hkl[keep]
            bragg = bragg[keep] if bragg is not None else None
            diffuse = diffuse[keep] if diffuse is not None else None

        logger.info(
            f"Targets: {len(hkl)} reflections"
            + (f", Bragg |F| {bragg.min():.1f}-{bragg.max():.1f}" if bragg is not None else "")
            + (f", diffuse {diffuse.min():.3g}-{diffuse.max():.3g}" if diffuse is not None else "")
        )
        return hkl, bragg, diffuse, cell, spacegroup

    @staticmethod
    def _pack(hkl: np.ndarray) -> np.ndarray:
        """View (n, 3) integer Miller indices as one comparable key per row.

        A structured view rather than arithmetic packing, so it cannot overflow
        or collide however large the indices get.
        """
        contiguous = np.ascontiguousarray(hkl, dtype=np.int64)
        return contiguous.view([("h", "i8"), ("k", "i8"), ("l", "i8")]).ravel()

    @staticmethod
    def _sole_column(dataset: rs.DataSet, dtype, path) -> str:
        """The one column of the given MTZ dtype, or an error naming the choices."""
        candidates = [c for c in dataset.columns if isinstance(dataset.dtypes[c], type(dtype))]
        if len(candidates) != 1:
            raise ValueError(
                f"{path} holds {len(candidates)} columns of type {type(dtype).__name__} "
                f"({candidates}); name one explicitly."
            )
        return candidates[0]

    @staticmethod
    def _finite_column(dataset: rs.DataSet, column: str) -> tuple[np.ndarray, np.ndarray]:
        """The (hkl, value) pairs of one column, dropping unmeasured entries."""
        values = dataset[column].to_numpy(dtype=np.float64)
        hkl = np.column_stack(
            [dataset.index.get_level_values(i).to_numpy() for i in ("H", "K", "L")]
        )
        finite = np.isfinite(values)
        return hkl[finite].astype(np.int64), values[finite]

    def prepare(self, atom_array: AtomArray, *, device: torch.device | str = "cpu") -> None:
        """Build the scattering setup and the anisotropic transform on ``device``.

        Must be called with the atom array the sampled coordinates correspond to
        — model atom space, so ``model_atom_array`` where the reconciler reports
        a mismatch — since its ordering fixes the columns of every coordinate
        tensor passed to :meth:`__call__`.
        """
        device = torch.device(device)
        self.setup = lunus_sf.build_setup(
            atom_array,
            self.unit_cell,
            self.space_group.hm,
            self.resolution or self._highest_resolution(),
            device=device,
        )
        self._hkl_t = torch.as_tensor(self._hkl, dtype=torch.long, device=device)

        if self._scores_bragg:
            self._bragg_t = torch.as_tensor(self._bragg_obs, dtype=torch.float32, device=device)

        if self._scores_diffuse:
            # Shell thickness is d* of (1,1,1), the convention to_aniso uses.
            shell_thickness = float(np.sqrt(self.unit_cell.calculate_1_d2((1, 1, 1))))
            d_star = np.sqrt(self.unit_cell.calculate_1_d2_array(self._hkl))
            self._aniso = self._isotropic_background(
                d_star, shell_thickness, device=device, dtype=torch.float32
            )
            # The spline basis is (n_refl, n_bins) and is built in float64 before
            # being cast, so peak setup memory is roughly three times what it
            # finally occupies. A diffuse map of a few million reflections puts
            # that in the hundreds of MB -- worth reporting rather than
            # discovering as an allocation failure.
            basis_mb = len(self._hkl) * self._aniso.n_bins * 4 / 1e6
            logger.info(
                f"Anisotropic transform: {self._aniso.n_bins} shells of "
                f"{shell_thickness:.5f} 1/A, basis {basis_mb:.0f} MB on {device}"
            )
            # The target is constant, so transform it once. Done in torch rather
            # than through the numpy one-shot so both sides pass through exactly
            # the same operator, leaving no room for the two to disagree.
            target = torch.as_tensor(self._diffuse_obs, dtype=torch.float32, device=device)
            self._diffuse_t = self._aniso(target)

        logger.info(
            f"Prepared DiffuseBraggRewardFunction: {len(self._hkl)} reflections, "
            f"bragg_weight={self.bragg_weight}, normalize={self.normalize}, "
            f"fit_scale={self.fit_scale}"
        )

    def _highest_resolution(self) -> float:
        """d_min of the target reflection list, for sizing the density grid."""
        return float(self.unit_cell.calculate_d_array(self._hkl).min())

    def _weight(self) -> float:
        """The Bragg weight for this evaluation."""
        if not callable(self.bragg_weight):
            return float(self.bragg_weight)
        if self.current_time is None:
            raise RuntimeError(
                "bragg_weight is a callable of the diffusion time, but current_time "
                "was never set. RewardFunctionProtocol carries no t, so a caller "
                "must set reward.current_time before each evaluation."
            )
        weight = float(self.bragg_weight(self.current_time))
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
        """Scaled, optionally normalized squared residual between calc and obs."""
        if self.fit_scale:
            # Least squares in the scale alone, detached: it removes an overall
            # factor the model has no reason to reproduce, without giving the
            # gradient a way to lower the loss by rescaling the structure.
            scale = (torch.dot(calc, obs) / torch.dot(calc, calc).clamp_min(1e-30)).detach()
            calc = scale * calc
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
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Score one ensemble. Call ``.backward()`` for gradients w.r.t. coordinates.

        ``elements`` and ``b_factors`` are ignored: both are baked into the
        kernels built by :meth:`prepare`, which is also why B-factors cannot vary
        per configuration here.

        ``occupancies`` arrive as ``1/N`` from ``RewardInputs``, a convention the
        sum-based rewards rely on. Diffuse is a *variance*, so each configuration
        must be splatted at its own occupancy and the moments taken afterwards —
        leaving the ``1/N`` in place would scale both moments by ``1/N²`` and
        collapse the variance toward zero, which looks like guidance doing
        nothing rather than like an error. The factor is divided out here.
        """
        if self.setup is None:
            raise RuntimeError(
                "DiffuseBraggRewardFunction.prepare() must be called with the model "
                "atom array before the reward is evaluated."
            )

        n_configs = coordinates.shape[0]
        occupancies = occupancies * n_configs

        f_configs = lunus_sf.structure_factors(self.setup, coordinates, occupancies, self._hkl_t)
        from lunus.sf import mean_and_diffuse

        mean_f, diffuse = mean_and_diffuse(f_configs)

        weight = self._weight()
        total = torch.zeros((), dtype=torch.float32, device=coordinates.device)

        # Zero-weight terms are skipped rather than multiplied by zero: 0 * nan
        # is nan, and a term that cannot contribute should not be able to poison
        # the loss.
        if weight > 0.0:
            total = total + weight * self._residual(torch.abs(mean_f), self._bragg_t)
        if weight < 1.0:
            total = total + (1.0 - weight) * self._residual(self._aniso(diffuse), self._diffuse_t)

        return total
