"""Adapter between sampleworks structures and the ``lunus.sf`` engine.

``lunus.sf`` computes structure factors by splatting atomic Gaussian density onto
a unit-cell grid and FFT-ing it: differentiable with respect to atomic fractional
coordinates and (since 2026-08-16) occupancies. It speaks
fractional coordinates, IT92 element symbols and integer grid symmetry
operations; sampleworks speaks Cartesian coordinates, biotite ``AtomArray``\\ s
and gemmi crystal metadata. This module is the translation.

Everything here is coordinate-independent setup, deliberately: building the
kernels, grid and symmetry operations is expensive and depends only on the
topology and crystal, so it is done once in :func:`build_setup` and reused for
every configuration and every step. That is the same two-phase split
``StructureFactorRewardFunction`` uses for ``SFcalculator``.

The one piece with no upstream equivalent is :func:`space_group_operations`:
lunus ships ``build_grid_ops_from_cctbx``, but cctbx is not in the sampleworks
environments, so the rotations and translations are read from gemmi instead.

Symmetry expansion
------------------
**Pass the asymmetric unit and the crystal's space group.** ``F(hkl)`` is the
transform of the whole unit cell, so :func:`build_setup` reads the group's
operations and expands the ASU onto the full cell before the FFT. Passing P1 is
also fine and simply means no expansion — the grid operations come back empty,
which lunus reads as "nothing to do".

**Coordinates that already fill the cell are a different case, and which group
to pass depends on the observable.** An MD box, possibly a supercell, or a
pre-expanded model is not an asymmetric unit.

For *Bragg* the expansion is legitimate rather than a bug. Summing over the
operations projects the box onto its symmetric component, which is what Bragg
measures, and leaves an overall factor of the operation count that any scale fit
absorbs. Folding an MD snapshot back this way is an effective route to predicted
amplitudes.

For *diffuse*, pass P1. The engine cannot tell the two intents apart -- it sees
an atom array and a group -- so the choice is the caller's.

The splat, the blur and the taper
---------------------------------
Three separate mechanisms, easily confused because two of them sound like
smoothing. None of them is a blur applied *to* the structure.

**Splatting** is how density reaches the grid: it adds the density of an atom to
the total grid density. An atom's density is the sum of Gaussians held in
``atom_A`` and ``atom_lam``, evaluated on the voxels near the atom and
accumulated there; the FFT of the finished grid gives ``F(hkl)``. Those
Gaussians are the atom's own density, not a smoothing kernel, and splatting is
the grid-based alternative to summing over atoms in reciprocal space.

**Blur is an anti-aliasing trick.** If B-factors are low the atomic densities are
sharp, and a finer grid is needed to avoid errors in the structure factors —
even when calculating only to lower resolution. If the B-factors are increased
uniformly one can use a coarser grid, and after calculating, the structure
factors can be corrected by removing the effect of the uniform B-factor. lunus
splats with ``B + blur`` and multiplies by ``exp(+blur / (4 d^2))`` after the
FFT, so in exact arithmetic the answer is unchanged; what changes is sampling.
lunus measures R = 0.186 against exact direct summation at B_iso = 2 with no
blur, versus 0.0004 with blur = 20 (7FPV, 0.633 A grid, d_min 2.0). The
correction is not free — it amplifies whatever sampling error is present, and by
more at higher resolution — so the blur is sized to reach a sampling target and
no further.

**The taper is about the cutoff, not the Gaussians.** A sum of Gaussians is
already smooth. The edge comes from decreasing each atom's footprint by
truncating its density at long distances, which commonly used codes all do —
gemmi by a density threshold, and lunus the same way, at 1e-5 in
``cutoff_radius_batch``. The truncation leaves a sharp, discontinuous edge in
each atomic density contribution, and the FFT rings on it, so the taper smooths
the edge, bringing the density continuously to zero over ``taper_width``.
Truncation is also lunus's main source of disagreement with gemmi.
"""

from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from typing import TYPE_CHECKING

import gemmi
import numpy as np
import torch
from jaxtyping import Complex, Float, Int
from loguru import logger
from lunus.sf import (
    adjust_grid_for_symmetry,
    build_atom_kernels_torch,
    build_grid_ops,
    grid_shape_for_resolution,
    orth_matrix as build_orth_matrix,
    structure_factors_batch,
)

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.unitcell import (
    UnitCell,
)
from sampleworks.utils.elements import it92_coefficients, normalize_element


if TYPE_CHECKING:
    from biotite.structure import AtomArray


# gemmi's default sampling rate for density grids; lunus's grid_shape_for_resolution
# takes the same convention (spacing = d_min / (2 * rate)).
DEFAULT_GRID_RATE = 1.5

# Isotropic B applied on top of each atom's own, widening the Gaussians so the
# grid samples them adequately, then divided back out analytically in
# compute_fcalc. 0.0 disables it.
DEFAULT_BLUR = 0.0


def space_group_operations(
    space_group: str | gemmi.SpaceGroup,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read a space group's symmetry operations from gemmi, for ``build_grid_ops``.

    The gemmi counterpart of lunus's ``build_grid_ops_from_cctbx``. Returns the
    operations in the fractional basis, in the form
    ``lunus.sf.symmetry_torch.build_grid_ops`` and ``adjust_grid_for_symmetry``
    expect: integer rotation matrices and fractional translations, *including*
    the identity (``build_grid_ops`` drops it itself, matching gemmi's
    ``get_scaled_ops_except_id``).

    Parameters
    ----------
    space_group
        Hermann-Mauguin symbol (e.g. ``"P 21 21 21"``) or a ``gemmi.SpaceGroup``.

    Returns
    -------
    rotations : list of numpy.ndarray
        ``(3, 3)`` integer-valued rotation matrices, one per operation.
    translations : list of numpy.ndarray
        ``(3,)`` fractional translations, one per operation.

    Raises
    ------
    ValueError
        If the symbol is not one gemmi recognizes, or if a rotation matrix is not
        integer-valued to within a small tolerance (which would mean the
        operations are not in the fractional basis and grid symmetrization would
        be silently wrong).
    """
    if isinstance(space_group, str):
        resolved = gemmi.SpaceGroup(space_group)
        if resolved is None:  # gemmi raises for most bad input, but be explicit
            raise ValueError(f"gemmi does not recognize space group {space_group!r}")
    else:
        resolved = space_group

    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for op in resolved.operations():
        # float_seitz() is the 4x4 augmented matrix with the rotation in the
        # upper-left block and the fractional translation in the last column.
        # gemmi holds both as integers scaled by Op.DEN, a fixed denominator
        # chosen so that every crystallographic rational translation (1/2, 1/3,
        # 1/4, 1/6, ...) is represented exactly; float_seitz() has already
        # divided it out, so op.rot and op.tran need no hand-scaling here.
        seitz = np.asarray(op.float_seitz(), dtype=np.float64)
        rotation = seitz[:3, :3]
        translation = seitz[:3, 3]

        if not np.allclose(rotation, np.rint(rotation), atol=1e-6):
            raise ValueError(
                f"Rotation for operation {op.triplet()!r} is not integer-valued "
                f"in the fractional basis:\n{rotation}"
            )
        rotations.append(np.rint(rotation).astype(np.int64))
        translations.append(translation % 1.0)

    return rotations, translations


@dataclass(frozen=True)
class LunusSetup:
    """Coordinate-independent inputs for one crystal + topology.

    Built by :func:`build_setup` and passed to :func:`structure_factors`. Holds
    the grid, symmetry operations and per-atom scattering kernels, none of which
    depend on where the atoms are — only on the cell, the space group, the
    elements and the B-factors.

    Attributes
    ----------
    grid_shape
        ``(Nu, Nv, Nw)`` unit-cell grid, symmetry-commensurate and FFT-friendly.
    orth_np, frac_np
        ``(3, 3)`` orthogonalization and fractionalization matrices as NumPy
        float64. Constructor-only: ``__post_init__`` derives the tensors and the
        cell volume from them, and neither is stored.
    orth_matrix
        ``(3, 3)`` orthogonalization matrix, ``cartesian = orth_matrix @ fractional``.
        Derived from ``orth_np``, taking the kernels' dtype and device.
    frac_matrix
        ``(3, 3)`` inverse of ``orth_matrix``, applied to Cartesian coordinates in
        :func:`structure_factors`. Taken from qFit's closed-form
        ``UnitCell.orth_to_frac`` rather than inverted here, so this path and the
        real-space density path share one formula. The fractional coordinates it
        produces are deliberately *not* wrapped into ``[0, 1)``: lunus's splat
        applies its own modulo in grid-index space, and wrapping earlier would
        put a discontinuity in the gradient at the cell boundary.
    cell_volume
        Unit-cell volume in Å³. Derived from ``orth_np``, not passed to the
        constructor.
    grid_ops
        Integer grid operations from ``build_grid_ops``, excluding the identity.
        Empty for P1, which lunus reads as "no symmetry expansion".
    element_idx
        ``(n_atoms,)`` index into the distinct-element ordering.
    atom_A, atom_lam
        ``(n_atoms, 5)`` per-atom Gaussian kernel coefficients.
    elem_offsets
        Candidate voxels for the splat, one set per distinct element: integer
        grid-index offsets from the voxel nearest the atom, covering that
        element's largest atom radius plus half a voxel diagonal. Candidates
        rather than contributors — an atom whose radius is smaller than its
        element's largest, or one sitting off-centre in its voxel, tapers to
        zero before the outermost offsets are reached. Per element rather than
        per atom so one set is built for the cell and shared by every atom of
        that element.
    atom_radius_ang
        ``(n_atoms,)`` cutoff radius per atom, Å.
    taper_width
        Width of the smooth density cutoff, Å.
    blur
        Extra isotropic B applied during the splat and divided back out in the FFT.
    n_atoms
        Atom count the kernels were built for. Coordinates passed to
        :func:`structure_factors` must match.
    """

    grid_shape: tuple[int, int, int]
    orth_np: InitVar[np.ndarray]
    frac_np: InitVar[np.ndarray]
    grid_ops: list
    element_idx: Int[torch.Tensor, " n_atoms"]
    atom_A: Float[torch.Tensor, "n_atoms 5"]
    atom_lam: Float[torch.Tensor, "n_atoms 5"]
    elem_offsets: object
    atom_radius_ang: Float[torch.Tensor, " n_atoms"]
    taper_width: float
    blur: float
    n_atoms: int
    orth_matrix: Float[torch.Tensor, "3 3"] = field(init=False)
    frac_matrix: Float[torch.Tensor, "3 3"] = field(init=False)
    cell_volume: float = field(init=False)

    def __post_init__(self, orth_np: np.ndarray, frac_np: np.ndarray) -> None:
        """Derive the cell volume and the matrix tensors from the NumPy inputs."""
        # |det| of the orthogonalization matrix is the cell volume by
        # construction, taken in float64 before the cast to the kernels' dtype:
        # the volume scales the structure factors, so it should not inherit
        # float32 truncation.
        orth = np.asarray(orth_np, dtype=np.float64)
        object.__setattr__(self, "cell_volume", float(abs(np.linalg.det(orth))))
        for name, matrix in (("orth_matrix", orth), ("frac_matrix", frac_np)):
            object.__setattr__(
                self,
                name,
                torch.as_tensor(
                    np.asarray(matrix, dtype=np.float64),
                    dtype=self.atom_A.dtype,
                    device=self.atom_A.device,
                ),
            )


def build_setup(
    atom_array: AtomArray,
    unit_cell: gemmi.UnitCell,
    space_group: str | gemmi.SpaceGroup,
    resolution: float,
    *,
    rate: float = DEFAULT_GRID_RATE,
    blur: float = DEFAULT_BLUR,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LunusSetup:
    """Build the coordinate-independent lunus inputs for one structure.

    Sizes a symmetry-commensurate grid for ``resolution``, reads the space
    group's operations from gemmi, and builds per-atom scattering kernels from
    the array's elements and B-factors.

    Per-atom (not per-element) kernels are used unconditionally: deposited
    structures carry a B per atom, and lunus's ``build_atom_kernels_torch``
    handles that directly. B-factors are baked in here and cannot vary per
    configuration, which is the same restriction the upstream engine has.

    Parameters
    ----------
    atom_array
        Structure defining the topology. Needs ``element`` and ``b_factor``
        annotations. Coordinates are not read — they are passed per call to
        :func:`structure_factors`.
    unit_cell
        Crystal unit cell.
    space_group
        Hermann-Mauguin symbol or ``gemmi.SpaceGroup``.
    resolution
        High-resolution limit (d_min) in Å, used to size the grid.
    rate
        Grid oversampling rate; spacing is ``d_min / (2 * rate)``.
    blur
        Extra isotropic B for grid sampling, divided back out in the FFT.
    device, dtype
        Torch placement for the kernel tensors.

    Returns
    -------
    LunusSetup
        Everything :func:`structure_factors` needs beyond coordinates and
        occupancies.

    Raises
    ------
    ValueError
        If the atom array carries non-finite B-factors, which would produce
        meaningless kernels.
    """
    b_factors = np.asarray(atom_array.b_factor, dtype=np.float64)
    if not np.isfinite(b_factors).all():
        raise ValueError(
            "atom_array carries non-finite B-factors; wrappers must replace them "
            "(e.g. with 20.0) before building lunus kernels."
        )

    elements = [normalize_element(e) for e in atom_array.element]
    # Sorted for determinism: this ordering defines elem_offsets' index space,
    # and an unstable order would make otherwise-identical runs differ.
    distinct_elements = sorted(set(elements))
    element_to_index = {symbol: i for i, symbol in enumerate(distinct_elements)}
    element_idx = torch.tensor(
        [element_to_index[e] for e in elements], dtype=torch.long, device=device
    )

    # Fetch coefficients for exactly the elements present, rather than using
    # lunus's IT92_COEFFS constant: that is a convenience table covering a
    # DEFAULT set, and anything outside it (Se in a selenomethionine structure,
    # metals, halides) raises KeyError deep inside the kernel builder.
    # it92_coefficients() reads any element gemmi knows, at run time, and is
    # shared with the real-space density path so both read one table.
    try:
        coefficients = it92_coefficients(distinct_elements)
    except KeyError as e:
        # KeyError renders as the repr of its argument, so interpolating the
        # exception itself would wrap the text in a second layer of quotes.
        # it92_coefficients puts a sentence naming the offending symbol in
        # args[0]; surface that directly.
        detail = e.args[0] if e.args else e
        raise ValueError(
            "No IT92 scattering coefficients for an element of "
            f"{distinct_elements}: {detail}. Check the structure's element "
            "annotations; strip or rename the offending atoms if the element "
            "is spurious."
        ) from e

    a, b, c = unit_cell.a, unit_cell.b, unit_cell.c
    orth_np = build_orth_matrix(a, b, c, unit_cell.alpha, unit_cell.beta, unit_cell.gamma)
    # Fractionalization comes from qFit's closed form, the same code the real-space
    # density path uses, rather than inverting orth_np here: one formula, no
    # divergence to discover when comparing the two methods. Its space group is
    # irrelevant to the matrix, which depends only on the cell parameters.
    frac_np = UnitCell(a, b, c, unit_cell.alpha, unit_cell.beta, unit_cell.gamma).orth_to_frac

    rotations, translations = space_group_operations(space_group)
    raw_shape = grid_shape_for_resolution(a, b, c, resolution, rate)
    # The grid has to be invariant under the group, or expanding the ASU onto it
    # would need interpolation between voxels. Two constraints: translations must
    # land on grid points, so a 2_1 screw with t = 1/2 forces an even N along that
    # axis; and axes a rotation maps onto each other must share an N, so a 4-fold
    # relating a and b forces Nu = Nv. Each axis is kept 5-smooth for the FFT on
    # top of that. gemmi applies equivalent constraints when it sizes its own
    # grid, which is why a naive per-axis rounding does not match it.
    grid_shape = adjust_grid_for_symmetry(raw_shape, rotations, translations)
    if tuple(grid_shape) != tuple(raw_shape):
        logger.debug(
            f"Grid {tuple(raw_shape)} adjusted to {tuple(grid_shape)} to satisfy "
            "space-group constraints."
        )
    grid_ops = build_grid_ops(rotations, translations, grid_shape)

    atom_A, atom_lam, elem_offsets, atom_radius_ang, taper_width, _ = build_atom_kernels_torch(
        elements,
        distinct_elements,
        coefficients,
        b_factors,
        blur,
        grid_shape,
        orth_np,
        device=device,
        dtype=dtype,
    )

    logger.info(
        f"lunus setup: {len(elements)} atoms, grid {tuple(grid_shape)}, "
        f"{len(rotations)} symmetry operations ({len(grid_ops)} beyond identity), "
        f"elements {distinct_elements}"
    )

    return LunusSetup(
        grid_shape=tuple(grid_shape),
        orth_np=orth_np,
        frac_np=frac_np,
        grid_ops=grid_ops,
        element_idx=element_idx,
        atom_A=atom_A,
        atom_lam=atom_lam,
        elem_offsets=elem_offsets,
        atom_radius_ang=atom_radius_ang,
        taper_width=taper_width,
        blur=blur,
        n_atoms=len(elements),
    )


def structure_factors(
    setup: LunusSetup,
    coords: Float[torch.Tensor, "n_configs n_atoms 3"],
    occupancies: Float[torch.Tensor, " n_atoms"] | Float[torch.Tensor, "n_configs n_atoms"],
    hkl: Int[torch.Tensor, "n_refl 3"],
    *,
    solvent: object | None = None,
    use_checkpoint: bool = False,
    compile_core: bool = True,
) -> Complex[torch.Tensor, "n_configs n_refl"]:
    """Compute ``F(hkl)`` for a batch of configurations.

    Thin wrapper over ``lunus.sf.structure_factors_batch``: converts Cartesian
    coordinates to fractional and forwards the prebuilt setup. Differentiable
    with respect to ``coords`` and ``occupancies``.

    Parameters
    ----------
    setup
        Built by :func:`build_setup`, for the same atom array these coordinates
        describe.
    coords
        Cartesian coordinates ``[n_configs, n_atoms, 3]`` in Å. A single
        configuration must still carry the leading axis.
    occupancies
        ``[n_atoms]`` shared across configurations, or ``[n_configs, n_atoms]``
        per configuration.
    hkl
        Miller indices ``[n_refl, 3]`` to extract.
    solvent
        A ``lunus.sf.SolventModel``, or None for no bulk solvent. Applied per
        configuration — each member gets a mask from its own density, which is
        the only choice that contributes to a diffuse observable.
    use_checkpoint
        Recompute each configuration's splat during backward instead of retaining
        it. Makes peak memory flat in ``n_configs`` for ~2.4x the time, with
        bit-identical gradients.
    compile_core
        Forwarded to the splat's ``torch.compile`` path.

    Returns
    -------
    torch.Tensor
        Complex ``F(hkl)``, shape ``[n_configs, n_refl]``. Pass to
        ``lunus.sf.mean_and_diffuse`` for the ensemble observables.

    Raises
    ------
    ValueError
        If ``coords`` is not 3-dimensional, or its atom count disagrees with the
        setup's.
    """
    if coords.ndim != 3:
        raise ValueError(
            f"coords must be [n_configs, n_atoms, 3]; got shape {tuple(coords.shape)}. "
            "A single configuration still needs the leading axis."
        )
    if coords.shape[1] != setup.n_atoms:
        raise ValueError(
            f"coords has {coords.shape[1]} atoms but the setup was built for "
            f"{setup.n_atoms}. Rebuild the setup for this atom array."
        )

    frac = coords @ setup.frac_matrix.to(dtype=coords.dtype, device=coords.device).T

    return structure_factors_batch(
        frac,
        setup.element_idx,
        occupancies,
        setup.atom_A,
        setup.atom_lam,
        setup.elem_offsets,
        setup.atom_radius_ang,
        setup.grid_shape,
        setup.orth_matrix,
        setup.cell_volume,
        hkl,
        setup.taper_width,
        blur=setup.blur,
        grid_ops=setup.grid_ops,
        compile_core=compile_core,
        use_checkpoint=use_checkpoint,
        solvent=solvent,
    )
