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
``F(hkl)`` is the transform of the whole unit cell, so an input that is only the
asymmetric unit must be symmetry-expanded before the FFT. That is this module's
regime, and :func:`build_setup` builds the grid operations accordingly (for P1
they come back empty, which lunus reads as "nothing to do").

It is *not* universally correct. Coordinates that already fill the cell — an MD
box, or pre-expanded models — must not be expanded, or every atom is counted once
per symmetry operation. Nothing here detects that case; if this module ever needs
to serve it, the expansion has to become an explicit choice by the caller rather
than an assumption baked into the setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gemmi
import numpy as np
import torch
from jaxtyping import Complex, Float, Int
from loguru import logger

from sampleworks.utils.elements import normalize_element


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
        # upper-left block and the fractional translation in the last column,
        # already divided by gemmi's DEN. Using it avoids hand-scaling op.rot.
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
    orth_matrix
        ``(3, 3)`` orthogonalization matrix, ``cartesian = orth_matrix @ fractional``.
    orth_matrix_np
        The same matrix as numpy, kept because several lunus setup helpers want
        it in that form.
    cell_volume
        Unit-cell volume in Å³.
    grid_ops
        Integer grid operations from ``build_grid_ops``, excluding the identity.
        Empty for P1, which lunus reads as "no symmetry expansion".
    element_idx
        ``(n_atoms,)`` index into the distinct-element ordering.
    atom_A, atom_lam
        ``(n_atoms, 5)`` per-atom Gaussian kernel coefficients.
    elem_offsets
        Per-element candidate voxel offsets for the splat.
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
    orth_matrix: Float[torch.Tensor, "3 3"]
    orth_matrix_np: np.ndarray
    cell_volume: float
    grid_ops: list
    element_idx: Int[torch.Tensor, " n_atoms"]
    atom_A: Float[torch.Tensor, "n_atoms 5"]
    atom_lam: Float[torch.Tensor, "n_atoms 5"]
    elem_offsets: object
    atom_radius_ang: Float[torch.Tensor, " n_atoms"]
    taper_width: float
    blur: float
    n_atoms: int


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
    from lunus.sf import (
        adjust_grid_for_symmetry,
        build_atom_kernels_torch,
        build_grid_ops,
        grid_shape_for_resolution,
        it92_coefficients,
        orth_matrix as build_orth_matrix,
    )

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
    # it92_coefficients() reads any element gemmi knows, at run time.
    try:
        coefficients = it92_coefficients(distinct_elements)
    except KeyError as e:
        raise ValueError(
            f"No IT92 scattering coefficients for element {e} in "
            f"{distinct_elements}. Check the structure's element annotations; "
            "strip or rename the offending atoms if the element is spurious."
        ) from e

    a, b, c = unit_cell.a, unit_cell.b, unit_cell.c
    orth_np = build_orth_matrix(a, b, c, unit_cell.alpha, unit_cell.beta, unit_cell.gamma)

    rotations, translations = space_group_operations(space_group)
    raw_shape = grid_shape_for_resolution(a, b, c, resolution, rate)
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

    orth_t = torch.as_tensor(orth_np, dtype=dtype, device=device)
    logger.info(
        f"lunus setup: {len(elements)} atoms, grid {tuple(grid_shape)}, "
        f"{len(rotations)} symmetry operations ({len(grid_ops)} beyond identity), "
        f"elements {distinct_elements}"
    )

    return LunusSetup(
        grid_shape=tuple(grid_shape),
        orth_matrix=orth_t,
        orth_matrix_np=orth_np,
        cell_volume=float(abs(np.linalg.det(orth_np))),
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


def cartesian_to_fractional(
    coords: Float[torch.Tensor, "*batch n_atoms 3"],
    orth_matrix: Float[torch.Tensor, "3 3"],
) -> Float[torch.Tensor, "*batch n_atoms 3"]:
    """Convert Cartesian coordinates (Å) to fractional, differentiably.

    ``cartesian = orth_matrix @ fractional``, so this applies the inverse. The
    result is *not* wrapped into ``[0, 1)``: lunus's splat applies its own modulo
    when scattering onto the grid, and wrapping here would introduce
    discontinuities in the gradient at cell boundaries.

    Parameters
    ----------
    coords
        Cartesian coordinates in Å.
    orth_matrix
        Orthogonalization matrix from :class:`LunusSetup`.

    Returns
    -------
    torch.Tensor
        Fractional coordinates, same shape as ``coords``.
    """
    inv_orth = torch.linalg.inv(orth_matrix.to(dtype=coords.dtype, device=coords.device))
    return coords @ inv_orth.T


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
    from lunus.sf import structure_factors_batch

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

    frac = cartesian_to_fractional(coords, setup.orth_matrix)

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
