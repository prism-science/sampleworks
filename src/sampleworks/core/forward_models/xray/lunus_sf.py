"""Adapter between sampleworks structures and ``lunus.sf``.

``lunus.sf`` computes structure factors by evaluating atomic Gaussian density on a
unit-cell grid and taking its FFT. It is differentiable with respect to fractional
coordinates and occupancies, and takes fractional coordinates, IT92 element
symbols and integer grid symmetry operations. This module converts sampleworks'
Cartesian coordinates, biotite ``AtomArray`` and gemmi crystal metadata to that
form.

:func:`build_setup` does the coordinate-independent work once (kernels, grid,
symmetry operations) and :func:`structure_factors` reuses it for every
configuration, the two-phase split ``StructureFactorRewardFunction`` uses for
SFcalculator. :func:`space_group_operations` reads the operations from gemmi;
lunus's own ``build_grid_ops_from_cctbx`` needs cctbx, which is not in the
sampleworks environments.

Symmetry
--------
Pass the asymmetric unit and the crystal's space group. ``F(hkl)`` is the
transform of the whole cell, so :func:`build_setup` expands the ASU onto it
before the FFT; P1 means no expansion.

Coordinates that already fill the cell (an MD box, a pre-expanded model) are not
an asymmetric unit. For Bragg amplitudes, expanding them anyway projects the box
onto its symmetric component, which is what Bragg measures, up to a factor of
the operation count that a scale fit absorbs. For diffuse intensities, pass P1.
The engine cannot distinguish the two cases.

Grid evaluation, blur and taper
-------------------------------
Each atom's density, the sum of Gaussians in ``atom_A`` and ``atom_lam``, is
evaluated on the voxels within a cutoff radius and added to the grid
(``lunus.sf.splat_density``); the FFT of the grid gives ``F(hkl)``. This is the
grid-based alternative to summing over atoms in reciprocal space.

The blur is an anti-aliasing correction. Atoms with low B need a fine grid even
at low resolution. lunus evaluates the density with ``B + blur`` and multiplies
by ``exp(+blur / (4 d^2))`` after the FFT, which leaves ``F(hkl)`` unchanged in
exact arithmetic and changes only the sampling. lunus measures R = 0.186 against
direct summation at B_iso = 2 with no blur and 0.0004 with blur = 20 (7FPV,
0.633 Å grid, d_min 2.0 Å). The correction amplifies sampling error, more at
high resolution, so lunus's ``recommended_blur`` sizes it to the grid's largest
voxel edge and no further.

The taper smooths the cutoff. Both engines truncate each atom's density below a
threshold (lunus's kernel default, 1e-5 e/Å³, is also gemmi's). lunus converts
the threshold to a per-atom radius in ``cutoff_radius_batch``; gemmi tests the
density on the grid. The truncation leaves a step that the FFT rings on; the
taper brings the density to zero continuously over ``taper_width``. Truncation
is the main source of disagreement with gemmi.
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
    orth_matrix,
    SolventModel,
    splat_density,
    structure_factors_batch,
    symmetrize_sum,
)
from lunus.sf.cell_utils import recommended_blur

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.unitcell import (
    UnitCell,
)
from sampleworks.synthetic.synthetic_utils import atomarray_to_gemmi
from sampleworks.utils.elements import normalize_element


if TYPE_CHECKING:
    from biotite.structure import AtomArray


# gemmi's default sampling rate for density grids; lunus's grid_shape_for_resolution
# takes the same convention (spacing = d_min / (2 * rate)).
DEFAULT_GRID_RATE = 1.5


def space_group_operations(
    space_group: str | gemmi.SpaceGroup,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read a space group's symmetry operations from gemmi, for ``build_grid_ops``.

    The gemmi counterpart of lunus's ``build_grid_ops_from_cctbx``. Returns the
    operations in the fractional basis as ``build_grid_ops`` and
    ``adjust_grid_for_symmetry`` expect: integer rotation matrices and fractional
    translations, including the identity, which ``build_grid_ops`` drops itself.

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
        If gemmi does not recognize the symbol, or a rotation matrix is not
        integer-valued, which would mean the operations are not in the
        fractional basis.
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
        # float_seitz() is the 4x4 matrix with the rotation in the upper-left block
        # and the fractional translation in the last column. gemmi stores both as
        # integers over Op.DEN, a common denominator that represents every
        # crystallographic translation (1/2, 1/3, 1/4, 1/6) exactly; float_seitz()
        # has already divided it out.
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
    """Coordinate-independent inputs for one crystal and topology.

    Built by :func:`build_setup` and passed to :func:`structure_factors`. Holds
    the grid, symmetry operations and per-atom scattering kernels, which depend
    on the cell, space group, elements and B-factors but not on coordinates.

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
        :func:`structure_factors`. Taken from qFit's ``UnitCell.orth_to_frac``, the
        formula the real-space density path uses. The fractional coordinates are
        not wrapped into ``[0, 1)``: lunus applies its own modulo in grid-index
        space, and wrapping earlier would put a discontinuity in the gradient at
        the cell boundary.
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
        Candidate voxels per distinct element: integer grid-index offsets from the
        voxel nearest the atom, covering that element's largest atom radius plus
        half a voxel diagonal. An atom with a smaller radius, or one off-centre in
        its voxel, tapers to zero before the outermost offsets. One set per
        element is shared by every atom of that element.
    atom_radius_ang
        ``(n_atoms,)`` cutoff radius per atom, Å.
    taper_width
        Width of the smooth density cutoff, Å.
    blur
        Extra isotropic B applied to the grid density and divided back out in the FFT.
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
        # |det| of the orthogonalization matrix is the cell volume. Taken in float64
        # before the cast to the kernels' dtype, since the volume scales F(hkl).
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
    blur: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LunusSetup:
    """Build the coordinate-independent lunus inputs for one structure.

    Sizes a symmetry-commensurate grid for ``resolution``, reads the space
    group's operations from gemmi, and builds per-atom scattering kernels from
    the array's elements and B-factors.

    Kernels are per atom, since deposited structures carry a B per atom. B-factors
    are fixed here and cannot vary per configuration, as in lunus itself.

    Parameters
    ----------
    atom_array
        Structure defining the topology. Needs ``element`` and ``b_factor``
        annotations. Coordinates are not read; they are passed per call to
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
        ``None`` takes lunus's ``recommended_blur`` for the sharpest atom on this
        grid. ``0.0`` disables it, which is only safe when no atom is sharp
        relative to the grid spacing.
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
    # Sorted so the element ordering, which indexes elem_offsets, is deterministic.
    distinct_elements = sorted(set(elements))
    element_to_index = {symbol: i for i, symbol in enumerate(distinct_elements)}
    element_idx = torch.tensor(
        [element_to_index[e] for e in elements], dtype=torch.long, device=device
    )

    # IT92 coefficients from gemmi, the table the real-space density path reads
    # with scattering_table="it1992", rather than lunus's IT92_COEFFS, which covers a default
    # set and raises KeyError inside the kernel builder for Se, metals or halides.
    # IT92 tabulates neutral atoms, so the charge suffix is dropped; gemmi returns
    # atomic number 0 for a symbol it does not recognize.
    coefficients: dict[str, tuple[float, ...]] = {}
    for symbol in distinct_elements:
        element = gemmi.Element(symbol.rstrip("+-0123456789"))
        if element.atomic_number == 0:
            raise ValueError(
                f"No IT92 scattering coefficients for element {symbol!r} of "
                f"{distinct_elements}. Check the structure's element annotations; "
                "strip or rename the offending atoms if the element is spurious."
            )
        it92 = element.it92
        coefficients[symbol] = (*it92.a, *it92.b, it92.c)

    a, b, c = unit_cell.a, unit_cell.b, unit_cell.c
    orth_np = orth_matrix(a, b, c, unit_cell.alpha, unit_cell.beta, unit_cell.gamma)
    # Fractionalization from qFit's closed form, the code the real-space density
    # path uses, rather than inverting orth_np here. The matrix depends only on
    # the cell parameters.
    frac_np = UnitCell(a, b, c, unit_cell.alpha, unit_cell.beta, unit_cell.gamma).orth_to_frac

    rotations, translations = space_group_operations(space_group)
    raw_shape = grid_shape_for_resolution(a, b, c, resolution, rate)
    # The grid must be invariant under the group, or expanding the ASU onto it
    # would need interpolation between voxels: translations must land on grid
    # points (a 2_1 screw with t = 1/2 forces an even N along that axis), and axes
    # a rotation maps onto each other must share an N (a 4-fold relating a and b
    # forces Nu = Nv). Each axis is also kept 5-smooth for the FFT. gemmi applies
    # the same constraints when sizing its grid.
    grid_shape = adjust_grid_for_symmetry(raw_shape, rotations, translations)
    if tuple(grid_shape) != tuple(raw_shape):
        logger.debug(
            f"Grid {tuple(raw_shape)} adjusted to {tuple(grid_shape)} to satisfy "
            "space-group constraints."
        )
    grid_ops = build_grid_ops(rotations, translations, grid_shape)

    if blur is None:
        blur = recommended_blur(float(np.min(b_factors)), grid_shape, orth_np, d_min=resolution)
        if blur > 0.0:
            logger.info(
                f"Blur {blur:.1f} A^2 added for grid sampling: the sharpest atom is at "
                f"B {float(np.min(b_factors)):.1f} and the grid spacing wants an effective "
                f"B of {blur + float(np.min(b_factors)):.1f}."
            )

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
        configuration: each member gets a mask from its own density, which is
        what a diffuse observable requires.
    use_checkpoint
        Recompute each configuration's grid density during backward instead of
        retaining it. Peak memory becomes flat in ``n_configs`` for ~2.4x the
        time, with bit-identical gradients.
    compile_core
        Forwarded to lunus's ``torch.compile`` path for the grid evaluation.

    Returns
    -------
    torch.Tensor
        Complex ``F(hkl)``, shape ``[n_configs, n_refl]``. Pass to
        ``lunus.sf.mean_and_diffuse`` for the ensemble observables.

    Raises
    ------
    ValueError
        If ``coords`` is not 3-dimensional, its atom count disagrees with the
        setup's, or it is on a different device from the setup.
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

    if coords.device != setup.atom_A.device:
        raise ValueError(
            f"coords are on {coords.device} but the setup was built on {setup.atom_A.device}. "
            "Build the setup on the device the coordinates will use."
        )

    frac = coords @ setup.frac_matrix.to(dtype=coords.dtype).T

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


def geometric_solvent_fraction(
    atom_array: AtomArray,
    coords: np.ndarray,
    unit_cell: gemmi.UnitCell,
    space_group: str | gemmi.SpaceGroup,
    grid_shape: tuple[int, int, int],
    radii_set: str = "Refmac",
) -> float:
    """Fraction of the cell that gemmi's geometric solvent mask calls solvent.

    The calibration target for :func:`calibrate_solvent`: the conventional
    (van der Waals radius plus probe, then shrink) mask on the same atoms and
    grid, so the threshold mask is tuned to call the same volume solvent.

    Parameters
    ----------
    atom_array
        Topology, converted with ``atomarray_to_gemmi`` as the structure-factor
        reward does; its own coordinates are replaced by ``coords``.
    coords
        ``(n_atoms, 3)`` Cartesian coordinates in Å for one configuration.
    unit_cell, space_group
        Crystal metadata.
    grid_shape
        ``(Nu, Nv, Nw)`` grid the mask is evaluated on, normally the setup's.
    radii_set
        A ``gemmi.AtomicRadiiSet`` name.

    Returns
    -------
    float
        Solvent fraction in ``[0, 1]``.
    """
    with_coords = atom_array.copy()
    with_coords.coord = np.asarray(coords, dtype=np.float32)
    hm = space_group if isinstance(space_group, str) else space_group.hm
    model = atomarray_to_gemmi(with_coords, unit_cell, hm)[0]

    grid = gemmi.FloatGrid()
    grid.set_size(*grid_shape)
    grid.set_unit_cell(unit_cell)
    grid.spacegroup = gemmi.SpaceGroup(space_group) if isinstance(space_group, str) else space_group
    gemmi.SolventMasker(getattr(gemmi.AtomicRadiiSet, radii_set)).put_mask_on_float_grid(
        grid, model
    )
    return float(np.asarray(grid, dtype=np.float64).mean())


def calibrate_solvent(
    setup: LunusSetup,
    coords: Float[torch.Tensor, "n_atoms 3"],
    occupancies: Float[torch.Tensor, " n_atoms"],
    target_fraction: float,
    *,
    taper_width_frac: float = 0.5,
) -> SolventModel:
    """A ``SolventModel`` whose cutoff calls ``target_fraction`` of the cell solvent.

    Evaluates the configuration's density on the setup's grid, symmetry-expands
    it, and runs ``SolventModel.calibrate`` with the setup's blur, so the cutoff
    is set on the density the mask will see. The matched cutoff depends on the
    system, its B-factors and the grid, so it is calibrated rather than fixed.

    Parameters
    ----------
    setup
        Built by :func:`build_setup`.
    coords
        ``(n_atoms, 3)`` Cartesian coordinates in Å of a representative
        configuration.
    occupancies
        ``(n_atoms,)`` occupancies for that configuration.
    target_fraction
        Solvent fraction to match, from :func:`geometric_solvent_fraction`.
    taper_width_frac
        Taper width as a fraction of the cutoff, lunus's convention.

    Returns
    -------
    SolventModel
        With ``cutoff`` and ``taper_width`` set; other parameters at lunus defaults.
    """
    frac = coords.to(setup.frac_matrix) @ setup.frac_matrix.T
    with torch.no_grad():
        density = splat_density(
            frac,
            setup.element_idx,
            occupancies.to(frac),
            setup.atom_A,
            setup.atom_lam,
            setup.elem_offsets,
            setup.atom_radius_ang,
            setup.grid_shape,
            setup.orth_matrix,
            setup.taper_width,
        )
        if setup.grid_ops:
            density = symmetrize_sum(density, setup.grid_ops)
        # Placeholders; calibrate() sets both.
        solvent = SolventModel(cutoff=1.0, taper_width=0.5)
        solvent.calibrate(
            density,
            setup.orth_matrix,
            target_fraction,
            taper_width_frac=taper_width_frac,
            pipeline_blur=setup.blur,
        )
    logger.info(
        f"Solvent cutoff calibrated to {solvent.cutoff:.4g} e/A^3 (taper "
        f"{solvent.taper_width:.4g}) for a solvent fraction of {target_fraction:.3f}"
    )
    return solvent
