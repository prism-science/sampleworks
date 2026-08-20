"""Generate synthetic structure factor amplitudes via lunus.sf.

The lunus counterpart of ``generate_synthetic_sf.py``, which uses
SFcalculator. Both write the same MTZ layout — ``F{label}`` / ``SIGF{label}`` /
``PHIF{label}`` plus an optional R-free flag — so the two engines can be compared
directly and either output can feed the structure-factor reward.

Two differences from the SFcalculator script, both deliberate:

**Ensembles, not altlocs.** Input may be a multi-model structure, in which case
each model is one configuration and the emitted amplitude is the ensemble mean
``<F>``. The SFcalculator script instead collapses altloc conformers into a
single weighted structure. Those agree for the first moment, but the collapsed
form cannot express the second (``<|F|²> − |<F>|²``), which is why diffuse work
needs this shape. ``--write-diffuse`` emits that second moment as a separate MTZ
from the same forward pass, and ``--altlocs-as-models`` turns a deposited
multi-conformer structure into the ensemble it already is — which is how to get
a nonzero diffuse target out of a file like ``1vme_final.cif``.

**Grid method, not direct summation.** lunus splats density onto a unit-cell
grid, symmetry-expands, and FFTs, where SFcalculator sums over atoms in
reciprocal space. Agreement is close but not exact — lunus measures correlation
0.999989 and R ≈ 0.0077 against gemmi, with its smooth density taper the main
source of the difference. Prefer to keep engine pairs consistent: a target
generated here and scored by the SFcalculator-backed reward carries that
difference as a floor.
"""

import argparse
import sys
import traceback
from pathlib import Path

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack
from loguru import logger

from sampleworks.core.forward_models.xray import lunus_sf
from sampleworks.synthetic.generate_synthetic_sf import BatchRowForMTZ, load_batch_csv
from sampleworks.synthetic.synthetic_utils import (
    load_structure_for_synthetic_reward,
    resolve_parallel_jobs,
)
from sampleworks.utils.torch_utils import try_gpu


# Bulk-solvent mask parameters. lunus takes an absolute density cutoff rather
# than a quantile (deliberately: a quantile is not permutation-invariant over an
# ensemble). These defaults are a starting point, not a calibration --
# lunus.sf.calibrate_cutoff derives a cutoff for a target solvent fraction and is
# the right tool once the map can be inspected.
DEFAULT_SOLVENT_CUTOFF = 0.20
DEFAULT_SOLVENT_TAPER_WIDTH = 0.10


def generate_asu_hkl(
    unit_cell: gemmi.UnitCell, space_group: gemmi.SpaceGroup, d_min: float
) -> np.ndarray:
    """Build the unique (ASU) Miller indices out to ``d_min``.

    lunus takes the reflection list as an input — it gathers those Miller indices
    off the FFT grid — so unlike SFcalculator, nothing generates one for us.

    Parameters
    ----------
    unit_cell
        Crystal unit cell.
    space_group
        Crystal space group, defining which reflections are unique.
    d_min
        High-resolution limit in Å.

    Returns
    -------
    numpy.ndarray
        ``(n_refl, 3)`` integer Miller indices in the reciprocal ASU, excluding
        ``(0, 0, 0)``.

    Raises
    ------
    ValueError
        If no reflections are produced, which means the cell, space group or
        resolution is wrong rather than that the crystal has no data.
    """
    hkl = np.asarray(
        gemmi.make_miller_array(unit_cell, space_group, d_min), dtype=np.int32
    ).reshape(-1, 3)
    hkl = hkl[np.abs(hkl).sum(axis=1) > 0]  # drop (0,0,0), which lunus cannot phase
    if hkl.size == 0:
        raise ValueError(
            f"No reflections generated for cell {unit_cell.parameters}, space group "
            f"{space_group.hm}, d_min {d_min} A."
        )
    logger.debug(f"Generated {len(hkl)} unique reflections to {d_min} A")
    return hkl


def load_configurations(
    structure_path: Path,
    row: BatchRowForMTZ,
    occupancy_mode: str,
    *,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    altlocs_as_models: bool = False,
) -> tuple[AtomArray, np.ndarray]:
    """Load a structure as a topology plus a stack of configuration coordinates.

    Single-model files go through the shared
    :func:`load_structure_for_synthetic_reward`, so selection, stripping and the
    occupancy modes behave exactly as in the SFcalculator script.

    Multi-model files are loaded directly and **support none of those options**:
    applying a selection consistently across models needs index plumbing that
    does not exist yet, and silently applying it to one model would be worse than
    refusing. Occupancies come from the file as deposited.

    With ``altlocs_as_models``, a single-model structure carrying alternate
    conformations is expanded into one configuration per altloc. A deposited
    multi-conformer model *is* an ensemble, written in the altloc convention
    rather than as models, and expanding it is what makes the diffuse term
    nonzero — the shared backbone contributes nothing to the variance and the
    alternate conformations contribute all of it. Note this is the exact inverse
    of what ``generate_synthetic_sf.py`` does, which collapses altlocs into one
    occupancy-weighted structure: right for amplitudes, and fatal for a second
    moment.

    Parameters
    ----------
    structure_path
        Path to the structure file.
    row
        Batch row supplying ``selection`` and ``occupancy_values``.
    occupancy_mode
        ``'default'``, ``'uniform'`` or ``'custom'``; single-model input only.
    strip_hydrogens, strip_waters, strip_ligands
        Filters; single-model input only.
    altlocs_as_models
        Expand alternate conformations into configurations.

    Returns
    -------
    atom_array : AtomArray
        Topology — elements, B-factors, occupancies. Its own coordinates are
        those of the first configuration.
    coords : numpy.ndarray
        ``(n_configs, n_atoms, 3)`` Cartesian coordinates in Å.

    Raises
    ------
    ValueError
        If a multi-model file is combined with selection, stripping or a
        non-default occupancy mode.
    """
    loaded = load_any(structure_path, altloc="all", extra_fields=["occupancy", "b_factor"])

    if isinstance(loaded, AtomArrayStack) and loaded.stack_depth() > 1:
        unsupported = [
            name
            for name, active in (
                ("selection", row.selection),
                ("--remove-hydrogens", strip_hydrogens),
                ("--remove-waters", strip_waters),
                ("--remove-ligands", strip_ligands),
                ("--occupancy-mode", occupancy_mode != "default"),
                # The models ARE the ensemble here; expanding altlocs on top of
                # them is not a meaningful composition, and this branch returns
                # before the altloc path is reached, so it would be ignored.
                ("--altlocs-as-models", altlocs_as_models),
            )
            if active
        ]
        if unsupported:
            raise ValueError(
                f"{structure_path.name} holds {loaded.stack_depth()} models, and "
                f"{', '.join(unsupported)} is not supported for multi-model input. "
                "Preprocess the ensemble, or use a single-model file."
            )
        logger.info(f"Loaded {loaded.stack_depth()} models from {structure_path.name}")
        return loaded[0], np.asarray(loaded.coord, dtype=np.float64)

    if altlocs_as_models:
        return _expand_altlocs(loaded, structure_path, row.selection)

    atom_array = load_structure_for_synthetic_reward(
        structure_path,
        occupancy_mode=occupancy_mode,
        occupancy_values=row.occupancy_values,
        strip_hydrogens=strip_hydrogens,
        strip_waters=strip_waters,
        strip_ligands=strip_ligands,
        selection=row.selection,
    )
    if atom_array is None:
        raise ValueError(f"Failed to load {structure_path}")
    return atom_array, np.asarray(atom_array.coord, dtype=np.float64)[None, ...]


def _expand_altlocs(
    loaded, structure_path: Path, selection: str | None
) -> tuple[AtomArray, np.ndarray]:
    """Turn a deposited multi-conformer model into one configuration per altloc.

    Wraps :func:`map_altlocs_to_stack`, which returns a stack with the shared
    atoms repeated in every model and the alternate conformations differing —
    exactly the ensemble the diffuse term needs.

    ``map_altlocs_to_stack`` strips ``occupancy``, ``b_factor`` and ``altloc_id``
    off the stack -- biotite cannot hold annotations that conflict between models
    -- and returns them as ``(n_altloc, n_atoms)`` arrays. They have to be put
    back on the topology, since the scattering kernels are built from elements and
    B-factors.

    Two choices are made here, both because the engine bakes B and occupancy in
    per atom rather than per configuration:

    **B-factors are averaged across conformers.** They are identical for the
    shared atoms, so this only affects atoms that genuinely differ, and it beats
    arbitrarily taking the first conformer's.

    **Alternate-conformation atoms are set to full occupancy.** Each
    configuration stands for a unit cell containing that conformer, so averaging
    over configurations reproduces the crystallographic
    ``F_shared + 0.5·F_A + 0.5·F_B`` of a 0.5/0.5 pair. Passing the deposited 0.5
    through as well would apply the weight twice. Atoms that are partially
    occupied for other reasons -- a half-occupied ion, say -- keep their
    deposited value, since only altloc atoms are reweighted.

    Configurations are then weighted equally by ``mean_and_diffuse``, which is
    right for uniform altloc occupancies and wrong for unequal ones; the measured
    populations are logged, and warn when they disagree.

    Raises
    ------
    ValueError
        If the structure has fewer than two altlocs, since a single conformation
        has no variance to report.
    """
    from sampleworks.utils.atom_array_utils import BLANK_ALTLOC_IDS, map_altlocs_to_stack

    stack, annotations = map_altlocs_to_stack(loaded, selection=selection, return_full_array=True)
    if stack.stack_depth() < 2:
        raise ValueError(
            f"{structure_path.name} has fewer than two alternate conformations, so "
            "--altlocs-as-models yields nothing to take a variance over."
        )

    b_factors = np.asarray(annotations["b_factor"], dtype=np.float64)
    occupancies = np.asarray(annotations["occupancy"], dtype=np.float64)
    altloc_ids = np.asarray(annotations["altloc_id"])

    topology = stack[0]
    topology.set_annotation("b_factor", b_factors.mean(axis=0).astype(np.float32))

    # A slot is an alternate if ANY configuration labels it: with
    # return_full_array=True every configuration holds the shared atoms plus its
    # own conformer, so a non-blank altloc in any row marks a position that
    # differs between them. filter_to_common_atoms drops slots missing from any
    # configuration, so the rows do agree in practice; reducing over them anyway
    # keeps this from resting on that.
    is_alternate = (~np.isin(altloc_ids, list(BLANK_ALTLOC_IDS))).any(axis=0)
    per_atom_occupancy = occupancies[0].copy()
    per_atom_occupancy[is_alternate] = 1.0
    topology.set_annotation("occupancy", per_atom_occupancy.astype(np.float32))

    # Averaged over the ALTERNATE atoms only. Averaging over every atom would be
    # dominated by the shared backbone at 1.0 and could never show an imbalance.
    populations = occupancies[:, is_alternate].mean(axis=1) if is_alternate.any() else None
    message = (
        f"Expanded {structure_path.name} into {stack.stack_depth()} configurations "
        f"from altlocs, {int(is_alternate.sum())} alternate atoms of {len(is_alternate)}"
    )
    if populations is None:
        logger.info(message)
    else:
        message += f"; deposited populations {np.round(populations, 3).tolist()}"
        if float(populations.max() - populations.min()) > 0.05:
            logger.warning(
                message + " — these are unequal, but the configurations are weighted "
                "equally, so the diffuse term will not reflect the deposited populations."
            )
        else:
            logger.info(message)

    return topology, np.asarray(stack.coord, dtype=np.float64)


def compute_ensemble_amplitudes(
    atom_array: AtomArray,
    coords: np.ndarray,
    unit_cell: gemmi.UnitCell,
    space_group: gemmi.SpaceGroup,
    resolution: float,
    device: torch.device,
    *,
    solvent_cutoff: float | None = None,
    solvent_taper_width: float = DEFAULT_SOLVENT_TAPER_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``<F>`` and the diffuse term for a set of configurations.

    Parameters
    ----------
    atom_array
        Topology: elements, B-factors and occupancies.
    coords
        ``(n_configs, n_atoms, 3)`` Cartesian coordinates in Å.
    unit_cell, space_group
        Crystal metadata. The atoms are treated as an asymmetric unit and
        symmetry-expanded onto the cell grid before the FFT.
    resolution
        High-resolution limit in Å.
    device
        Torch device.
    solvent_cutoff
        Density below which a voxel is solvent, e/Å³. ``None`` disables bulk
        solvent entirely (no solvent code runs).
    solvent_taper_width
        Width of the mask's smooth transition, e/Å³.

    Returns
    -------
    hkl : numpy.ndarray
        ``(n_refl, 3)`` Miller indices.
    mean_f : numpy.ndarray
        ``(n_refl,)`` complex ensemble mean ``<F>``.
    diffuse : numpy.ndarray
        ``(n_refl,)`` real ``<|F|²> − |<F>|²``. Zero for a single configuration.
    """
    from lunus.sf import mean_and_diffuse

    setup = lunus_sf.build_setup(atom_array, unit_cell, space_group, resolution, device=device)

    hkl_np = generate_asu_hkl(unit_cell, space_group, resolution)
    hkl = torch.as_tensor(hkl_np, dtype=torch.long, device=device)
    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=device)
    occupancies = torch.as_tensor(np.asarray(atom_array.occupancy, dtype=np.float32), device=device)

    solvent = None
    if solvent_cutoff is not None:
        from lunus.sf import SolventModel

        solvent = SolventModel(cutoff=solvent_cutoff, taper_width=solvent_taper_width)

    with torch.no_grad():  # generation only; nothing here needs gradients
        f_configs = lunus_sf.structure_factors(setup, coords_t, occupancies, hkl, solvent=solvent)
        mean_f, diffuse = mean_and_diffuse(f_configs)

    if solvent is not None:
        # SolventModel warns only when the mask is degenerate (all solvent or
        # none). A mask can clear that bar and still be wrong, so report what it
        # measured: `cutoff` is an absolute density in e/A^3 and means nothing
        # until checked against the structure it was applied to.
        #
        # Occupancy should look like a protein crystal's, roughly 0.4-0.7.
        # shell_voxels is a COUNT of voxels inside the taper (strictly between
        # solvent and protein), not a thickness -- report it as a fraction of
        # the grid, since what matters is that the taper is resolved at all. Too
        # few and the mask is a hard threshold sampled on a grid, which for a
        # variance observable like diffuse adds frame-to-frame noise that
        # depends on grid alignment rather than on the structure.
        #
        # Populated only when check_occupancy is on, SolventModel's default.
        if solvent.last_occupancy is None:
            logger.info("Solvent mask applied; occupancy not measured (check_occupancy off)")
        else:
            n_voxels = int(np.prod(setup.grid_shape))
            shell_fraction = solvent.last_shell_voxels / n_voxels
            logger.info(
                f"Solvent mask: occupancy {solvent.last_occupancy:.3f}, "
                f"taper shell {solvent.last_shell_voxels} voxels "
                f"({shell_fraction:.1%} of the {n_voxels} in the grid) "
                f"(cutoff {solvent.cutoff}, taper {solvent.taper_width} e/A^3)"
            )

    return hkl_np, mean_f.cpu().numpy(), diffuse.cpu().numpy()


def dataset_from_intensities(
    hkl: np.ndarray,
    intensities: np.ndarray,
    unit_cell: gemmi.UnitCell,
    space_group: gemmi.SpaceGroup,
    *,
    label: str = "ID",
    output_path: Path | None = None,
) -> rs.DataSet:
    """Write diffuse intensities as an MTZ.

    The column is named ``ID`` and carries the MTZ intensity type ``J``, which is
    what ``lunus/sf/xtraj.py`` writes for ``diffuse=<name>.mtz``. Matching it
    means a target from either source is read the same way, and
    ``DiffuseBraggRewardFunction`` can auto-detect the column in both.

    Parameters
    ----------
    hkl
        ``(n_refl, 3)`` integer Miller indices.
    intensities
        ``(n_refl,)`` real intensities. Diffuse is a variance and so is
        non-negative in exact arithmetic, but float32 cancellation on strong
        reflections can make it slightly negative; the values are written as
        computed rather than clipped, since clipping would bias the target.
    unit_cell, space_group
        Crystal metadata written into the MTZ.
    label
        Column name.
    output_path
        If given, write the dataset there.

    Returns
    -------
    reciprocalspaceship.DataSet
        Indexed by H, K, L with one intensity column.
    """
    dataset = rs.DataSet(
        {
            "H": hkl[:, 0].astype(np.int32),
            "K": hkl[:, 1].astype(np.int32),
            "L": hkl[:, 2].astype(np.int32),
            label: intensities.astype(np.float32),
        },
        cell=unit_cell,
        spacegroup=space_group,
    )
    for miller_column in ("H", "K", "L"):
        dataset[miller_column] = dataset[miller_column].astype(rs.HKLIndexDtype())
    dataset[label] = dataset[label].astype(rs.IntensityDtype())
    dataset = dataset.set_index(["H", "K", "L"])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_mtz(str(output_path))
        logger.info(f"Saved diffuse intensities to {output_path}")

    return dataset


def dataset_from_amplitudes(
    hkl: np.ndarray,
    structure_factors: np.ndarray,
    unit_cell: gemmi.UnitCell,
    space_group: gemmi.SpaceGroup,
    *,
    label: str = "protein",
    sigma_f_scale: float = 0.2,
    test_fraction: float = 0.05,
    seed: int | None = None,
    ccp4_convention: bool = False,
    output_path: Path | None = None,
) -> rs.DataSet:
    """Build an MTZ-ready dataset from Miller indices and complex amplitudes.

    The engine-agnostic half of ``generate_synthetic_sf.process_amplitudes_to_dataset``,
    taking plain arrays rather than an ``SFcalculator``. Emits the same column
    layout so either engine's output is interchangeable downstream.

    Parameters
    ----------
    hkl
        ``(n_refl, 3)`` integer Miller indices.
    structure_factors
        ``(n_refl,)`` complex amplitudes.
    unit_cell, space_group
        Crystal metadata written into the MTZ.
    label
        Column suffix: ``F{label}`` / ``SIGF{label}`` / ``PHIF{label}``.
    sigma_f_scale
        Multiplier synthesizing the sigma column from the amplitudes. The values
        are dummies; they matter only where an R-factor is computed.
    test_fraction
        Fraction of reflections flagged as the R-free test set; 0 disables.
    seed
        Seed for reproducible R-free assignment.
    ccp4_convention
        R-free convention; False (default) is Phenix (1 = test).
    output_path
        If given, write the dataset there as MTZ.

    Returns
    -------
    reciprocalspaceship.DataSet
        Indexed by H, K, L, carrying the amplitude, sigma, phase and optional
        R-free columns.
    """
    f_col, sig_col, phi_col = f"F{label}", f"SIGF{label}", f"PHIF{label}"
    amplitude = np.abs(structure_factors)

    dataset = rs.DataSet(
        {
            "H": hkl[:, 0].astype(np.int32),
            "K": hkl[:, 1].astype(np.int32),
            "L": hkl[:, 2].astype(np.int32),
            f_col: amplitude.astype(np.float32),
            sig_col: (amplitude * sigma_f_scale).astype(np.float32),
            phi_col: np.rad2deg(np.angle(structure_factors)).astype(np.float32),
        },
        cell=unit_cell,
        spacegroup=space_group,
    )

    # Every column needs an MTZ dtype before writing, the Miller indices
    # included: plain int32 has no MTZ type mapping, and rs rejects it at
    # write_mtz() rather than at construction. Cast before set_index, since the
    # index levels keep whatever dtype the columns had.
    for miller_column in ("H", "K", "L"):
        dataset[miller_column] = dataset[miller_column].astype(rs.HKLIndexDtype())
    dataset[f_col] = dataset[f_col].astype(rs.StructureFactorAmplitudeDtype())
    dataset[sig_col] = dataset[sig_col].astype(rs.StandardDeviationDtype())
    dataset[phi_col] = dataset[phi_col].astype(rs.PhaseDtype())

    dataset = dataset.set_index(["H", "K", "L"])

    if test_fraction > 0:
        dataset = rs.utils.add_rfree(
            dataset, ccp4_convention=ccp4_convention, fraction=test_fraction, seed=seed
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_mtz(str(output_path))
        logger.info(f"Saved structure factors to {output_path}")

    return dataset


def _resolve_crystal_metadata(
    row: BatchRowForMTZ, structure_path: Path
) -> tuple[gemmi.UnitCell, gemmi.SpaceGroup]:
    """Resolve cell and space group, preferring per-row overrides over the file."""
    unit_cell = row.unit_cell
    space_group_hm = row.space_group
    if unit_cell is None or space_group_hm is None:
        meta = gemmi.read_structure(str(structure_path))
        if unit_cell is None:
            unit_cell = meta.cell
        if space_group_hm is None:
            space_group_hm = meta.spacegroup_hm
    return unit_cell, gemmi.SpaceGroup(space_group_hm)


def _process_single_row(
    row: BatchRowForMTZ,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    occupancy_mode: str,
    test_fraction: float,
    seed: int | None,
    device: torch.device,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    solvent_cutoff: float | None = None,
    solvent_taper_width: float = DEFAULT_SOLVENT_TAPER_WIDTH,
    write_diffuse: bool = False,
    altlocs_as_models: bool = False,
) -> None:
    """Compute and write synthetic amplitudes for one structure.

    With ``write_diffuse``, a second MTZ of diffuse intensities is written
    alongside the amplitudes, from the same forward pass. It requires a
    multi-model input, since the diffuse term of a single configuration is zero.

    Errors are logged and swallowed so a batch run continues past a bad row,
    matching the SFcalculator script's behaviour.
    """
    structure_path = base_dir / row.filename
    try:
        atom_array, coords = load_configurations(
            structure_path,
            row,
            occupancy_mode,
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            altlocs_as_models=altlocs_as_models,
        )
        unit_cell, space_group = _resolve_crystal_metadata(row, structure_path)
    except Exception as e:
        logger.error(
            f"Failed to load {row.filename} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    try:
        hkl, mean_f, diffuse = compute_ensemble_amplitudes(
            atom_array,
            coords,
            unit_cell,
            space_group,
            resolution,
            device,
            solvent_cutoff=solvent_cutoff,
            solvent_taper_width=solvent_taper_width,
        )
    except Exception as e:
        logger.error(
            f"Failed to compute structure factors for {row.filename} "
            f"({type(e).__name__}): {e}\n{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    if write_diffuse:
        # <|F|²> − |<F>|² is identically zero for one configuration, so a diffuse
        # target from a single model would be a file full of float32 noise.
        # Refusing is better than writing something that looks like data.
        if coords.shape[0] < 2:
            logger.error(
                f"{row.filename}: --write-diffuse needs a multi-model structure; "
                f"got {coords.shape[0]} configuration, whose diffuse term is zero "
                "by construction. Supply an ensemble."
            )
        else:
            logger.info(
                f"{row.filename}: {coords.shape[0]} configurations, "
                f"mean diffuse intensity {float(diffuse.mean()):.4g}"
            )
            diffuse_path = output_dir / (f"{structure_path.stem}_{resolution:.2f}A_diffuse.mtz")
            try:
                dataset_from_intensities(
                    hkl, diffuse, unit_cell, space_group, output_path=diffuse_path
                )
            except Exception as e:
                logger.error(
                    f"Failed to write diffuse MTZ for {row.filename} to {diffuse_path} "
                    f"({type(e).__name__}): {e}\n"
                    f"{''.join(traceback.format_tb(e.__traceback__))}"
                )

    label = "total" if solvent_cutoff is not None else "protein"
    output_path = output_dir / (row.mtzfile or f"{structure_path.stem}_{resolution:.2f}A.mtz")
    try:
        dataset_from_amplitudes(
            hkl,
            mean_f,
            unit_cell,
            space_group,
            label=label,
            test_fraction=test_fraction,
            seed=seed,
            output_path=output_path,
        )
    except Exception as e:
        logger.error(
            f"Failed to write MTZ for {row.filename} to {output_path} "
            f"({type(e).__name__}): {e}\n{''.join(traceback.format_tb(e.__traceback__))}"
        )


def process_batch(
    csv_path: Path,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    occupancy_mode: str,
    test_fraction: float,
    seed: int | None,
    device: torch.device,
    n_jobs: int = -1,
    **row_kwargs,
) -> None:
    """Process every structure listed in a batch CSV.

    Parameters mirror :func:`_process_single_row`; ``n_jobs`` is clamped to 1 on
    CUDA by :func:`resolve_parallel_jobs` to avoid multiple CUDA contexts.
    """
    from joblib import delayed, Parallel

    rows = load_batch_csv(csv_path)
    effective_n_jobs = resolve_parallel_jobs(device, n_jobs)
    logger.info(f"Processing {len(rows)} structures from {csv_path} using {effective_n_jobs} jobs")

    Parallel(n_jobs=effective_n_jobs, backend="loky")(
        delayed(_process_single_row)(
            row=row,
            base_dir=base_dir,
            output_dir=output_dir,
            resolution=resolution,
            occupancy_mode=occupancy_mode,
            test_fraction=test_fraction,
            seed=seed,
            device=device,
            **row_kwargs,
        )
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic structure factor amplitudes via lunus.sf"
    )

    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument("--structure", "-s", type=Path, help="Input structure (mmCIF or PDB)")
    input_group.add_argument("--batch-csv", type=Path, help="CSV for batch processing")
    input_group.add_argument(
        "--base-dir", type=Path, default=Path("."), help="Base directory for CSV relative paths"
    )
    input_group.add_argument("--selection", type=str, help="Atom selection (single-model only)")

    occ_group = parser.add_argument_group("Occupancy Options (single-model only)")
    occ_group.add_argument(
        "--occupancy-mode", choices=["default", "uniform", "custom"], default="default"
    )
    occ_group.add_argument("--occupancy-values", type=str, help="Colon-separated, e.g. '0.3:0.7'")

    sf_group = parser.add_argument_group("Structure Factor Options")
    sf_group.add_argument("--resolution", "-r", type=float, default=1.0, help="d_min in Angstroms")
    sf_group.add_argument("--remove-hydrogens", action="store_true", help="Single-model input only")
    sf_group.add_argument("--remove-waters", action="store_true", help="Single-model input only")
    sf_group.add_argument("--remove-ligands", action="store_true", help="Single-model input only")

    solvent_group = parser.add_argument_group("Bulk Solvent Options")
    solvent_group.add_argument(
        "--simulate-solvent",
        action="store_true",
        help=(
            "Add a flat bulk-solvent contribution and write the total set "
            "(Ftotal/SIGFtotal/PHIFtotal) instead of the protein set. Masks are built "
            "per configuration."
        ),
    )
    solvent_group.add_argument(
        "--solvent-cutoff",
        type=float,
        default=DEFAULT_SOLVENT_CUTOFF,
        help="Density below which a voxel is solvent, e/A^3 (see lunus calibrate_cutoff)",
    )
    solvent_group.add_argument(
        "--solvent-taper-width",
        type=float,
        default=DEFAULT_SOLVENT_TAPER_WIDTH,
        help="Width of the mask's smooth transition, e/A^3",
    )

    diffuse_group = parser.add_argument_group("Diffuse Options")
    diffuse_group.add_argument(
        "--altlocs-as-models",
        action="store_true",
        help=(
            "Treat each alternate conformation as a configuration, rather than "
            "collapsing them into one occupancy-weighted structure. A deposited "
            "multi-conformer model is already an ensemble; this is what makes the "
            "diffuse term nonzero for a single-model file. Configurations are "
            "weighted equally, so unequal altloc occupancies are not reproduced "
            "(a warning says so)."
        ),
    )
    diffuse_group.add_argument(
        "--write-diffuse",
        action="store_true",
        help=(
            "Also write <|F|^2> - |<F>|^2 as an MTZ with an ID intensity column, "
            "matching what lunus xtraj writes. Requires a multi-model structure: "
            "the diffuse term of a single configuration is zero by construction."
        ),
    )

    rfree_group = parser.add_argument_group("R-free Options")
    rfree_group.add_argument("--test-fraction", type=float, default=0.05)
    rfree_group.add_argument("--seed", type=int, default=None)

    crystal_group = parser.add_argument_group("Crystal Options (single-structure mode only)")
    crystal_group.add_argument("--unit-cell", type=str, help="'a:b:c:alpha:beta:gamma'")
    crystal_group.add_argument("--space-group", type=str, help="H-M string or number")

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output", "-o", type=Path, help="Output MTZ path")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for outputs, in both single-structure and batch mode",
    )

    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for batch mode")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = try_gpu()
    solvent_cutoff = args.solvent_cutoff if args.simulate_solvent else None

    row_kwargs = dict(
        strip_hydrogens=args.remove_hydrogens,
        strip_waters=args.remove_waters,
        strip_ligands=args.remove_ligands,
        solvent_cutoff=solvent_cutoff,
        solvent_taper_width=args.solvent_taper_width,
        write_diffuse=args.write_diffuse,
        altlocs_as_models=args.altlocs_as_models,
    )

    if args.batch_csv:
        process_batch(
            csv_path=args.batch_csv,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            resolution=args.resolution,
            occupancy_mode=args.occupancy_mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=device,
            n_jobs=args.n_jobs,
            **row_kwargs,
        )
    elif args.structure:
        row = BatchRowForMTZ.from_dict(
            {
                "filename": args.structure.name,
                "mtzfile": args.output.name if args.output else None,
                "unit_cell": args.unit_cell,
                "space_group": args.space_group,
                "selection": args.selection,
                "occupancy_values": args.occupancy_values,
            }
        )
        # --output names a file and wins when given; otherwise --output-dir
        # applies, in single-structure mode as well as batch. (The SFcalculator
        # script silently ignores --output-dir here and writes to the CWD.)
        _process_single_row(
            row=row,
            base_dir=args.structure.parent,
            output_dir=args.output.parent if args.output else args.output_dir,
            resolution=args.resolution,
            occupancy_mode=args.occupancy_mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=device,
            **row_kwargs,
        )
    else:
        logger.error("Please specify --structure or --batch-csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
