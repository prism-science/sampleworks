"""Generate synthetic structure factor amplitudes via lunus.sf.

The lunus counterpart of ``generate_synthetic_sf.py``, which uses SFcalculator.
Both write the same MTZ layout (``F{label}`` / ``SIGF{label}`` / ``PHIF{label}``
plus an optional R-free flag), so either output can feed the structure-factor
reward. On 6B8X at 2.2 Å the two engines agree to correlation 1.000000 and
R 0.0001 (``tests/synthetic/test_generate_synthetic_sf_lunus.py``).

Two differences from the SFcalculator script. Input may be a multi-model
structure, in which case each model is one configuration and the amplitude
written is the ensemble mean ``<F>``; ``--write-diffuse`` also writes the second
moment ``<|F|²> − |<F>|²``, and ``--altlocs-as-models`` expands a deposited
multi-conformer structure into that ensemble. And lunus evaluates density on a
unit-cell grid and FFTs it rather than summing over atoms in reciprocal space;
see ``core/forward_models/xray/lunus_sf.py`` for the grid, blur and taper.

Each row runs in three stages: :func:`load_configurations` gives the
configurations as an ``AtomArrayStack``, :func:`compute_ensemble_amplitudes` runs
the forward pass and returns arrays, and :func:`dataset_from_bragg_amplitudes` /
:func:`dataset_from_diffuse_intensities` build the MTZ objects :func:`save_mtz`
writes. :func:`_process_single_row` wraps each stage in its own ``try`` so
``batch_report.json`` records which stage a row failed in.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, cast

import gemmi
import numpy as np
import torch
from atomworks.io.transforms.atom_array import remove_waters
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArrayStack, stack
from loguru import logger
from lunus.sf import mean_and_diffuse, SolventModel

from sampleworks.core.forward_models.xray import lunus_sf
from sampleworks.synthetic.synthetic_utils import (
    BatchRowForMTZ,
    load_batch_csv,
    load_structure_for_synthetic_reward,
    resolve_crystal_metadata,
    resolve_parallel_jobs,
    rfree_flags,
)
from sampleworks.utils.atom_array_utils import (
    apply_selection,
    BLANK_ALTLOC_IDS,
    keep_amino_acids,
    keep_polymer,
    map_altlocs_to_stack,
    remove_hydrogens,
)
from sampleworks.utils.torch_utils import try_gpu


# Width in density, e/A^3, of the solvent mask's transition for an explicit cutoff:
# the mask falls from 1 to 0 as the density rises over this interval below the cutoff.
DEFAULT_SOLVENT_TAPER_WIDTH = 0.10


def generate_asu_hkl(
    unit_cell: gemmi.UnitCell, space_group: gemmi.SpaceGroup, d_min: float
) -> np.ndarray:
    """Build the unique (ASU) Miller indices out to ``d_min``.

    lunus gathers the requested Miller indices off the FFT grid, so the list is
    built here rather than by the engine.

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
        If no reflections are produced.
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
) -> AtomArrayStack:
    """Load a structure as a stack of configurations.

    Single-model files go through :func:`load_structure_for_synthetic_reward`, so
    selection, stripping and the occupancy modes behave as in the SFcalculator
    script, and come back as a one-model stack. Multi-model files are loaded as a
    stack; selection and stripping apply to every model, and occupancies are taken
    as deposited.

    With ``altlocs_as_models``, a single-model structure with alternate
    conformations is expanded into one configuration per altloc letter, the
    inverse of the SFcalculator script's collapse into one occupancy-weighted
    structure. Atoms not present under every letter are dropped from all
    configurations. The shared atoms contribute nothing to the diffuse term and
    the alternate conformations contribute all of it.

    Parameters
    ----------
    structure_path
        Path to the structure file.
    row
        Batch row supplying ``selection`` and ``occupancy_values``.
    occupancy_mode
        ``'default'``, ``'uniform'`` or ``'custom'``; single-model input only.
    strip_hydrogens, strip_waters, strip_ligands
        Filters, applied on every path.
    altlocs_as_models
        Expand alternate conformations into configurations.

    Returns
    -------
    AtomArrayStack
        One model per configuration. Elements, B-factors and occupancies are
        per-atom annotations shared by every model; ``stack.coord`` is
        ``(n_configs, n_atoms, 3)`` in Å.

    Raises
    ------
    ValueError
        If a multi-model file or ``--altlocs-as-models`` is combined with an
        occupancy option, or if selection and stripping leave no atoms.
    """
    loaded = load_any(structure_path, altloc="all", extra_fields=["occupancy", "b_factor"])

    if isinstance(loaded, AtomArrayStack) and loaded.stack_depth() > 1:
        unsupported = [
            name
            for name, active in (
                # A stack has one shared occupancy annotation, so per-model
                # occupancies have nowhere to go.
                ("--occupancy-mode", occupancy_mode != "default"),
                ("--occupancy-values", bool(row.occupancy_values)),
                # The models are already the ensemble.
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

        # Per-atom annotations are shared across models, so one mask filters every model.
        loaded = apply_selection(loaded, row.selection)
        loaded = remove_hydrogens(loaded) if strip_hydrogens else loaded
        loaded = remove_waters(loaded) if strip_waters else loaded
        loaded = keep_polymer(keep_amino_acids(loaded)) if strip_ligands else loaded
        if loaded.array_length() == 0:
            raise ValueError(f"{structure_path.name}: selection and stripping left no atoms.")

        logger.info(
            f"Loaded {loaded.stack_depth()} models, {loaded.array_length()} atoms "
            f"from {structure_path.name}"
        )
        return cast(AtomArrayStack, loaded)

    if altlocs_as_models:
        # Alternate atoms are set to full occupancy below and the configurations are
        # weighted equally, so there is no place for a requested occupancy to go.
        if occupancy_mode != "default" or row.occupancy_values:
            raise ValueError(
                f"{structure_path.name}: --occupancy-mode and --occupancy-values are not "
                "supported with --altlocs-as-models; configurations are weighted equally."
            )
        # map_altlocs_to_stack repeats the shared atoms in every model and varies the
        # alternate conformations. It removes occupancy, b_factor and altloc_id from
        # the stack and returns them as (n_altloc, n_atoms) arrays; they go back onto
        # the stack below. Selection and stripping are applied first, since
        # map_altlocs_to_stack's own selection argument keeps unselected atoms that
        # carry no altloc.
        # TODO(#475): one configuration per letter couples every altloc A in the structure.
        # Altlocs in consecutive residues usually belong together, but those separated
        # by non-altloc regions need not; equal occupancies are often a sign that
        # two regions' altlocs are coupled. A future option could take a number of
        # configurations and draw A or B independently for each uncoupled region,
        # with probability given by occupancy.
        loaded = apply_selection(loaded, row.selection)
        loaded = remove_hydrogens(loaded) if strip_hydrogens else loaded
        loaded = remove_waters(loaded) if strip_waters else loaded
        loaded = keep_polymer(keep_amino_acids(loaded)) if strip_ligands else loaded
        configurations, annotations = map_altlocs_to_stack(
            loaded, selection=None, return_full_array=True
        )
        b_factors = np.asarray(annotations["b_factor"], dtype=np.float64)
        occupancies = np.asarray(annotations["occupancy"], dtype=np.float64)
        altloc_ids = np.asarray(annotations["altloc_id"])

        # B-factors averaged across conformers; identical for the shared atoms.
        configurations.set_annotation("b_factor", b_factors.mean(axis=0).astype(np.float32))

        # An atom is an alternate if any configuration gives it a non-blank altloc.
        is_alternate = (~np.isin(altloc_ids, list(BLANK_ALTLOC_IDS))).any(axis=0)

        # Alternate atoms get full occupancy: averaging over configurations supplies
        # the conformer weights, so the deposited 0.5 would apply them twice. Other
        # partially occupied atoms keep their deposited value.
        per_atom_occupancy = occupancies[0].copy()
        per_atom_occupancy[is_alternate] = 1.0
        configurations.set_annotation("occupancy", per_atom_occupancy.astype(np.float32))

        # Deposited populations, averaged over the alternate atoms. mean_and_diffuse
        # weights configurations equally regardless.
        populations = occupancies[:, is_alternate].mean(axis=1) if is_alternate.any() else None
        logger.info(
            f"Expanded {structure_path.name} into {configurations.stack_depth()} configurations "
            f"from altlocs, {int(is_alternate.sum())} alternate atoms of {len(is_alternate)}"
            + (
                f"; deposited populations {np.round(populations, 3).tolist()}, "
                "configurations weighted equally"
                if populations is not None
                else ""
            )
        )
        return configurations

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
    return stack([atom_array])


def compute_ensemble_amplitudes(
    configurations: AtomArrayStack,
    unit_cell: gemmi.UnitCell,
    space_group: gemmi.SpaceGroup,
    resolution: float,
    device: torch.device,
    *,
    bulk_solvent: bool = False,
    solvent_cutoff: float | None = None,
    solvent_taper_width: float = DEFAULT_SOLVENT_TAPER_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``<F>`` and the diffuse term for a set of configurations.

    Parameters
    ----------
    configurations
        One model per configuration, as :func:`load_configurations` returns:
        elements, B-factors and occupancies shared by every model.
    unit_cell, space_group
        Crystal metadata. The atoms are treated as an asymmetric unit and
        symmetry-expanded onto the cell grid before the FFT.
    resolution
        High-resolution limit in Å.
    device
        Torch device.
    bulk_solvent
        If True, add a per-configuration bulk-solvent contribution.
    solvent_cutoff
        Density below which a voxel is solvent, e/Å³. ``None`` calibrates the
        cutoff so the mask calls the same fraction of the cell solvent as gemmi's
        geometric mask on the first configuration.
    solvent_taper_width
        Width of the mask's smooth transition, e/Å³, for an explicit cutoff; a
        calibrated cutoff sets its own taper.

    Returns
    -------
    hkl : numpy.ndarray
        ``(n_refl, 3)`` Miller indices.
    mean_f : numpy.ndarray
        ``(n_refl,)`` complex ensemble mean ``<F>``.
    diffuse : numpy.ndarray
        ``(n_refl,)`` real ``<|F|²> − |<F>|²``. Zero for a single configuration.
    """
    atom_array = configurations[0]
    setup = lunus_sf.build_setup(atom_array, unit_cell, space_group, resolution, device=device)

    hkl_np = generate_asu_hkl(unit_cell, space_group, resolution)
    hkl = torch.as_tensor(hkl_np, dtype=torch.long, device=device)
    coords_t = torch.as_tensor(configurations.coord, dtype=torch.float32, device=device)
    occupancies = torch.as_tensor(np.asarray(atom_array.occupancy, dtype=np.float32), device=device)

    solvent = None
    if bulk_solvent and solvent_cutoff is not None:
        solvent = SolventModel(cutoff=solvent_cutoff, taper_width=solvent_taper_width)
    elif bulk_solvent:
        target = lunus_sf.geometric_solvent_fraction(
            atom_array, atom_array.coord, unit_cell, space_group, setup.grid_shape
        )
        solvent = lunus_sf.calibrate_solvent(setup, coords_t[0], occupancies, target)

    with torch.no_grad():  # generation only; nothing here needs gradients
        f_configs = lunus_sf.structure_factors(setup, coords_t, occupancies, hkl, solvent=solvent)
        mean_f, diffuse = mean_and_diffuse(f_configs)

    if solvent is not None:
        # SolventModel warns only for a degenerate mask (all solvent or none), so
        # the measured solvent occupancy and taper shell (a voxel count, reported
        # as a fraction of the grid) are logged. Reference values on 1VME at 2.0 A
        # (occupancy / shell of grid): the full ASU gives 0.534 / 13.4% at the
        # defaults, chain A alone 0.770 / 7.0%. Off-range values from the same run:
        #
        #   cutoff 0.02, taper 0.10   0.026 / 33.8%   cutoff too low
        #   cutoff 1.00, taper 0.10   1.000 /  0.0%   cutoff too high
        #   cutoff 0.20, taper 0.001  0.603 /  0.1%   taper unresolved on the grid
        #   cutoff 0.20, taper 0.50   0.144 / 60.3%   taper too wide
        #
        # Occupancy near 0 or 1, or a shell fraction near zero, means the mask is
        # not working as intended. last_occupancy is set only when check_occupancy
        # is on, SolventModel's default.
        if solvent.last_occupancy is None:
            logger.info("Solvent mask applied; occupancy not measured (check_occupancy off)")
        else:
            n_voxels = int(np.prod(setup.grid_shape))
            shell_fraction = solvent.last_shell_voxels / n_voxels
            logger.info(
                f"Solvent mask: occupancy {solvent.last_occupancy:.3f}, "
                f"taper shell {solvent.last_shell_voxels} voxels = "
                f"{shell_fraction:.1%} of the {n_voxels} in the grid "
                f"(cutoff {solvent.cutoff}, taper {solvent.taper_width} e/A^3). "
                "1VME at 2.0 A with cutoff 0.20 measures occupancy 0.534 and a "
                "13.4% shell for its full asymmetric unit."
            )

    return hkl_np, mean_f.cpu().numpy(), diffuse.cpu().numpy()


def _new_mtz(unit_cell: gemmi.UnitCell, space_group: gemmi.SpaceGroup) -> gemmi.Mtz:
    """An empty MTZ with H, K, L columns, the cell and the space group set."""
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = space_group
    mtz.set_cell_for_all(unit_cell)
    mtz.add_dataset("sampleworks")
    return mtz


def _set_columns(
    mtz: gemmi.Mtz, hkl: np.ndarray, columns: dict[str, tuple[str, np.ndarray]]
) -> None:
    """Add ``label -> (MTZ type, values)`` columns and fill the data table."""
    for label, (mtz_type, _) in columns.items():
        mtz.add_column(label, mtz_type)
    data = np.column_stack(
        [hkl.astype(np.float32)] + [v.astype(np.float32) for _, v in columns.values()]
    )
    mtz.set_data(data)
    mtz.update_reso()


def dataset_from_diffuse_intensities(
    hkl: np.ndarray,
    intensities: np.ndarray,
    unit_cell: gemmi.UnitCell,
    space_group: gemmi.SpaceGroup,
    *,
    label: str = "ID",
) -> gemmi.Mtz:
    """Build an in-memory MTZ of diffuse intensities.

    The second central moment of the ensemble, ``<|F|²> − |<F>|²``, which cannot
    be derived from the amplitudes :func:`dataset_from_bragg_amplitudes` writes.
    The column is named ``ID`` with MTZ intensity type ``J``, matching what lunus
    xtraj writes for ``diffuse=<name>.mtz``.

    Parameters
    ----------
    hkl
        ``(n_refl, 3)`` integer Miller indices.
    intensities
        ``(n_refl,)`` real intensities. Float32 cancellation can make a few
        slightly negative; they are written as computed, not clipped.
    unit_cell, space_group
        Crystal metadata written into the MTZ.
    label
        Column name.

    Returns
    -------
    gemmi.Mtz
        H, K, L plus one intensity column.
    """
    mtz = _new_mtz(unit_cell, space_group)
    _set_columns(mtz, hkl, {label: ("J", intensities)})
    return mtz


def dataset_from_bragg_amplitudes(
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
) -> gemmi.Mtz:
    """Build an in-memory MTZ from Miller indices and complex amplitudes.

    The first moment of the ensemble, ``<F>``, written as ``|<F>|`` plus its phase,
    in the column layout of ``generate_synthetic_sf.process_amplitudes_to_dataset``.

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
        Multiplier synthesizing the sigma column from the amplitudes.
    test_fraction
        Fraction of reflections flagged as the R-free test set; 0 disables.
    seed
        Seed for reproducible R-free assignment.
    ccp4_convention
        R-free convention; see :func:`rfree_flags`.

    Returns
    -------
    gemmi.Mtz
        H, K, L plus the amplitude, sigma, phase and optional ``R-free-flags`` columns.
    """
    amplitude = np.abs(structure_factors)
    columns = {
        f"F{label}": ("F", amplitude),
        f"SIGF{label}": ("Q", amplitude * sigma_f_scale),
        f"PHIF{label}": ("P", np.rad2deg(np.angle(structure_factors))),
    }
    if test_fraction > 0:
        label, flags = rfree_flags(len(hkl), test_fraction, seed, ccp4_convention)
        columns[label] = ("I", flags)

    mtz = _new_mtz(unit_cell, space_group)
    _set_columns(mtz, hkl, columns)
    return mtz


def save_mtz(mtz: gemmi.Mtz, output_path: Path, description: str) -> None:
    """Write an MTZ to disk.

    Parameters
    ----------
    mtz
        MTZ to write.
    output_path
        Destination; parent directories are created.
    description
        What is being written, for the log line (e.g. ``"diffuse intensities"``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mtz.write_to_file(str(output_path))
    logger.info(f"Saved {description} to {output_path}")


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
    bulk_solvent: bool = False,
    solvent_cutoff: float | None = None,
    solvent_taper_width: float = DEFAULT_SOLVENT_TAPER_WIDTH,
    write_diffuse: bool = False,
    altlocs_as_models: bool = False,
) -> dict[str, Any]:
    """Compute and write synthetic amplitudes for one structure.

    With ``write_diffuse``, a second MTZ of diffuse intensities is written
    alongside the amplitudes, from the same forward pass.

    Errors are logged and swallowed so a batch continues past a bad row, as in the
    SFcalculator script. The returned record is what :func:`process_batch`
    collects into ``batch_report.json``.

    Returns
    -------
    dict
        ``filename`` and ``status`` ("success" or "failed"), ``output_path`` and
        ``diffuse_output_path`` for the files written, and ``failure_stage`` and
        ``error`` when the row failed.
    """
    record: dict[str, Any] = {"filename": row.filename, "status": "failed"}
    structure_path = base_dir / row.filename
    try:
        configurations = load_configurations(
            structure_path,
            row,
            occupancy_mode,
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            altlocs_as_models=altlocs_as_models,
        )
        unit_cell, space_group_hm = resolve_crystal_metadata(row, structure_path)
        space_group = gemmi.SpaceGroup(space_group_hm)
    except Exception as e:
        logger.error(
            f"Failed to load {row.filename} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        record["failure_stage"] = "load"
        record["error"] = f"{type(e).__name__}: {e}"
        return record

    try:
        hkl, mean_f, diffuse = compute_ensemble_amplitudes(
            configurations,
            unit_cell,
            space_group,
            resolution,
            device,
            bulk_solvent=bulk_solvent,
            solvent_cutoff=solvent_cutoff,
            solvent_taper_width=solvent_taper_width,
        )
    except Exception as e:
        logger.error(
            f"Failed to compute structure factors for {row.filename} "
            f"({type(e).__name__}): {e}\n{''.join(traceback.format_tb(e.__traceback__))}"
        )
        record["failure_stage"] = "structure_factors"
        record["error"] = f"{type(e).__name__}: {e}"
        return record

    if write_diffuse:
        logger.info(
            f"{row.filename}: {configurations.stack_depth()} configurations, "
            f"mean diffuse intensity {float(diffuse.mean()):.4g}"
        )
        diffuse_path = output_dir / (f"{structure_path.stem}_{resolution:.2f}A_diffuse.mtz")
        try:
            save_mtz(
                dataset_from_diffuse_intensities(hkl, diffuse, unit_cell, space_group),
                diffuse_path,
                "diffuse intensities",
            )
            record["diffuse_output_path"] = str(diffuse_path)
        except Exception as e:
            logger.error(
                f"Failed to write diffuse MTZ for {row.filename} to {diffuse_path} "
                f"({type(e).__name__}): {e}\n"
                f"{''.join(traceback.format_tb(e.__traceback__))}"
            )
            record["failure_stage"] = "write_diffuse"
            record["error"] = f"{type(e).__name__}: {e}"
            return record

    label = "total" if bulk_solvent else "protein"
    output_path = output_dir / (row.mtzfile or f"{structure_path.stem}_{resolution:.2f}A.mtz")
    try:
        save_mtz(
            dataset_from_bragg_amplitudes(
                hkl,
                mean_f,
                unit_cell,
                space_group,
                label=label,
                test_fraction=test_fraction,
                seed=seed,
            ),
            output_path,
            "structure factors",
        )
        record["output_path"] = str(output_path)
    except Exception as e:
        logger.error(
            f"Failed to write MTZ for {row.filename} to {output_path} "
            f"({type(e).__name__}): {e}\n{''.join(traceback.format_tb(e.__traceback__))}"
        )
        record["failure_stage"] = "write"
        record["error"] = f"{type(e).__name__}: {e}"
        return record

    record["status"] = "success"
    return record


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
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    bulk_solvent: bool = False,
    solvent_cutoff: float | None = None,
    solvent_taper_width: float = DEFAULT_SOLVENT_TAPER_WIDTH,
    write_diffuse: bool = False,
    altlocs_as_models: bool = False,
) -> dict[str, Any]:
    """Process every structure listed in a batch CSV.

    Writes ``batch_report.json`` into ``output_dir``, a summary plus one record per
    row, and returns the same report. Rows fail independently.

    Parameters
    ----------
    csv_path
        Batch CSV, as read by :func:`load_batch_csv`.
    base_dir
        Directory the rows' ``filename`` entries are relative to.
    output_dir
        Where the MTZs and ``batch_report.json`` are written.
    resolution
        High-resolution limit (d_min) in Å.
    occupancy_mode
        ``'default'``, ``'uniform'`` or ``'custom'``; single-model input only.
    test_fraction
        Fraction of reflections flagged R-free, or 0 for no flags.
    seed
        Seed for the R-free selection. ``None`` gives different flags per run.
    device
        Torch device for the forward pass.
    n_jobs
        Worker count, clamped to 1 on CUDA by :func:`resolve_parallel_jobs` to
        avoid multiple CUDA contexts.
    strip_hydrogens, strip_waters, strip_ligands
        Filters applied after any per-row selection.
    bulk_solvent
        Add a per-configuration bulk-solvent contribution and write the total set.
    solvent_cutoff
        Density below which a voxel is solvent, e/Å³; ``None`` calibrates it to
        gemmi's geometric mask.
    solvent_taper_width
        Width of the solvent mask's smooth transition, e/Å³, for an explicit cutoff.
    write_diffuse
        Also write the diffuse intensities.
    altlocs_as_models
        Expand alternate conformations into configurations.

    Returns
    -------
    dict[str, Any]
        The report written to ``batch_report.json``.
    """
    from joblib import delayed, Parallel

    rows = load_batch_csv(csv_path)
    effective_n_jobs = resolve_parallel_jobs(device, n_jobs)
    logger.info(f"Processing {len(rows)} structures from {csv_path} using {effective_n_jobs} jobs")

    records = Parallel(n_jobs=effective_n_jobs, backend="loky")(
        delayed(_process_single_row)(
            row=row,
            base_dir=base_dir,
            output_dir=output_dir,
            resolution=resolution,
            occupancy_mode=occupancy_mode,
            test_fraction=test_fraction,
            seed=seed,
            device=device,
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            bulk_solvent=bulk_solvent,
            solvent_cutoff=solvent_cutoff,
            solvent_taper_width=solvent_taper_width,
            write_diffuse=write_diffuse,
            altlocs_as_models=altlocs_as_models,
        )
        for row in rows
    )

    succeeded = sum(1 for r in records if r["status"] == "success")
    # Same shape as run_grid_search.py's report: a "runs" list plus a "summary" block.
    report = {
        "runs": list(records),
        "summary": {
            "total": len(records),
            "successful": succeeded,
            "failed": len(records) - succeeded,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "batch_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"{succeeded}/{len(records)} rows succeeded; per-row detail in {report_path}")
    return report


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
        default=None,
        help="Density below which a voxel is solvent, e/A^3 (default: calibrated so the "
        "mask matches the solvent fraction of gemmi's geometric mask)",
    )
    solvent_group.add_argument(
        "--solvent-taper-width",
        type=float,
        default=DEFAULT_SOLVENT_TAPER_WIDTH,
        help="Width of the mask's smooth transition, e/A^3, for an explicit --solvent-cutoff",
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
            "weighted equally, so unequal altloc occupancies are not reproduced."
        ),
    )
    diffuse_group.add_argument(
        "--write-diffuse",
        action="store_true",
        help=(
            "Also write <|F|^2> - |<F>|^2 as an MTZ with an ID intensity column, "
            "matching what lunus xtraj writes. Zero for a single configuration."
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
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            bulk_solvent=args.simulate_solvent,
            solvent_cutoff=args.solvent_cutoff,
            solvent_taper_width=args.solvent_taper_width,
            write_diffuse=args.write_diffuse,
            altlocs_as_models=args.altlocs_as_models,
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
        # --output names the file when given; otherwise --output-dir applies here too.
        record = _process_single_row(
            row=row,
            base_dir=args.structure.parent,
            output_dir=args.output.parent if args.output else args.output_dir,
            resolution=args.resolution,
            occupancy_mode=args.occupancy_mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=device,
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            bulk_solvent=args.simulate_solvent,
            solvent_cutoff=args.solvent_cutoff,
            solvent_taper_width=args.solvent_taper_width,
            write_diffuse=args.write_diffuse,
            altlocs_as_models=args.altlocs_as_models,
        )
        if record["status"] != "success":
            sys.exit(1)
    else:
        logger.error("Please specify --structure or --batch-csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
