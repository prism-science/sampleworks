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

Each row runs in three stages: :func:`load_configurations` gives a topology plus
a stack of configuration coordinates, :func:`compute_ensemble_amplitudes` runs
the forward pass and returns arrays, and :func:`dataset_from_bragg_amplitudes` /
:func:`dataset_from_diffuse_intensities` build the datasets :func:`save_mtz`
writes. :func:`_process_single_row` wraps each stage in its own ``try`` so
``batch_report.json`` records which stage a row failed in.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
from atomworks.io.transforms.atom_array import remove_waters
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack
from loguru import logger
from lunus.sf import mean_and_diffuse

from sampleworks.core.forward_models.xray import lunus_sf
from sampleworks.synthetic.synthetic_utils import (
    BatchRowForMTZ,
    load_batch_csv,
    load_structure_for_synthetic_reward,
    resolve_parallel_jobs,
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


# Bulk-solvent mask parameters, e/A^3. lunus takes an absolute density cutoff,
# not a quantile; lunus.sf.calibrate_cutoff derives one for a target solvent
# fraction.
DEFAULT_SOLVENT_CUTOFF = 0.20
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
) -> tuple[AtomArray, np.ndarray]:
    """Load a structure as a topology plus a stack of configuration coordinates.

    Single-model files go through :func:`load_structure_for_synthetic_reward`, so
    selection, stripping and the occupancy modes behave as in the SFcalculator
    script. Multi-model files are loaded as a stack; selection and stripping
    apply to every model, and occupancies are taken as deposited.

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
        If a multi-model file is combined with a non-default occupancy mode or
        ``--altlocs-as-models``, if selection and stripping leave no atoms, or if
        ``--altlocs-as-models`` finds fewer than two alternate conformations.
    """
    loaded = load_any(structure_path, altloc="all", extra_fields=["occupancy", "b_factor"])

    if isinstance(loaded, AtomArrayStack) and loaded.stack_depth() > 1:
        unsupported = [
            name
            for name, active in (
                # A stack has one shared occupancy annotation, so per-model
                # occupancies have nowhere to go.
                ("--occupancy-mode", occupancy_mode != "default"),
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
        return loaded[0], np.asarray(loaded.coord, dtype=np.float64)

    if altlocs_as_models:
        # map_altlocs_to_stack repeats the shared atoms in every model and varies the
        # alternate conformations. It removes occupancy, b_factor and altloc_id from
        # the stack and returns them as (n_altloc, n_atoms) arrays; they go back onto
        # the topology below. Selection is applied first, since map_altlocs_to_stack's
        # own selection argument keeps unselected atoms that carry no altloc.
        # TODO: one configuration per letter couples every altloc A in the structure.
        # Altlocs in consecutive residues usually belong together, but those separated
        # by non-altloc regions need not; equal occupancies are often a sign that
        # two regions' altlocs are coupled. A future option could take a number of
        # configurations and draw A or B independently for each uncoupled region,
        # with probability given by occupancy.
        loaded = apply_selection(loaded, row.selection)
        stack, annotations = map_altlocs_to_stack(loaded, selection=None, return_full_array=True)
        if stack.stack_depth() < 2:
            raise ValueError(
                f"{structure_path.name} has fewer than two alternate conformations, so "
                "--altlocs-as-models yields nothing to take a variance over."
            )

        b_factors = np.asarray(annotations["b_factor"], dtype=np.float64)
        occupancies = np.asarray(annotations["occupancy"], dtype=np.float64)
        altloc_ids = np.asarray(annotations["altloc_id"])

        # B-factors averaged across conformers; identical for the shared atoms.
        topology = stack[0]
        topology.set_annotation("b_factor", b_factors.mean(axis=0).astype(np.float32))

        # An atom is an alternate if any configuration gives it a non-blank altloc.
        is_alternate = (~np.isin(altloc_ids, list(BLANK_ALTLOC_IDS))).any(axis=0)

        # Alternate atoms get full occupancy: averaging over configurations supplies
        # the conformer weights, so the deposited 0.5 would apply them twice. Other
        # partially occupied atoms keep their deposited value.
        per_atom_occupancy = occupancies[0].copy()
        per_atom_occupancy[is_alternate] = 1.0
        topology.set_annotation("occupancy", per_atom_occupancy.astype(np.float32))

        # Deposited populations, averaged over the alternate atoms. mean_and_diffuse
        # weights configurations equally, so unequal populations are not reproduced.
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
                    message + "; these are unequal, but the configurations are weighted "
                    "equally, so the diffuse term will not reflect the deposited populations."
                )
            else:
                logger.info(message)

        return topology, np.asarray(stack.coord, dtype=np.float64)

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
    setup = lunus_sf.build_setup(atom_array, unit_cell, space_group, resolution, device=device)

    hkl_np = generate_asu_hkl(unit_cell, space_group, resolution)
    hkl = torch.as_tensor(hkl_np, dtype=torch.long, device=device)
    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=device)
    occupancies = torch.as_tensor(np.asarray(atom_array.occupancy, dtype=np.float32), device=device)

    solvent = None
    if solvent_cutoff is not None:
        # Inline: lunus.sf loads submodules on first access (PEP 562), so this
        # loads solvent_torch only when a cutoff is requested.
        from lunus.sf import SolventModel

        solvent = SolventModel(cutoff=solvent_cutoff, taper_width=solvent_taper_width)

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


def dataset_from_diffuse_intensities(
    hkl: np.ndarray,
    intensities: np.ndarray,
    unit_cell: gemmi.UnitCell,
    space_group: gemmi.SpaceGroup,
    *,
    label: str = "ID",
) -> rs.DataSet:
    """Build an MTZ-ready dataset of diffuse intensities.

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
    dataset = dataset.set_index(["H", "K", "L"]).infer_mtz_dtypes()
    # infer_mtz_dtypes types a column by its name and gives intensity type only to
    # names starting with "I", so the type is set explicitly for any label.
    dataset[label] = dataset[label].astype(rs.IntensityDtype())

    return dataset


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
) -> rs.DataSet:
    """Build an MTZ-ready dataset from Miller indices and complex amplitudes.

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
        R-free convention; False (default) is Phenix (1 = test).

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

    # Every column, Miller indices included, needs an MTZ dtype before write_mtz().
    dataset = dataset.set_index(["H", "K", "L"]).infer_mtz_dtypes()

    if test_fraction > 0:
        dataset = rs.utils.add_rfree(
            dataset, ccp4_convention=ccp4_convention, fraction=test_fraction, seed=seed
        )

    return dataset


def save_mtz(dataset: rs.DataSet, output_path: Path, description: str) -> None:
    """Write a dataset to disk as an MTZ.

    Parameters
    ----------
    dataset
        Dataset to write, already carrying MTZ dtypes.
    output_path
        Destination; parent directories are created.
    description
        What is being written, for the log line (e.g. ``"diffuse intensities"``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_mtz(str(output_path))
    logger.info(f"Saved {description} to {output_path}")


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
) -> dict[str, Any]:
    """Compute and write synthetic amplitudes for one structure.

    With ``write_diffuse``, a second MTZ of diffuse intensities is written
    alongside the amplitudes, from the same forward pass. It requires a
    multi-model input, since the diffuse term of a single configuration is zero.

    Errors are logged and swallowed so a batch continues past a bad row, as in the
    SFcalculator script. The returned record is what :func:`process_batch`
    collects into ``batch_report.json``.

    Returns
    -------
    dict
        ``filename`` and ``status`` ("success" or "failed"), ``output_path`` and
        ``diffuse_output_path`` for the files written, and ``failure_stage`` and
        ``error`` when the row failed. A failed diffuse write still leaves the
        amplitudes, so a failed row can carry an ``output_path``.
    """
    record: dict[str, Any] = {"filename": row.filename, "status": "failed"}
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
        return record | {"failure_stage": "load", "error": f"{type(e).__name__}: {e}"}

    if write_diffuse and coords.shape[0] < 2:
        # The diffuse term of one configuration is identically zero; refuse before
        # the structure-factor call.
        logger.error(
            f"{row.filename}: --write-diffuse needs a multi-model structure; "
            f"got {coords.shape[0]} configuration, whose diffuse term is zero "
            "by construction. Supply an ensemble. Row skipped."
        )
        return record | {
            "failure_stage": "validate",
            "error": (
                f"--write-diffuse needs a multi-model structure; got "
                f"{coords.shape[0]} configuration"
            ),
        }

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
        return record | {
            "failure_stage": "structure_factors",
            "error": f"{type(e).__name__}: {e}",
        }

    if write_diffuse:
        logger.info(
            f"{row.filename}: {coords.shape[0]} configurations, "
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
            # The amplitudes are still written; the record ends up failed with an output_path.
            record["failure_stage"] = "write_diffuse"
            record["error"] = f"{type(e).__name__}: {e}"

    label = "total" if solvent_cutoff is not None else "protein"
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
        return record | {"failure_stage": "write", "error": f"{type(e).__name__}: {e}"}

    # A failed diffuse write records a stage without stopping the row.
    if "failure_stage" not in record:
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
    solvent_cutoff
        Density below which a voxel is solvent, e/Å³. ``None`` disables bulk
        solvent entirely.
    solvent_taper_width
        Width of the solvent mask's smooth transition, e/Å³.
    write_diffuse
        Also write the diffuse intensities, which needs multi-model input.
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
            solvent_cutoff=solvent_cutoff,
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
        _process_single_row(
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
            solvent_cutoff=solvent_cutoff,
            solvent_taper_width=args.solvent_taper_width,
            write_diffuse=args.write_diffuse,
            altlocs_as_models=args.altlocs_as_models,
        )
    else:
        logger.error("Please specify --structure or --batch-csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
