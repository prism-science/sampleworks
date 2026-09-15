"""Generate synthetic structure factor amplitudes via SFcalculator-torch.

For each input PDB/mmCIF structure, produces an MTZ file of protein structure factors
only, or when ``--simulate-solvent-and-scale`` is set, both the protein and total sets
(Fprotein/SIGFprotein/PHIFprotein and Ftotal/SIGFtotal/PHIFtotal) in the same
MTZ. The MTZ file has dummy values for the SIGF column(s) and optionally an R-free flag
column. Each structure can be optionally overridden with unit cell, space group, atom
selection, and occupancy.
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import cast

import gemmi
import reciprocalspaceship as rs
import reciprocalspaceship.utils
import torch
from loguru import logger
from SFC_Torch import SFcalculator
from SFC_Torch.io import PDBParser

from sampleworks.synthetic.synthetic_utils import (
    atomarray_to_gemmi,
    BatchRowForMTZ,
    load_batch_csv,
    load_structure_for_synthetic_reward,
    resolve_mtz_column,
    resolve_parallel_jobs,
)
from sampleworks.utils.torch_utils import try_gpu


def _build_rs_dataset_for_one_label(
    sfc: SFcalculator,
    label: str,
    structure_factor_column: str,
    miller_index_column: str,
    sigma_f_scale: float,
) -> rs.DataSet:
    """Build an rs.DataSet that contains only one set of F / SIGF / PHIF columns.

    ``sfc.prepare_dataset`` returns an amplitude column and a phase column (degrees)
    for the given ``structure_factor_column`` attribute. We auto-detect those by MTZ
    dtype (rather than assuming the unexposed ``FMODEL`` / ``PHIFMODEL`` names),
    rename them to ``F{label}`` / ``PHIF{label}``, and synthesize a ``SIGF{label}``
    column so several structure-factor sets (e.g. protein and total) can coexist in
    one MTZ.

    Parameters
    ----------
    sfc : SFcalculator
        Structure-factor calculator holding the requested ASU amplitudes and phases.
    label : str
        Mtz column suffix. The resulted column names are ``F{label}``/``SIGF{label}``/
        ``PHIF{label}``. The mtz column names should match what the structure-factor reward
        later reads. In this script, we use ``"protein"`` and ``"total"`` as the labels.
    structure_factor_column : str
        SFcalculator attribute name passed to ``prepare_dataset`` (the amplitudes to emit).
    miller_index_column : str
        SFcalculator attribute name holding the Miller indices for ``prepare_dataset``.
    sigma_f_scale : float
        Multiplier used to synthesize the ``SIGF{label}`` column from the amplitudes.

    Returns
    -------
    rs.DataSet
        Dataset with columns ``F{label}``, ``SIGF{label}``, ``PHIF{label}`` (in that order).
    """
    dataset: rs.DataSet = sfc.prepare_dataset(miller_index_column, structure_factor_column)
    amplitude_column = resolve_mtz_column(dataset, rs.StructureFactorAmplitudeDtype())
    phase_column = resolve_mtz_column(dataset, rs.PhaseDtype())
    logger.debug(
        f"Auto-detected amplitude column: {amplitude_column}, "
        f"phase column: {phase_column} for {label}"
    )
    f_col, phi_col, sig_col = f"F{label}", f"PHIF{label}", f"SIGF{label}"
    # rs.DataSet.rename returns a DataSet at runtime, but the stub types it as DataFrame.
    dataset = cast(
        rs.DataSet,
        dataset.rename(columns={amplitude_column: f_col, phase_column: phi_col}),
    )
    dataset[sig_col] = (dataset[f_col] * sigma_f_scale).astype(rs.StandardDeviationDtype())
    return dataset[[f_col, sig_col, phi_col]]


def process_amplitudes_to_dataset(
    sfc: SFcalculator,
    structure_factor_columns: dict[str, str],
    test_fraction: float = 0.05,
    seed: int | None = None,
    miller_index_column: str = "Hasu_array",
    ccp4_convention: bool = False,
    sigma_f_scale: float = 0.2,
    output_path: Path | None = None,
) -> rs.DataSet:
    """Build an rs.DataSet from SFcalculator outputs, optionally writing it as MTZ.

    Parameters
    ----------
    sfc: SFcalculator
        SFcalculator instance
    structure_factor_columns: dict[str, str]
        Mapping of ``label -> SFcalculator attribute``. Label must match the pattern of
        SFcalculator attribute ``F{label}_asu`` (e.g. ``protein`` ->  ``Fprotein_asu``).
        One structure-factor set (``F{label}``/``SIGF{label}``/``PHIF{label}``) is
        emitted per label, and multiple labels are merged into one MTZ sharing the same
        HKL list (``miller_index_column``). For example, ``{"protein": "Fprotein_asu",
        "total": "Ftotal_asu"}`` produces ``Fprotein``/``SIGFprotein``/``PHIFprotein``/
        ``Ftotal``/``SIGFtotal``/``PHIFtotal`` and optionallyan R-free flag column.
    test_fraction: float
        Fraction of reflections to mark as R-free test set (0 disables)
    seed: int | None
        Optional seed for reproducible R-free flag assignment
    miller_index_column: str
        Attribute name in SFcalculator for hkl indices
    ccp4_convention: bool
        If True, use CCP4 convention for R-free flag assignment. Default
        is False, which uses Phenix convention (1 = test, 0 = working).
    sigma_f_scale: float
        Scale factor to make a fake sigma column from the structure factor amplitude
        values so that SFcalculator can load the output MTZ file as synthetic reward
        without errors. The actual sigma values only matter when computing R-factor.
    output_path: Path | None
        If provided, write the dataset to this MTZ file path.

    Returns
    -------
    rs.DataSet
        Dataset with structure factor amplitudes, dummy sigma column(s), phases,
        and optionally R-free flags.
    """
    # One dataset per label. Labels are free-form column suffixes (F{label}/SIGF{label}/
    # PHIF{label}). User should supply the resulted column names when using the mtz for
    # structure-factor reward. In this script, we use "protein" and "total" (for protein
    # and bulk solvent) as the labels.
    # All sets share the same HKL index, so this is just a column-wise concat.
    dataset: rs.DataSet | None = None
    for label, attribute in structure_factor_columns.items():
        ds = _build_rs_dataset_for_one_label(
            sfc, label, attribute, miller_index_column, sigma_f_scale
        )
        if dataset is None:
            dataset = ds
        else:
            collisions = dataset.columns.intersection(ds.columns).tolist()
            if collisions:
                raise ValueError(
                    f"Structure-factor label {label!r} would overwrite already-populated "
                    f"column(s) {collisions} in the merged dataset. Ensure each label maps "
                    "to a distinct set of column names."
                )
            dataset[ds.columns] = ds  # graft the remaining columns on (index-aligned)
    if dataset is None:  # loop never ran -> the mapping was empty
        raise ValueError(
            "structure_factor_columns must contain at least one "
            "'label -> SFcalculator attribute' entry (e.g. {'protein': 'Fprotein_asu'}), "
            f"got {structure_factor_columns!r}."
        )
    if test_fraction > 0:
        dataset = rs.utils.add_rfree(
            dataset,
            ccp4_convention=ccp4_convention,
            fraction=test_fraction,
            seed=seed,
        )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_mtz(str(output_path))
        logger.info(f"Saved structure factors to {output_path}")
    return dataset


def _process_single_row(
    row: BatchRowForMTZ,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    scattering_factor_mode: str,
    occupancy_mode: str,
    test_fraction: float,
    seed: int | None,
    device: torch.device,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    simulate_solvent_and_scale: bool = False,
    save_structure: bool = False,
) -> None:
    """Compute synthetic protein structure factors for a single structure.
    Assume no anomalous scattering.

    Parameters
    ----------
    row
        BatchRowForMTZ describing the input structure and optional per-row overrides
    base_dir
        Base directory for resolving relative structure file paths
    output_dir
        Directory where output MTZ files will be written
    resolution
        High-resolution (dmin) limit in Angstroms
    scattering_factor_mode
        SFcalculator mode: "xray" or "cryoem"
    occupancy_mode
        Occupancy assignment mode: 'default' (keep file values), 'uniform'
        (1/n_altlocs each), or 'custom' (use per-row occupancy_values from BatchRowForMTZ)
    test_fraction
        Fraction of reflections to mark as R-free test set (0 disables)
    seed
        Optional seed for reproducible R-free flag assignment
    device
        PyTorch device for SFcalculator
    strip_hydrogens
        If True, remove hydrogen atoms before computing structure factors. Default is False.
    strip_waters
        If True, remove water molecules before computing structure factors. Default is False.
    strip_ligands
        If True, remove ligand molecules (non-water heteroatoms) before computing structure
        factors. Default is False.
    simulate_solvent_and_scale
        If True, compute bulk solvent and scale factors and write a single MTZ containing
        both the protein and total structure factor sets. If False (default), only the
        protein set is written. One set contains F{label}/SIGF{label}/PHIF{label}.
    save_structure
        If True, save the processed structure (after selection and occupancy assignment)
        as mmCIF to output_dir. Unit cell and space group are preserved. Default is False.
    """
    structure_path = base_dir / row.filename
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
        return

    # Convert to gemmi for SFcalculator
    try:
        unit_cell = row.unit_cell
        space_group = row.space_group
        if unit_cell is None or space_group is None:
            gemmi_meta = gemmi.read_structure(str(structure_path))
            if unit_cell is None:
                unit_cell = gemmi_meta.cell
            if space_group is None:
                space_group = gemmi_meta.spacegroup_hm
        gemmi_structure = atomarray_to_gemmi(atom_array, unit_cell, space_group)
    except Exception as e:
        logger.error(
            f"Failed to convert {row.filename} to gemmi ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    if save_structure:
        structure_output_path = output_dir / f"{structure_path.stem}_sf_input.cif"
        structure_output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            gemmi_structure.make_mmcif_document().write_file(str(structure_output_path))
            logger.info(f"Saved processed structure to {structure_output_path}")
        except Exception as e:
            logger.error(
                f"Failed to save structure for {row.filename} ({type(e).__name__}): {e}\n"
                f"{''.join(traceback.format_tb(e.__traceback__))}"
            )

    # Compute structure factors
    try:
        sfc = SFcalculator(
            pdbmodel=PDBParser(gemmi_structure),
            mtzdata=None,
            dmin=resolution,
            mode=scattering_factor_mode,
            anomalous=False,
            set_experiment=False,
            device=device,
        )
        logger.debug(
            f"SFC info for {row.filename}: cell: {sfc.unit_cell}, "
            f"space group: {sfc.space_group.hm}, "
            f"n_atoms: {len(sfc.atom_pos_orth)}"
        )
        sfc.calc_fprotein()
        structure_factor_columns = {"protein": "Fprotein_asu"}
        if simulate_solvent_and_scale:
            sfc.inspect_data()
            sfc.calc_fsolvent()
            sfc.init_scales(requires_grad=False)
            sfc.calc_ftotal()
            structure_factor_columns.update({"total": "Ftotal_asu"})
        logger.debug(
            f"Computed {'Fprotein + Ftotal' if simulate_solvent_and_scale else 'Fprotein'} "
            f"for {row.filename} on {device}"
        )
    except Exception as e:
        logger.error(
            f"Failed to compute for {row.filename} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    # Output MTZ file
    output_path = output_dir / (row.mtzfile or f"{structure_path.stem}_{resolution:.2f}A.mtz")
    try:
        process_amplitudes_to_dataset(
            sfc,
            structure_factor_columns=structure_factor_columns,
            test_fraction=test_fraction,
            seed=seed,
            output_path=output_path,
        )
    except Exception as e:
        logger.error(
            f"Failed to write MTZ for {row.filename} to {output_path} "
            f"({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )


def process_batch(
    csv_path: Path,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    scattering_factor_mode: str,
    occupancy_mode: str,
    test_fraction: float,
    seed: int | None,
    device: torch.device,
    n_jobs: int = -1,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    simulate_solvent_and_scale: bool = False,
    save_structure: bool = False,
) -> None:
    """Process multiple structures from a CSV file in batch mode.

    Parameters
    ----------
    csv_path
        Path to CSV file listing structures to process
    base_dir
        Base directory for resolving relative structure file paths
    output_dir
        Directory where output MTZ files will be written
    resolution
        High-resolution (dmin) limit in Angstroms
    scattering_factor_mode
        Scattering factor type: 'xray' or 'cryoem'
    occupancy_mode
        Occupancy assignment mode: 'default', 'uniform', or 'custom'
    test_fraction
        Fraction of reflections to mark as R-free test set (0 disables)
    seed
        Optional seed for reproducible R-free flag assignment
    device
        PyTorch device for computation
    n_jobs
        Number of parallel jobs. -1 means use all available CPUs.
    strip_hydrogens
        If True, remove hydrogen atoms before computing structure factors.
    strip_waters
        If True, remove water molecules before computing structure factors.
    strip_ligands
        If True, keep only polymer amino-acid atoms (removes ligands and waters).
    simulate_solvent_and_scale
        If True, compute bulk solvent and scale factors in addition to F_protein.
    save_structure
        If True, save each processed structure as mmCIF to output_dir.
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
            scattering_factor_mode=scattering_factor_mode,
            occupancy_mode=occupancy_mode,
            test_fraction=test_fraction,
            seed=seed,
            device=device,
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            simulate_solvent_and_scale=simulate_solvent_and_scale,
            save_structure=save_structure,
        )
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic structure factor amplitudes from atomic structures"
    )

    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--structure", "-s", type=Path, help="Path to input structure file (mmCIF or PDB)"
    )
    input_group.add_argument("--batch-csv", type=Path, help="Path to CSV file for batch processing")
    input_group.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory for relative paths in CSV, not used in single-structure mode",
    )

    selection_group = parser.add_argument_group("Selection Options")
    selection_group.add_argument(
        "--selection",
        type=str,
        help="Atom selection (e.g., 'chain A and resi 10-50' or 'chain A and resi 10')",
    )

    occupancy_group = parser.add_argument_group("Occupancy Options")
    occupancy_group.add_argument(
        "--occupancy-mode",
        choices=["default", "uniform", "custom"],
        default="default",
        help="Occupancy assignment mode",
    )
    occupancy_group.add_argument(
        "--occupancy-values",
        type=str,
        help="Colon-separated occupancy values for custom mode (e.g., '0.3:0.7')",
    )

    sf_group = parser.add_argument_group("Structure Factor Options")
    sf_group.add_argument(
        "--resolution",
        "-r",
        type=float,
        default=1.0,
        help="High-resolution (dmin) limit in Angstroms",
    )
    sf_group.add_argument(
        "--scattering-factor-mode",
        choices=["xray", "cryoem"],
        default="xray",
        help="Scattering factor type",
    )
    sf_group.add_argument(
        "--simulate-solvent-and-scale",
        action="store_true",
        help=(
            "Compute bulk solvent and overall scale factors and write both protein and "
            "total structure factor in one MTZ. Without this flag, protein only. Each set "
            "is named by its label (either 'protein' or 'total') and contains: "
            "F{label}/SIGF{label}/PHIF{label}."
        ),
    )
    sf_group.add_argument(
        "--remove-hydrogens",
        action="store_true",
        help="Remove hydrogen atoms before computing structure factors",
    )
    sf_group.add_argument(
        "--remove-waters",
        action="store_true",
        help="Remove water molecules before computing structure factors",
    )
    sf_group.add_argument(
        "--remove-ligands",
        action="store_true",
        help="Remove ligand molecules (non-water heteroatoms) before computing structure factors",
    )

    rfree_group = parser.add_argument_group("R-free Options")
    rfree_group.add_argument(
        "--test-fraction",
        type=float,
        default=0.05,
        help="Fraction of reflections flagged as R-free test set (0 disables)",
    )
    rfree_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible R-free flag assignment",
    )

    crystal_group = parser.add_argument_group("Crystal Options (single-structure mode only)")
    crystal_group.add_argument(
        "--unit-cell",
        type=str,
        help="Unit cell as 'a:b:c:alpha:beta:gamma' (overrides CRYST1 record)",
    )
    crystal_group.add_argument(
        "--space-group",
        type=str,
        help="Space group as Hermann-Mauguin string or number (overrides CRYST1 record)",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save-structure",
        action="store_true",
        help="Save the processed structure (after selection, occupancy assignment) to CIF",
    )
    output_group.add_argument("--output", "-o", type=Path, help="Output MTZ file path")
    output_group.add_argument(
        "--output-dir", type=Path, default=Path("."), help="Output directory for batch mode"
    )

    parallel_group = parser.add_argument_group("Parallelization Options")
    parallel_group.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for batch processing (-1 uses all CPUs)",
    )

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
            scattering_factor_mode=args.scattering_factor_mode,
            occupancy_mode=args.occupancy_mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=device,
            n_jobs=args.n_jobs,
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            simulate_solvent_and_scale=args.simulate_solvent_and_scale,
            save_structure=args.save_structure,
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
        _process_single_row(
            row=row,
            base_dir=args.structure.parent,
            output_dir=args.output.parent if args.output else Path("."),
            resolution=args.resolution,
            scattering_factor_mode=args.scattering_factor_mode,
            occupancy_mode=args.occupancy_mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=device,
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            simulate_solvent_and_scale=args.simulate_solvent_and_scale,
            save_structure=args.save_structure,
        )
    else:
        logger.error("Please specify --structure or --batch-csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
