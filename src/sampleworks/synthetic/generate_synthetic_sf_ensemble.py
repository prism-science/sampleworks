"""Generate synthetic structure factors from separate conformer files, without merging them.

``generate_synthetic_sf.py`` computes one structure's structure factors in a single pass,
which means one ``[N_atom, N_HKL]`` scattering table and a handful of intermediates of the
same shape. The atoms are mostly blank-altloc atoms plus a few atoms in altlocs. In
the latest synthetic data generation pipeline that uses PDBFixer, each altloc ends up as a
separate conformer structure file. Even though they can be trivially merged back as one
structure where all atoms have altlocs, such a structure easily leads to OOM.

This script computes Fprotein from one conformer at a time to reduce memory usage, and then
sums Fprotein. The occupancy each conformer contributes is set here. Bulk solvent is currently
excluded; it will be added in a future PR.

The command-line interface mirrors ``generate_synthetic_sf.py``, including CSV batch mode, with
``--conformations`` (or a colon-separated ``filenames`` CSV column) in place of ``--structure``
and ``--occupancy-values`` giving one population per conformer.

Each processed conformer (after filtering and B-factor override) can optionally be saved as
``<input_stem>_sf_input.cif`` beside the MTZ, at occupancy 1.0 so each file is a standalone
structure; the conformer populations are recorded only in the occupancy values.
"""

import argparse
import csv
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import torch
from loguru import logger
from SFC_Torch import SFcalculator

from sampleworks.synthetic.generate_synthetic_sf import (
    build_sfcalculator,
    parse_mtz_batch_row_columns,
    process_amplitudes_to_dataset,
    write_sf_input_structure,
)
from sampleworks.synthetic.synthetic_utils import (
    atomarray_to_gemmi,
    load_structure_for_synthetic_reward,
    resolve_parallel_jobs,
    validate_occupancy_values,
    validate_structure_extension,
)
from sampleworks.utils.torch_utils import try_gpu


CELL_TOLERANCE = 1e-3


@dataclass
class EnsembleBatchRowForMTZ:
    """A row from the ensemble batch CSV file. Each row is one ensemble, written to one MTZ.

    Attributes
    ----------
    filenames
        Conformer structure files (relative to base_dir), in the same order as
        ``occupancy_values``.
    occupancy_values
        Population per conformer, each in [0.0, 1.0] and summing to 1.0.
    mtzfile
        Optional custom output filename for the MTZ.
    unit_cell
        Optional unit cell overriding the one in the conformer files.
    space_group
        Optional Hermann-Mauguin space group overriding the one in the conformer files.
    selection
        Optional atom selection string in PyMOL-like syntax, applied to every conformer.
    """

    filenames: Sequence[Path | str]
    occupancy_values: list[float]
    mtzfile: str | None = None
    unit_cell: gemmi.UnitCell | None = None
    space_group: str | None = None
    selection: str | None = None

    def __post_init__(self) -> None:
        if not self.filenames:
            raise ValueError("An ensemble needs at least one conformer file.")
        for filename in self.filenames:
            validate_structure_extension(filename)
        if len(self.occupancy_values) != len(self.filenames):
            raise ValueError(
                f"Expected one occupancy per conformation, got {len(self.occupancy_values)} "
                f"for {len(self.filenames)} conformations."
            )
        validate_occupancy_values(self.occupancy_values)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "EnsembleBatchRowForMTZ":
        """Create an EnsembleBatchRowForMTZ from a CSV row dictionary.

        CSV columns: filenames (required, colon-separated conformer files relative to
        base_dir, e.g. '1abc_A.cif:1abc_B.cif'), occupancy_values (required, one per
        conformer), plus the optional columns described in ``parse_mtz_batch_row_columns``.
        """
        if "filenames" not in row:
            raise KeyError("CSV is missing required 'filenames' column")
        filenames = [name.strip() for name in row["filenames"].split(":") if name.strip()]
        return cls(filenames=filenames, **parse_mtz_batch_row_columns(row))


def _resolve_ensemble_crystal_metadata(
    structure_paths: list[Path],
    unit_cell: gemmi.UnitCell | None = None,
    space_group: str | None = None,
) -> tuple[gemmi.UnitCell, str]:
    """Resolve the ensemble's shared unit cell and space group, reading whichever is not
    overridden from the conformer files.

    Parameters
    ----------
    structure_paths
        Conformer files carrying crystallographic metadata. Every value read from them
        must agree across conformers, since a summed structure factor is only meaningful
        on one reflection list.
    unit_cell
        Optional unit cell override.
    space_group
        Optional Hermann-Mauguin space group override.

    Returns
    -------
    tuple of (gemmi.UnitCell, str)
        The unit cell and Hermann-Mauguin space group string.

    Raises
    ------
    ValueError
        If the files disagree on a value that was not overridden.
    """
    if unit_cell is not None and space_group is not None:
        return unit_cell, space_group
    structures = [gemmi.read_structure(str(path)) for path in structure_paths]
    reference = structures[0]
    for path, structure in zip(structure_paths[1:], structures[1:], strict=True):
        cell_differs = unit_cell is None and not np.allclose(
            reference.cell.parameters, structure.cell.parameters, atol=CELL_TOLERANCE
        )
        space_group_differs = (
            space_group is None and structure.spacegroup_hm != reference.spacegroup_hm
        )
        if cell_differs or space_group_differs:
            raise ValueError(
                f"{path} is on cell {structure.cell.parameters} / "
                f"'{structure.spacegroup_hm}', but {structure_paths[0]} is on "
                f"{reference.cell.parameters} / '{reference.spacegroup_hm}'."
            )
    return (
        unit_cell if unit_cell is not None else reference.cell,
        space_group if space_group is not None else reference.spacegroup_hm,
    )


def compute_ensemble_fprotein(
    conformation_paths: list[Path],
    occupancies: list[float],
    cell: gemmi.UnitCell,
    space_group: str,
    resolution: float,
    scattering_factor_mode: str,
    device: torch.device,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    selection: str | None = None,
    b_factor: float | None = None,
    save_structure_dir: Path | None = None,
) -> SFcalculator:
    """Sum each conformer's Fprotein onto one reflection list.

    Parameters
    ----------
    conformation_paths
        Conformers of the ensemble, in the same order as ``occupancies``.
    occupancies
        Population per conformer, written over every atom's occupancy.
    cell
        Unit cell shared by the ensemble.
    space_group
        Hermann-Mauguin space group shared by the ensemble.
    resolution
        High-resolution (dmin) limit in Angstroms.
    scattering_factor_mode
        SFcalculator mode: "xray" or "cryoem".
    device
        Torch device for the calculation.
    strip_hydrogens
        If True, remove hydrogen atoms from every conformer.
    strip_waters
        If True, remove water molecules from every conformer.
    strip_ligands
        If True, keep only polymer amino-acid atoms (removes ligands and waters).
    selection
        Optional atom selection applied to every conformer.
    b_factor
        Isotropic B-factor written over every atom, or None to keep the files' values.
    save_structure_dir
        Optional directory in which to save each processed conformer as mmCIF, at
        occupancy 1.0 so each file is a standalone structure.

    Returns
    -------
    SFcalculator
        The last conformer's sfcalculator, carrying the ensemble sum in ``Fprotein_asu``
        and a zero ``Fmask_asu``, ready for ``process_amplitudes_to_dataset``.

    Raises
    ------
    ValueError
        If a conformer cannot be loaded or the conformers do not land on the same
        reflection list.

    Notes
    -----
    Only one conformer's scattering table is alive at a time; each is released before the
    next is built, which is the whole point of summing rather than merging.
    """
    total = None  # [n_hkl] complex ASU amplitudes accumulated over conformers
    sfcalculator = None
    reference_hkl = None  # [n_hkl, 3] ASU Miller indices every conformer must land on
    for path, occupancy in zip(conformation_paths, occupancies, strict=True):
        sfcalculator = None  # release the previous conformer's scattering table first
        torch.cuda.empty_cache()
        atom_array = load_structure_for_synthetic_reward(
            path,
            occupancy_mode="default",
            occupancy_values=[],
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            selection=selection,
            b_factor=b_factor,
        )
        if atom_array is None:
            raise ValueError(f"Failed to load conformation {path}")
        if save_structure_dir is not None:
            # A saved conformer stands alone; its population lives in ``occupancies``.
            atom_array.occupancy[:] = 1.0
            write_sf_input_structure(
                atomarray_to_gemmi(atom_array, cell, space_group), path, save_structure_dir
            )
        atom_array.occupancy[:] = occupancy
        logger.info(f"{path.name}: {len(atom_array)} atoms at occupancy {occupancy:g}")
        gemmi_structure = atomarray_to_gemmi(atom_array, cell, space_group)
        sfcalculator = build_sfcalculator(
            gemmi_structure, resolution, scattering_factor_mode, device
        )
        if reference_hkl is None:
            reference_hkl = sfcalculator.Hasu_array
        elif not np.array_equal(reference_hkl, sfcalculator.Hasu_array):
            raise ValueError(
                f"{path} produced a different reflection list than "
                f"{conformation_paths[0]}; the conformers cannot be summed."
            )
        sfcalculator.calc_fprotein()
        total = sfcalculator.Fprotein_asu if total is None else total + sfcalculator.Fprotein_asu

    # The loop always runs: EnsembleBatchRowForMTZ requires at least one conformation.
    assert sfcalculator is not None and total is not None
    sfcalculator.Fprotein_asu = total
    sfcalculator.Fmask_asu = torch.zeros_like(total)
    return sfcalculator


def _process_single_row(
    row: EnsembleBatchRowForMTZ,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    scattering_factor_mode: str,
    test_fraction: float,
    seed: int | None,
    device: torch.device,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    save_structure: bool = False,
    b_factor: float | None = None,
) -> None:
    """Compute and write the summed protein structure factors of one ensemble.

    Failures are logged rather than raised so one bad row does not stop a batch.

    Parameters
    ----------
    row
        EnsembleBatchRowForMTZ describing the conformers and optional per-row overrides.
    base_dir
        Base directory for resolving relative conformer file paths.
    output_dir
        Directory where the MTZ (and saved conformers) will be written.
    resolution
        High-resolution (dmin) limit in Angstroms.
    scattering_factor_mode
        SFcalculator mode: "xray" or "cryoem".
    test_fraction
        Fraction of reflections to mark as R-free test set (0 disables).
    seed
        Optional seed for reproducible R-free flag assignment.
    device
        PyTorch device for SFcalculator.
    strip_hydrogens
        If True, remove hydrogen atoms before computing structure factors.
    strip_waters
        If True, remove water molecules before computing structure factors.
    strip_ligands
        If True, keep only polymer amino-acid atoms (removes ligands and waters).
    save_structure
        If True, save each processed conformer as ``<input_stem>_sf_input.cif`` in
        output_dir.
    b_factor
        Optional isotropic B-factor assigned to every retained atom.

    Notes
    -----
    Writes the MTZ ``row.mtzfile`` (default ``<first_conformer_stem>_ensemble_<res>A.mtz``)
    to output_dir. The columns are ``Fprotein``/``SIGFprotein``/``PHIFprotein`` plus
    optional R-free flags, the layout ``generate_synthetic_sf.py`` writes without
    ``--simulate-solvent-and-scale``.
    """
    conformation_paths = [base_dir / filename for filename in row.filenames]
    label = ", ".join(path.name for path in conformation_paths)
    try:
        cell, space_group = _resolve_ensemble_crystal_metadata(
            conformation_paths, row.unit_cell, row.space_group
        )
        logger.info(f"Ensemble cell {cell.parameters}, space group '{space_group}'")
        sfcalculator = compute_ensemble_fprotein(
            conformation_paths,
            row.occupancy_values,
            cell,
            space_group,
            resolution,
            scattering_factor_mode,
            device,
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            selection=row.selection,
            b_factor=b_factor,
            save_structure_dir=output_dir if save_structure else None,
        )
    except Exception as e:
        logger.error(
            f"Failed to compute for {label} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    default_name = f"{conformation_paths[0].stem}_ensemble_{resolution:.2f}A.mtz"
    output_path = output_dir / (row.mtzfile or default_name)
    try:
        process_amplitudes_to_dataset(
            sfcalculator,
            structure_factor_columns={"protein": "Fprotein_asu"},
            test_fraction=test_fraction,
            seed=seed,
            output_path=output_path,
        )
    except Exception as e:
        logger.error(
            f"Failed to write MTZ for {label} to {output_path} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return
    logger.info(
        f"Summed {len(conformation_paths)} conformations at occupancies "
        f"{row.occupancy_values!r} into {output_path}"
    )


def load_batch_csv(csv_path: Path) -> list[EnsembleBatchRowForMTZ]:
    """Load and parse a CSV file for batch processing.

    Parameters
    ----------
    csv_path
        Path to CSV file with columns: filenames and occupancy_values (required), mtzfile,
        unit_cell, space_group, selection (all optional)

    Returns
    -------
    list[EnsembleBatchRowForMTZ]
        List of validated batch processing rows

    Raises
    ------
    KeyError
        If the CSV is missing the required 'filenames' column
    """
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "filenames" not in reader.fieldnames:
            raise KeyError(f"CSV file '{csv_path}' is missing required 'filenames' column")
        for row in reader:
            rows.append(EnsembleBatchRowForMTZ.from_dict(row))
    return rows


def process_batch(
    csv_path: Path,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    scattering_factor_mode: str,
    test_fraction: float,
    seed: int | None,
    device: torch.device,
    n_jobs: int = -1,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    save_structure: bool = False,
    b_factor: float | None = None,
) -> None:
    """Process multiple ensembles from a CSV file in batch mode.

    Parameters
    ----------
    csv_path
        Path to CSV file listing ensembles to process, one per row.
    base_dir
        Base directory for resolving relative conformer file paths.
    output_dir
        Directory where output MTZ files will be written.
    resolution
        High-resolution (dmin) limit in Angstroms.
    scattering_factor_mode
        Scattering factor type: 'xray' or 'cryoem'.
    test_fraction
        Fraction of reflections to mark as R-free test set (0 disables).
    seed
        Optional seed for reproducible R-free flag assignment.
    device
        PyTorch device for computation.
    n_jobs
        Number of parallel jobs. -1 means use all available CPUs.
    strip_hydrogens
        If True, remove hydrogen atoms before computing structure factors.
    strip_waters
        If True, remove water molecules before computing structure factors.
    strip_ligands
        If True, keep only polymer amino-acid atoms (removes ligands and waters).
    save_structure
        If True, save each processed conformer as mmCIF to output_dir.
    b_factor
        Optional isotropic B-factor assigned to every retained atom.
    """
    from joblib import delayed, Parallel

    rows = load_batch_csv(csv_path)
    effective_n_jobs = resolve_parallel_jobs(device, n_jobs)
    logger.info(f"Processing {len(rows)} ensembles from {csv_path} using {effective_n_jobs} jobs")

    Parallel(n_jobs=effective_n_jobs, backend="loky")(
        delayed(_process_single_row)(
            row=row,
            base_dir=base_dir,
            output_dir=output_dir,
            resolution=resolution,
            scattering_factor_mode=scattering_factor_mode,
            test_fraction=test_fraction,
            seed=seed,
            device=device,
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            save_structure=save_structure,
            b_factor=b_factor,
        )
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed ensemble structure factor configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sum Fprotein over separate conformer files and write one synthetic MTZ, "
            "without building a merged altloc structure"
        )
    )

    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--conformations",
        "-c",
        type=Path,
        nargs="+",
        help="Conformer files (mmCIF or PDB) of one ensemble, in --occupancy-values order",
    )
    input_group.add_argument("--batch-csv", type=Path, help="Path to CSV file for batch processing")
    input_group.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory for relative paths in CSV, not used in single-ensemble mode",
    )

    selection_group = parser.add_argument_group("Selection Options")
    selection_group.add_argument(
        "--selection",
        type=str,
        help="Atom selection applied to every conformer (e.g., 'chain A and resi 10-50')",
    )

    occupancy_group = parser.add_argument_group("Occupancy Options")
    occupancy_group.add_argument(
        "--occupancy-values",
        type=str,
        help="Colon-separated population per conformation, summing to 1 (e.g., '0.3:0.7')",
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
        "--b-factor",
        type=float,
        default=None,
        help="Override every retained atom's isotropic B-factor",
    )
    sf_group.add_argument(
        "--scattering-factor-mode",
        choices=["xray", "cryoem"],
        default="xray",
        help="Scattering factor type",
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

    crystal_group = parser.add_argument_group("Crystal Options (single-ensemble mode only)")
    crystal_group.add_argument(
        "--unit-cell",
        type=str,
        help="Unit cell as 'a:b:c:alpha:beta:gamma' (overrides the conformers' CRYST1 record)",
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
        help="Save each processed conformer as <input_stem>_sf_input.cif beside the MTZ",
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
    """Generate ensemble MTZs using command-line arguments."""
    args = parse_args()
    device = try_gpu()

    if args.batch_csv:
        process_batch(
            csv_path=args.batch_csv,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            resolution=args.resolution,
            scattering_factor_mode=args.scattering_factor_mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=device,
            n_jobs=args.n_jobs,
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            save_structure=args.save_structure,
            b_factor=args.b_factor,
        )
    elif args.conformations:
        row = EnsembleBatchRowForMTZ(
            filenames=args.conformations,
            **parse_mtz_batch_row_columns(
                {
                    "mtzfile": args.output.name if args.output else None,
                    "unit_cell": args.unit_cell,
                    "space_group": args.space_group,
                    "selection": args.selection,
                    "occupancy_values": args.occupancy_values,
                }
            ),
        )
        _process_single_row(
            row=row,
            base_dir=Path("."),
            output_dir=args.output.parent if args.output else Path("."),
            resolution=args.resolution,
            scattering_factor_mode=args.scattering_factor_mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=device,
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            save_structure=args.save_structure,
            b_factor=args.b_factor,
        )
    else:
        logger.error("Please specify --conformations or --batch-csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
