import argparse
import csv
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch
from loguru import logger

from sampleworks.core.forward_models.xray.real_space_density import XMap_torch
from sampleworks.synthetic.synthetic_utils import (
    load_structure_for_synthetic_reward,
    resolve_parallel_jobs,
    validate_occupancy_values,
)
from sampleworks.utils.atom_array_utils import save_structure_to_cif
from sampleworks.utils.density_utils import compute_density_from_atomarray
from sampleworks.utils.torch_utils import try_gpu


@dataclass
class BatchRow:
    """A row from the batch processing CSV file.

    Attributes
    ----------
    filename
        Path to the structure file (relative to base_dir)
    selection
        Optional atom selection string in pyMOL-like syntax (e.g., 'chain A and resi 10-50')
    occupancy_values
        Custom occupancy values for altlocs, must be in range [0.0, 1.0]
    mapfile
        Optional custom output filename for the density map
    """

    VALID_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".cif", ".mmcif"})
    LEGACY_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".pdb", ".ent"})

    filename: str
    selection: str | None = None
    occupancy_values: list[float] = field(default_factory=list)
    mapfile: str | None = None

    def __post_init__(self) -> None:
        ext = Path(self.filename).suffix.lower()
        all_supported = self.VALID_EXTENSIONS | self.LEGACY_EXTENSIONS
        if ext not in all_supported:
            raise ValueError(
                f"Invalid file extension '{ext}' for '{self.filename}'. "
                f"Expected one of: {', '.join(sorted(all_supported))}"
            )
        if ext in self.LEGACY_EXTENSIONS:
            logger.warning(
                f"'{ext}' is a legacy PDB format and support may be removed in a future version. "
                "Prefer .cif or .mmcif (mmCIF format)."
            )
        validate_occupancy_values(self.occupancy_values)

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "BatchRow":
        """Create a BatchRow from a CSV row dictionary.

        Parameters
        ----------
        row
            Dictionary with keys 'filename' (required), and optionally
            'selection', 'occupancy_values' (colon-separated), and 'mapfile'

        Returns
        -------
        BatchRow
            Validated batch row instance

        Raises
        ------
        KeyError
            If required 'filename' column is missing
        ValueError
            If occupancy values are invalid
        """
        if "filename" not in row:
            raise KeyError("CSV is missing required 'filename' column")

        occupancy_values: list[float] = []
        occupancy_values_csv = row.get("occupancy_values") or row.get("occ_values")
        if occupancy_values_csv:
            occupancy_values = [float(v.strip()) for v in occupancy_values_csv.split(":")]

        return cls(
            filename=row["filename"],
            selection=row.get("selection") or None,
            occupancy_values=occupancy_values,
            mapfile=row.get("mapfile") or None,
        )


def save_density(density: torch.Tensor, xmap_torch: XMap_torch, output_path: Path) -> None:
    """Save a density map to disk in CCP4 format.

    Parameters
    ----------
    density
        Computed density tensor
    xmap_torch
        XMap_torch object containing grid parameters
    output_path
        Path where the CCP4 map file will be written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xmap_torch.tofile(str(output_path), density)
    logger.info(f"Saved density map to {output_path}")


def load_batch_csv(csv_path: Path) -> list[BatchRow]:
    """Load and parse a CSV file for batch processing.

    Parameters
    ----------
    csv_path
        Path to CSV file with columns: filename (required), selection (optional),
        occupancy_values (optional), mapfile (optional)

    Returns
    -------
    list[BatchRow]
        List of validated batch processing rows

    Raises
    ------
    KeyError
        If the CSV is missing the required 'filename' column
    """
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise KeyError(f"CSV file '{csv_path}' is missing required 'filename' column")
        for row in reader:
            rows.append(BatchRow.from_dict(row))
    return rows


def _process_single_row(
    row: BatchRow,
    occupancy_mode: str,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    em_mode: bool,
    device: torch.device,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    save_structure: bool = True,
) -> None:
    """Process a single structure row.

    Parameters
    ----------
    row
        BatchRow containing structure information
    occupancy_mode
        Occupancy assignment mode: 'default', 'uniform', or 'custom'
    base_dir
        Base directory for resolving relative structure file paths
    output_dir
        Directory where output density maps will be written
    resolution
        Map resolution in Angstroms
    em_mode
        If True, use electron scattering factors. If False, use X-ray factors.
    device
        PyTorch device for computation
    strip_hydrogens
        If True, remove hydrogen atoms before computing density. Default is False.
    strip_waters
        If True, remove water molecules before computing density. Default is False.
    strip_ligands
        If True, remove ligand molecules (non-water heteroatoms) before computing density. Default
        is False.
        This is done by keeping only polymer atoms in the selection, which are typically not
        ligands.
        TODO: be more thorough with this? We could make this a transform
    save_structure
        If True, save the processed structure to a CIF file in the input directory. Default is True.
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

    try:
        density, xmap_torch = compute_density_from_atomarray(
            atom_array, resolution=resolution, em_mode=em_mode, device=device
        )
    except Exception as e:
        logger.error(
            f"Failed to compute density for {row.filename} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    if save_structure:
        # Shift coordinates into the grid frame so the saved CIF aligns with
        # the CCP4 map. CCP4 format (unlike MRC) cannot encode an arbitrary Cartesian
        # origin, so we move the atoms instead. Possible the better way is to resample the map?
        atom_array.coord = atom_array.coord - xmap_torch.origin
        structure_output_path = structure_path.parent / f"{structure_path.stem}_density_input.cif"
        try:
            save_structure_to_cif(atom_array, structure_output_path)
            logger.info(f"Saved processed structure to {structure_output_path}")
        except Exception as e:
            logger.error(
                f"Failed to save structure for {row.filename} ({type(e).__name__}): {e}\n"
                f"{''.join(traceback.format_tb(e.__traceback__))}"
            )

    if row.mapfile:
        output_path = output_dir / row.mapfile
    else:
        output_path = output_dir / f"{structure_path.stem}_{resolution:.2f}A.ccp4"

    try:
        save_density(density, xmap_torch, output_path)
    except Exception as e:
        logger.error(
            f"Failed to save density for {row.filename} to {output_path} "
            f"({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return


def process_batch(
    csv_path: Path,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    occupancy_mode: str,
    em_mode: bool,
    device: torch.device,
    n_jobs: int = -1,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
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
        Directory where output density maps will be written
    resolution
        Map resolution in Angstroms
    em_mode
        If True, use electron scattering factors. If False, use X-ray factors.
    device
        PyTorch device for computation
    n_jobs
        Number of parallel jobs. -1 means use all available CPUs.
    strip_hydrogens
        If True, remove hydrogen atoms before computing density.
    strip_waters
        If True, remove water molecules before computing density.
    strip_ligands
        If True, remove ligand molecules (non-water heteroatoms) before computing density.
    save_structure
        If True, save the processed structure to a CIF file in the input directory.
    """
    from joblib import delayed, Parallel

    rows = load_batch_csv(csv_path)
    effective_n_jobs = resolve_parallel_jobs(device, n_jobs)
    logger.info(f"Processing {len(rows)} structures from {csv_path} using {effective_n_jobs} jobs")

    Parallel(n_jobs=effective_n_jobs, backend="loky")(
        delayed(_process_single_row)(
            row=row,
            occupancy_mode=occupancy_mode,
            base_dir=base_dir,
            output_dir=output_dir,
            resolution=resolution,
            em_mode=em_mode,
            device=device,
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            save_structure=save_structure,
        )
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic electron density maps from atomic structures"
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
        help="Base directory for relative paths in CSV",
    )

    selection_group = parser.add_argument_group("Selection Options")
    selection_group.add_argument(
        "--selection",
        type=str,
        help="Atom selection (e.g., 'chain A and resi 10-50' or 'chain A and resi 10')",
    )

    occ_group = parser.add_argument_group("Occupancy Options")
    occ_group.add_argument(
        "--occupancy-mode",
        "--occ-mode",
        dest="occupancy_mode",
        choices=["default", "uniform", "custom"],
        default="default",
        help="Occupancy assignment mode",
    )
    occ_group.add_argument(
        "--occupancy-values",
        "--occ-values",
        dest="occupancy_values",
        type=str,
        help="Colon-separated occupancy values for custom mode (e.g., '0.3:0.7')",
    )

    density_group = parser.add_argument_group("Density Options")
    density_group.add_argument(
        "--resolution", "-r", type=float, default=2.0, help="Map resolution in Angstroms"
    )
    density_group.add_argument(
        "--em-mode", action="store_true", help="Use electron scattering factors (EM mode)"
    )
    density_group.add_argument(
        "--remove-hydrogens",
        action="store_true",
        help="Remove hydrogen atoms before computing density",
    )
    density_group.add_argument(
        "--remove-waters",
        action="store_true",
        help="Remove water molecules before computing density",
    )
    density_group.add_argument(
        "--remove-ligands",
        action="store_true",
        help="Remove ligand molecules (non-water heteroatoms) before computing density",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save-structure",
        action="store_true",
        help="Save the processed structure (after selection, occupancy assignment) to CIF",
    )
    output_group.add_argument("--output", "-o", type=Path, help="Output CCP4 map file path")
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
            occupancy_mode=args.occupancy_mode,
            em_mode=args.em_mode,
            device=device,
            n_jobs=args.n_jobs,
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            save_structure=args.save_structure,
        )
    elif args.structure:
        row = BatchRow(
            filename=str(args.structure),
            selection=args.selection,
            occupancy_values=[float(v.strip()) for v in args.occupancy_values.split(":")]
            if args.occupancy_values
            else [],
            mapfile=args.output.name if args.output else None,
        )
        _process_single_row(
            row=row,
            occupancy_mode=args.occupancy_mode,
            base_dir=args.structure.parent,
            output_dir=args.output.parent if args.output else Path("."),
            resolution=args.resolution,
            em_mode=args.em_mode,
            device=device,
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            save_structure=args.save_structure,
        )
    else:
        logger.error("Please specify --structure or --batch-csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
