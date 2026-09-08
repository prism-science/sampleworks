import csv
import math
from dataclasses import dataclass
from pathlib import Path

from sampleworks.utils.sequence import resolve_sequence_arg


@dataclass
class ProteinInput:
    """
    Parse and validate protein input from a CSV file.
    """

    name: str
    structure: Path
    density: Path
    resolution: float
    sequence: str | None = None

    def __post_init__(self):
        self.structure = Path(self.structure)
        self.density = Path(self.density)
        if not self.name:
            raise ValueError("Protein name must not be empty.")
        if not self.structure.exists():
            raise FileNotFoundError(
                f"Structure file does not exist for protein '{self.name}': {self.structure}"
            )
        if not self.density.exists():
            raise FileNotFoundError(
                f"Density file does not exist for protein '{self.name}': {self.density}"
            )
        if not math.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError(
                f"Resolution must be a positive finite number for protein '{self.name}', "
                f"got {self.resolution}."
            )

        if self.sequence is not None and not isinstance(self.sequence, str):
            raise ValueError(f"Sequence must be a string or None, got {type(self.sequence)}")

    @classmethod
    def from_csv(cls, csv_path: Path) -> list["ProteinInput"]:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        csv_dir = csv_path.parent

        required_columns = {"name", "structure", "density", "resolution"}
        protein_inputs = []
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            if reader.fieldnames is None:
                raise ValueError(f"CSV file {csv_path} appears to be empty")

            missing_columns = required_columns - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    f"CSV file missing required columns: {missing_columns}. "
                    f"Found columns: {reader.fieldnames}"
                )

            for row_idx, row in enumerate(reader, start=2):
                name = (row.get("name") or "").strip()
                structure_raw = (row.get("structure") or "").strip()
                density_raw = (row.get("density") or "").strip()
                resolution_raw = (row.get("resolution") or "").strip()
                sequence_raw = (row.get("sequence") or "").strip()

                structure = Path(structure_raw)
                if not structure.is_absolute():
                    structure = csv_dir / structure

                density = Path(density_raw)
                if not density.is_absolute():
                    density = csv_dir / density

                try:
                    resolution = float(resolution_raw)
                except ValueError as err:
                    raise ValueError(
                        f"Row {row_idx}: invalid resolution '{resolution_raw}' for protein '{name}'"
                    ) from err

                sequence = resolve_sequence_arg(sequence_raw, csv_dir)

                protein_inputs.append(
                    cls(
                        name=name,
                        structure=structure,
                        density=density,
                        resolution=resolution,
                        sequence=sequence,
                    )
                )

        return protein_inputs
