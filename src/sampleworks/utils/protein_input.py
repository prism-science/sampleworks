import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sampleworks.core.losses.registry import (
    is_path_option,
    loss_option_names,
    option_from_text,
)
from sampleworks.utils.sequence import resolve_sequence_arg


# Columns describing the protein itself; every other column is a loss option.
_PROTEIN_COLUMNS = ("name", "structure", "sequence")


@dataclass
class ProteinInput:
    """
    Parse and validate protein input from a CSV file.

    Each CSV row maps to one instance. ``name`` and ``structure`` columns are
    required and ``sequence`` is optional. Every other column is named after a
    loss option and holds this protein's value for it: ``density`` and
    ``resolution`` for the real-space density loss, ``mtzfile`` for the
    structure-factor loss, ``bragg_target``/``diffuse_target`` for the diffuse
    loss, and so on. A column feeds every configured loss that has an option
    of that name, so one CSV can serve sweeps over different losses.

    Attributes
    ----------
    name : str
        Protein identifier; must be non-empty.
    structure : Path
        Structure file (``.cif``/``.pdb``). Relative CSV paths resolve against
        the CSV's directory.
    sequence : str or None
        Full one-letter protein sequence overriding the structure's, may be useful when
        termini or internal residues are unresolved. In the CSV this may be a
        raw sequence or a path to a single-record FASTA file. ``None`` (empty
        cell or missing column) keeps the sequence from the coordinates in the
        structure file.
    loss_options : dict[str, Any]
        Loss option name to this protein's value, typed as the option declares.
        Path options are absolute in CSV-read instances, and must exist. Empty
        cells are omitted, leaving the option to the run's loss configuration.
    """

    name: str
    structure: Path
    sequence: str | None = None
    loss_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.structure = Path(self.structure)
        if not self.name:
            raise ValueError("Protein name must not be empty.")
        if not self.structure.exists():
            raise FileNotFoundError(
                f"Structure file does not exist for protein '{self.name}': {self.structure}"
            )
        for option, value in self.loss_options.items():
            if is_path_option(option) and not Path(value).exists():
                raise FileNotFoundError(
                    f"The {option} file does not exist for protein '{self.name}': {value}"
                )

        if self.sequence is not None and not isinstance(self.sequence, str):
            raise ValueError(f"Sequence must be a string or None, got {type(self.sequence)}")

    @classmethod
    def from_csv(cls, csv_path: Path) -> list["ProteinInput"]:
        """Read one ``ProteinInput`` per row of a proteins CSV.

        Parameters
        ----------
        csv_path : Path
            CSV with ``name`` and ``structure`` columns, an optional ``sequence``
            column, and any number of loss-option columns.

        Returns
        -------
        list[ProteinInput]
            One entry per row, in file order.

        Raises
        ------
        FileNotFoundError
            If the CSV, or a file a row names, does not exist.
        ValueError
            If a required column is missing, a column names no loss option, or a
            cell is not a valid value for its option.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        csv_dir = csv_path.parent

        required_columns = {"name", "structure"}
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

            option_columns = [c for c in reader.fieldnames if c not in _PROTEIN_COLUMNS]
            unknown_columns = sorted(set(option_columns) - set(loss_option_names()))
            if unknown_columns:
                raise ValueError(
                    f"CSV file {csv_path} has column(s) {unknown_columns} that are neither "
                    f"protein columns {list(_PROTEIN_COLUMNS)} nor loss options "
                    f"{loss_option_names()}."
                )

            for row_idx, row in enumerate(reader, start=2):
                name = (row.get("name") or "").strip()
                structure_raw = (row.get("structure") or "").strip()
                sequence_raw = (row.get("sequence") or "").strip()

                structure = Path(structure_raw)
                if not structure.is_absolute():
                    structure = csv_dir / structure

                if not name:
                    raise ValueError(f"Row {row_idx}: Protein name must not be empty.")

                loss_options = {}
                for column in option_columns:
                    text = (row.get(column) or "").strip()
                    if not text:
                        continue
                    if is_path_option(column):
                        path = Path(text)
                        loss_options[column] = str(path if path.is_absolute() else csv_dir / path)
                        continue
                    try:
                        loss_options[column] = option_from_text(column, text)
                    except ValueError as err:
                        raise ValueError(
                            f"Row {row_idx}: invalid {column} '{text}' for protein '{name}'"
                        ) from err

                sequence = resolve_sequence_arg(sequence_raw, csv_dir)

                protein_inputs.append(
                    cls(
                        name=name,
                        structure=structure,
                        sequence=sequence,
                        loss_options=loss_options,
                    )
                )

        return protein_inputs
