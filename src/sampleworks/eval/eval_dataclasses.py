import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.eval.occupancy_utils import occupancy_to_str


@dataclass
class Trial:
    protein: str
    altloc_occupancies: dict[str, float]
    model: str
    method: str | None
    scaler: str
    ensemble_size: int
    guidance_weight: float | None
    gd_steps: int | None
    trial_dir: Path
    refined_cif_path: Path
    protein_dir_name: str
    input_structure_path: Path | None = None
    density_path: Path | None = None
    resolution: float | None = None
    rscc: float = np.nan  # these last three are placeholders for RSCC calculations.
    base_map_path: Path | None = None
    error: Exception | None = None

    @property
    def occ_key(self) -> tuple[tuple[str, float], ...]:
        """Hashable key for caches. Zero-occupancy altlocs are omitted for
        consistency with occupancy_to_str / extract_protein_and_occupancy."""
        return tuple(sorted((k, v) for k, v in self.altloc_occupancies.items() if abs(v) > 1e-6))


class TrialList(list[Trial]):
    def summarize(self):
        logger.info(f"Proteins: {set(trial.protein for trial in self)}")
        logger.info(f"Models: {set(trial.model for trial in self)}")
        logger.info(f"Scalers: {set(trial.scaler for trial in self)}")


@dataclass
class ProteinConfig:
    """
    Configuration metadata for a set of protein maps and structures
    associated with some PDB id.
    """

    protein: str
    base_map_dir: Path
    selection: list[str]
    resolution: float
    map_pattern: str
    structure_pattern: str = ""

    def __post_init__(self):
        # TODO validate structure patterns? Anything else we should check to avoid later errors?
        self.base_map_dir = Path(self.base_map_dir)  # just in case someone passes a string
        self.selection = [s.strip() for s in self.selection if self.is_selection_valid(s)]
        if not self.selection:
            raise ValueError(f"No valid selection provided for protein {self.protein}")

    def is_selection_valid(self, selection: str) -> bool:
        return selection is not None and selection.strip() != ""

    def get_base_map_path_for_occupancy(self, altloc_occupancies: dict[str, float]) -> Path | None:
        """Return the base-map path for the given altloc occupancies, or ``None``.

        Parameters
        ----------
        altloc_occupancies : dict[str, float]
            Mapping of altloc labels to occupancy values,
            e.g. ``{"A": 0.5, "B": 0.5}`` or ``{"A": 0.5, "B": 0.3, "C": 0.2}``.
        """
        try:
            occ_str = occupancy_to_str(**altloc_occupancies)
        except ValueError as e:
            logger.warning(
                f"Cannot determine occupancy string for {self.protein} with occupancies"
                f" {altloc_occupancies}: {e}"
            )
            return None
        map_path = self.base_map_dir / self.map_pattern.format(occ_str=occ_str)
        if map_path.exists():
            return map_path

        logger.warning(f"Base map for protein {self.protein} ({map_path}) NOT FOUND")
        return None

    def load_map(
        self,
        map_path: Path,
        canonical_unit_cell=True,
        selection_coords=None,
        padding=0.0,
        resolution: float | None = None,
    ) -> XMap | None:
        """Load a map using recorded trial resolution when available."""
        xmap = XMap.fromfile(str(map_path), resolution=resolution or self.resolution)
        if canonical_unit_cell:
            xmap = xmap.canonical_unit_cell()
        if selection_coords is not None:
            xmap = xmap.extract(selection_coords, padding=padding)

        return xmap

    def get_reference_structure_path(self, altloc_occupancies: dict[str, float]) -> Path | None:
        """Return the reference-structure path for the given altloc occupancies, or ``None``.

        Parameters
        ----------
        altloc_occupancies : dict[str, float]
            Mapping of altloc labels to occupancy values,
            e.g. ``{"A": 0.5, "B": 0.5}`` or ``{"A": 0.5, "B": 0.3, "C": 0.2}``.
        """
        if not self.structure_pattern:
            return None

        try:
            occ_str = occupancy_to_str(**altloc_occupancies)
        except ValueError as e:
            logger.warning(
                f"Cannot determine occupancy string for {self.protein} with occupancies"
                f" {altloc_occupancies}: {e}"
            )
            return None
        structure_path = self.base_map_dir / self.structure_pattern.format(occ_str=occ_str)
        if structure_path.exists():
            return structure_path

        logger.warning(
            f"Reference structure for {self.protein} with occupancies {altloc_occupancies} "
            f"not found: {structure_path}"
        )
        return None

    @classmethod
    def from_csv(cls, workspace_root: Path, csv_path: Path) -> dict[str, "ProteinConfig"]:
        """Load protein configurations from a CSV file.

        Parameters
        ----------
        workspace_root : Path
            Root directory for resolving relative paths in base_map_dir.
        csv_path : Path
            Path to the CSV file containing protein configurations.

        Returns
        -------
        dict[str, ProteinConfig]
            Dictionary mapping protein names to ProteinConfig objects.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist.
        ValueError
            If the CSV file is missing required columns or has invalid data.

        Notes
        -----
        Expected CSV format:
        - Required columns: protein, base_map_dir, selection, resolution, map_pattern
        - Optional columns: structure_pattern (defaults to empty string)
        - The base_map_dir should be relative to workspace_root or absolute.
        """
        csv_path = Path(csv_path)
        workspace_root = Path(workspace_root)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        if not workspace_root.exists():
            raise FileNotFoundError(f"Workspace root not found: {workspace_root}")

        protein_configs = {}
        required_columns = {"protein", "base_map_dir", "selection", "resolution", "map_pattern"}

        with open(csv_path) as f:
            reader = csv.DictReader(f)

            # Validate CSV headers
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {csv_path} appears to be empty")

            missing_columns = required_columns - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    f"CSV file missing required columns: {missing_columns}. "
                    f"Found columns: {reader.fieldnames}"
                )

            for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                try:
                    # Validate required fields are not empty
                    for col in required_columns:
                        if not row[col] or not row[col].strip():
                            raise ValueError(f"Row {row_num}: Required field '{col}' is empty")

                    protein = row["protein"].strip()

                    # Resolve base_map_dir relative to workspace_root
                    base_map_dir = Path(row["base_map_dir"].strip())
                    if not base_map_dir.is_absolute():
                        base_map_dir = workspace_root / base_map_dir

                    # Parse resolution as float
                    try:
                        resolution = float(row["resolution"])
                        if resolution <= 0:
                            raise ValueError("Resolution must be positive")
                    except ValueError as e:
                        raise ValueError(
                            f"Row {row_num}: Invalid resolution value '{row['resolution']}': {e}"
                        )

                    # Structure pattern is optional
                    structure_pattern = row.get("structure_pattern", "").strip()

                    # Create ProteinConfig object
                    config = cls(
                        protein=protein,
                        base_map_dir=base_map_dir,
                        selection=row["selection"].split(";"),
                        resolution=resolution,
                        map_pattern=row["map_pattern"].strip(),
                        structure_pattern=structure_pattern,
                    )

                    # Check for duplicate protein names
                    if protein in protein_configs:
                        logger.warning(
                            f"Row {row_num}: Duplicate protein name '{protein}'. "
                            "Overwriting previous entry."
                        )

                    protein_configs[protein] = config

                except Exception as e:
                    raise ValueError(f"Error processing row {row_num} in {csv_path}: {e}") from e

        if not protein_configs:
            raise ValueError(f"No valid protein configurations found in {csv_path}")

        logger.info(f"Loaded {len(protein_configs)} protein configurations from {csv_path}")
        return protein_configs
