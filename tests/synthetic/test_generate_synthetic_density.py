"""Tests for synthetic density batch-row argument handling."""

import pytest
from sampleworks.synthetic.generate_synthetic_density import BatchRow


def test_batch_row_accepts_occupancy_values_column() -> None:
    """Batch CSV parsing uses the canonical occupancy_values column name."""
    row = BatchRow.from_dict(
        {"filename": "input.cif", "selection": "chain A", "occupancy_values": "0.25:0.75"}
    )

    assert row.occupancy_values == [0.25, 0.75]
    assert row.selection == "chain A"


def test_batch_row_accepts_legacy_occ_values_column() -> None:
    """The old occ_values column remains accepted for existing batch CSVs."""
    row = BatchRow.from_dict({"filename": "input.cif", "occ_values": "0.4:0.6"})

    assert row.occupancy_values == [0.4, 0.6]


def test_batch_row_rejects_occupancy_values_that_do_not_sum_to_one() -> None:
    """Density generation now uses the shared occupancy-value validation helper."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        BatchRow(filename="input.cif", occupancy_values=[0.2, 0.3])


def test_batch_row_rejects_unsupported_extension() -> None:
    """Only mmCIF and legacy PDB-like structure extensions are supported."""
    with pytest.raises(ValueError, match="Invalid file extension"):
        BatchRow(filename="input.txt")
