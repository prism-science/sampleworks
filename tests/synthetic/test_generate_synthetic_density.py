"""Tests for synthetic density batch-row argument handling."""

import pytest
from sampleworks.synthetic.generate_synthetic_density import BatchRow, main
from sampleworks.utils.atom_array_utils import load_structure_with_altlocs, save_structure_to_cif


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


def test_em_mode_with_it1992_exits_nonzero(monkeypatch, tmp_path) -> None:
    """The pair is rejected before any row runs, not swallowed per row with exit 0."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_synthetic_density",
            "--structure",
            str(tmp_path / "x.cif"),
            "--em-mode",
            "--scattering-table",
            "it1992",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1


def test_relative_structure_path_is_read_and_written_beside_itself(
    monkeypatch, tmp_path, resources_dir
) -> None:
    """A relative ``--structure`` is not joined onto its own directory a second time.

    The map lands at ``--output`` and the grid-aligned structure next to the input, both in
    the subdirectory the relative paths name.
    """
    deposited = load_structure_with_altlocs(
        resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    save_structure_to_cif(deposited[deposited.res_id <= 20], inputs / "fragment.cif")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_synthetic_density",
            "--structure",
            "inputs/fragment.cif",
            "--resolution",
            "3.0",
            "--save-structure",
            "--output",
            "inputs/fragment.ccp4",
        ],
    )

    main()

    assert (inputs / "fragment.ccp4").stat().st_size > 0
    assert (inputs / "fragment_density_input.cif").exists()
