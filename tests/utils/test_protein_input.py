"""Tests for the grid-search ``proteins.csv`` reader."""

from pathlib import Path

import pytest
from sampleworks.utils.protein_input import ProteinInput


HEADER = "name,structure,density,resolution"


@pytest.fixture
def inputs_dir(tmp_path: Path) -> Path:
    """A directory holding the files a CSV row can point at."""
    (tmp_path / "s.cif").write_text("")
    (tmp_path / "d.ccp4").write_text("")
    (tmp_path / "chains.fasta").write_text(">A\nGATTACA\n")
    return tmp_path


def write_csv(directory: Path, header: str, row: str) -> Path:
    path = directory / "proteins.csv"
    path.write_text(f"{header}\n{row}\n")
    return path


class TestSequencesColumn:
    def test_is_optional(self, inputs_dir):
        csv_path = write_csv(inputs_dir, HEADER, "1ABC,s.cif,d.ccp4,1.8")

        (protein,) = ProteinInput.from_csv(csv_path)

        assert protein.sequences is None

    def test_resolves_a_relative_path_against_the_csv(self, inputs_dir):
        csv_path = write_csv(
            inputs_dir, f"{HEADER},sequences", "1ABC,s.cif,d.ccp4,1.8,chains.fasta"
        )

        (protein,) = ProteinInput.from_csv(csv_path)

        assert protein.sequences == inputs_dir / "chains.fasta"

    def test_keeps_an_absolute_path(self, inputs_dir):
        absolute = inputs_dir / "chains.fasta"
        csv_path = write_csv(inputs_dir, f"{HEADER},sequences", f"1ABC,s.cif,d.ccp4,1.8,{absolute}")

        (protein,) = ProteinInput.from_csv(csv_path)

        assert protein.sequences == absolute

    def test_an_empty_cell_means_no_sequence(self, inputs_dir):
        csv_path = write_csv(inputs_dir, f"{HEADER},sequences", "1ABC,s.cif,d.ccp4,1.8,")

        (protein,) = ProteinInput.from_csv(csv_path)

        assert protein.sequences is None

    def test_rejects_a_path_that_does_not_exist(self, inputs_dir):
        csv_path = write_csv(
            inputs_dir, f"{HEADER},sequences", "1ABC,s.cif,d.ccp4,1.8,absent.fasta"
        )

        with pytest.raises(FileNotFoundError, match="Sequence file does not exist"):
            ProteinInput.from_csv(csv_path)

    def test_reports_an_empty_name_before_checking_the_sequence(self, inputs_dir):
        csv_path = write_csv(inputs_dir, f"{HEADER},sequences", ",s.cif,d.ccp4,1.8,absent.fasta")

        with pytest.raises(ValueError, match="Protein name must not be empty"):
            ProteinInput.from_csv(csv_path)
