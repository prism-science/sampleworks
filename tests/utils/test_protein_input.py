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
    (tmp_path / "x.mtz").write_text("")
    (tmp_path / "chains.fasta").write_text(">A\nACDEFGHIKLMNPQRSTVWY\n")
    return tmp_path


def write_csv(directory: Path, header: str, row: str) -> Path:
    path = directory / "proteins.csv"
    path.write_text(f"{header}\n{row}\n")
    return path


class TestSequenceColumn:
    def test_is_optional(self, inputs_dir):
        csv_path = write_csv(inputs_dir, HEADER, "1ABC,s.cif,d.ccp4,1.8")

        (protein,) = ProteinInput.from_csv(csv_path)

        assert protein.sequence is None

    def test_resolves_fasta_to_sequence_string(self, inputs_dir):
        csv_path = write_csv(inputs_dir, f"{HEADER},sequence", "1ABC,s.cif,d.ccp4,1.8,chains.fasta")

        (protein,) = ProteinInput.from_csv(csv_path)

        assert protein.sequence == "ACDEFGHIKLMNPQRSTVWY"

    def test_resolves_absolute_fasta_path(self, inputs_dir):
        absolute = inputs_dir / "chains.fasta"
        csv_path = write_csv(inputs_dir, f"{HEADER},sequence", f"1ABC,s.cif,d.ccp4,1.8,{absolute}")

        (protein,) = ProteinInput.from_csv(csv_path)

        assert protein.sequence == "ACDEFGHIKLMNPQRSTVWY"

    def test_an_empty_cell_means_no_sequence(self, inputs_dir):
        csv_path = write_csv(inputs_dir, f"{HEADER},sequence", "1ABC,s.cif,d.ccp4,1.8,")

        (protein,) = ProteinInput.from_csv(csv_path)

        assert protein.sequence is None

    def test_rejects_a_nonexistent_path_as_invalid_sequence(self, inputs_dir):
        csv_path = write_csv(inputs_dir, f"{HEADER},sequence", "1ABC,s.cif,d.ccp4,1.8,absent.fasta")

        with pytest.raises(ValueError, match="Invalid protein sequence"):
            ProteinInput.from_csv(csv_path)

    def test_reports_an_empty_name_before_checking_the_sequence(self, inputs_dir):
        """Report the missing required name before an invalid optional sequence."""
        csv_path = write_csv(inputs_dir, f"{HEADER},sequence", ",s.cif,d.ccp4,1.8,absent.fasta")

        with pytest.raises(ValueError, match="Protein name must not be empty"):
            ProteinInput.from_csv(csv_path)


class TestRewardOptionColumns:
    def test_density_columns_become_typed_reward_options(self, inputs_dir):
        csv_path = write_csv(inputs_dir, HEADER, "1ABC,s.cif,d.ccp4,1.8")

        (protein,) = ProteinInput.from_csv(csv_path)

        assert protein.reward_options == {"density": str(inputs_dir / "d.ccp4"), "resolution": 1.8}

    def test_a_structure_factor_row_needs_no_density_column(self, inputs_dir):
        csv_path = write_csv(
            inputs_dir, "name,structure,mtzfile,bulk_solvent", "1ABC,s.cif,x.mtz,combined"
        )

        (protein,) = ProteinInput.from_csv(csv_path)

        assert protein.reward_options == {
            "mtzfile": str(inputs_dir / "x.mtz"),
            "bulk_solvent": "combined",
        }

    def test_an_empty_cell_leaves_the_option_unset(self, inputs_dir):
        csv_path = write_csv(inputs_dir, "name,structure,mtzfile,resolution", "1ABC,s.cif,x.mtz,")

        (protein,) = ProteinInput.from_csv(csv_path)

        assert "resolution" not in protein.reward_options

    def test_rejects_a_column_that_is_no_reward_option(self, inputs_dir):
        csv_path = write_csv(inputs_dir, "name,structure,mtz_file", "1ABC,s.cif,x.mtz")

        with pytest.raises(ValueError, match=r"column\(s\) \['mtz_file'\]"):
            ProteinInput.from_csv(csv_path)

    def test_rejects_a_missing_data_file(self, inputs_dir):
        csv_path = write_csv(inputs_dir, "name,structure,mtzfile", "1ABC,s.cif,absent.mtz")

        with pytest.raises(FileNotFoundError, match="mtzfile file does not exist"):
            ProteinInput.from_csv(csv_path)

    def test_reports_the_row_of_an_unparseable_value(self, inputs_dir):
        csv_path = write_csv(inputs_dir, HEADER, "1ABC,s.cif,d.ccp4,high")

        with pytest.raises(ValueError, match="Row 2: invalid resolution 'high'"):
            ProteinInput.from_csv(csv_path)
