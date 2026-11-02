"""Protenix featurization never reuses another structure's input dump (#412)."""

import json
from pathlib import Path

import pytest


pytest.importorskip("protenix", reason="Protenix not installed")

from sampleworks.models.protenix.wrapper import annotate_structure_for_protenix


def _protein_sequences(input_json: Path) -> list[str]:
    """Return the protein chain sequences stored in a Protenix input JSON.

    Parameters
    ----------
    input_json : Path
        Protenix input JSON file (a one-element list of sample dicts).

    Returns
    -------
    list[str]
        Sequence of every ``proteinChain`` entry, in file order.
    """
    (sample,) = json.loads(input_json.read_text())
    sequences = []
    for entry in sample["sequences"]:
        if "proteinChain" in entry:
            sequences.append(entry["proteinChain"]["sequence"])
    return sequences


@pytest.mark.slow
def test_featurize_does_not_reuse_another_structures_msa_input(
    protenix_wrapper, structure_6b8x: dict, structure_5sop: dict, tmp_path: Path
):
    """Featurizing two structures into one out_dir yields each structure's own atoms."""
    # Protenix keeps every heavy atom of both fixtures, so a reused dump shows up as the
    # other structure's count.
    structures = {"6b8x": structure_6b8x, "5sop": structure_5sop}
    expected_atom_counts = {
        name: structure["asym_unit"].array_length() for name, structure in structures.items()
    }
    assert expected_atom_counts["6b8x"] != expected_atom_counts["5sop"]

    out_dir = tmp_path / "protenix_input"
    atom_counts = {}
    for name, structure in structures.items():
        annotated = annotate_structure_for_protenix(structure, out_dir=out_dir, use_msa=True)
        features = protenix_wrapper.featurize(annotated)
        atom_counts[name] = len(features.conditioning.model_atom_array)

        add_msa_json = out_dir / "protenix_input-add-msa.json"
        assert _protein_sequences(add_msa_json) == _protein_sequences(
            out_dir / "protenix_input.json"
        )

    assert atom_counts == expected_atom_counts
