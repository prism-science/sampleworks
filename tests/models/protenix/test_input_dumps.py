"""Protenix featurization never reuses another structure's input dump (#412)."""

import json
from pathlib import Path

import pytest


pytest.importorskip("protenix", reason="Protenix not installed")

from sampleworks.models.protenix.wrapper import annotate_structure_for_protenix
from sampleworks.utils.atom_reconciler import AtomReconciler
from sampleworks.utils.structure_utils import get_valid_atom_mask


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
    with input_json.open() as f:
        (sample,) = json.load(f)
    return [
        entry["proteinChain"]["sequence"]
        for entry in sample["sequences"]
        if "proteinChain" in entry
    ]


@pytest.mark.slow
def test_featurize_does_not_reuse_another_structures_msa_input(
    protenix_wrapper, structure_6b8x: dict, structure_5sop: dict, tmp_path: Path
):
    """Featurizing two structures into one out_dir yields each structure's own atoms."""
    out_dir = tmp_path / "protenix_input"
    for structure in (structure_6b8x, structure_5sop):
        annotated = annotate_structure_for_protenix(structure, out_dir=out_dir, use_msa=True)
        conditioning = protenix_wrapper.featurize(annotated).conditioning

        add_msa_json = out_dir / "protenix_input-add-msa.json"
        assert _protein_sequences(add_msa_json) == _protein_sequences(
            out_dir / "protenix_input.json"
        )

        # A reused dump models the other structure's sequences, so it would leave most
        # resolved input atoms without a model counterpart.
        input_atoms = conditioning.true_atom_array
        resolved_input_atoms = input_atoms[get_valid_atom_mask(input_atoms)]
        reconciler = AtomReconciler.from_arrays(conditioning.model_atom_array, resolved_input_atoms)
        assert reconciler.n_common == len(resolved_input_atoms)
