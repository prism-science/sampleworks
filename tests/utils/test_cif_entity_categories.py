"""Tests for carrying deposited polymer entity categories into output CIFs."""

from pathlib import Path

import numpy as np
import pytest
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArrayStack, stack
from biotite.structure.io.pdbx import set_structure
from biotite.structure.io.pdbx.cif import CIFFile
from sampleworks.utils.cif_utils import carry_polymer_entity_categories


def _multi_model_cif(structure_path: Path) -> CIFFile:
    """Build a two-model CIF from a structure path.

    Parameters
    ----------
    structure_path : Path
        Input structure path.

    Returns
    -------
    CIFFile
        Two-model CIF containing only structure categories.
    """
    atom_array = load_any(
        structure_path,
        altloc="first",
        extra_fields=["occupancy", "b_factor", "atom_id"],
    )
    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    output = CIFFile()
    set_structure(output, stack([atom_array, atom_array]))
    return output


def test_carry_polymer_entity_categories_aligns_multimodel_output(resources_dir, tmp_path):
    """Carried categories align with every modeled residue after a round trip."""
    output = _multi_model_cif(
        resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    )
    reference = resources_dir / "1vme" / "1vme_final.cif"

    categories = carry_polymer_entity_categories(output, reference)
    output_path = tmp_path / "carried.cif"
    output.write(str(output_path))
    reloaded = CIFFile.read(str(output_path)).block

    assert categories == ("entity", "entity_poly", "entity_poly_seq", "struct_asym")
    atom_site = reloaded["atom_site"]
    assert set(atom_site["pdbx_PDB_model_num"].as_array(str)) == {"1", "2"}
    assert set(atom_site["label_entity_id"].as_array(str)) == set(
        reloaded["entity"]["id"].as_array(str)
    )
    assert set(reloaded["entity"]["type"].as_array(str)) == {"polymer"}

    sequence = reloaded["entity_poly_seq"]
    carried = set(
        zip(
            sequence["entity_id"].as_array(str),
            sequence["num"].as_array(str),
            sequence["mon_id"].as_array(str),
            strict=True,
        )
    )
    modeled = set(
        zip(
            atom_site["label_entity_id"].as_array(str),
            atom_site["label_seq_id"].as_array(str),
            atom_site["label_comp_id"].as_array(str),
            strict=True,
        )
    )
    assert modeled <= carried


def test_carry_polymer_entity_categories_does_not_mutate_on_failed_match(resources_dir):
    """An unmatched output sequence leaves entity categories absent."""
    output = _multi_model_cif(
        resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    )
    output.block["atom_site"]["label_comp_id"] = np.full(output.block["atom_site"].row_count, "ZZZ")

    with pytest.raises(ValueError, match="reference entity sequence match"):
        carry_polymer_entity_categories(output, resources_dir / "1vme" / "1vme_final.cif")

    for category in ("entity", "entity_poly", "entity_poly_seq", "struct_asym"):
        assert category not in output.block
