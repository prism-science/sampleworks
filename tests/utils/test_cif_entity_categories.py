"""Tests for carrying deposited polymer entity categories into output CIFs."""

from pathlib import Path

import numpy as np
import pytest
import torch
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArrayStack, stack
from biotite.structure.io.pdbx import set_structure
from biotite.structure.io.pdbx.cif import CIFCategory, CIFFile
from sampleworks.utils.cif_utils import carry_polymer_entity_categories
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import save_everything


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


def test_carry_polymer_entity_categories_ignores_author_numbering(resources_dir):
    """Align using label sequence despite unrelated author numbers and insertion codes."""
    output = _multi_model_cif(
        resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    )
    atom_site = output.block["atom_site"]
    atom_site["auth_seq_id"] = np.full(atom_site.row_count, "500")
    atom_site["pdbx_PDB_ins_code"] = np.full(atom_site.row_count, "A")

    carry_polymer_entity_categories(output, resources_dir / "1vme" / "1vme_final.cif")

    carried = output.block["entity_poly_seq"]
    modeled_numbers = sorted(set(atom_site["label_seq_id"].as_array(str)), key=int)
    assert carried["num"].as_array(str).tolist() == modeled_numbers
    modeled_sequence = output.block["entity_poly"]["pdbx_seq_one_letter_code"].as_item()
    assert len(modeled_sequence) == carried.row_count


def test_carry_polymer_entity_categories_rejects_ambiguous_alignment(resources_dir):
    """Reject two possible ordered mappings without mutating the output."""
    output = _multi_model_cif(
        resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    )
    reference = CIFFile.read(str(resources_dir / "1vme" / "1vme_final.cif"))
    atom_site = output.block["atom_site"]
    modeled = {}
    for number, name in zip(
        atom_site["label_seq_id"].as_array(str),
        atom_site["label_comp_id"].as_array(str),
        strict=True,
    ):
        modeled[int(number)] = str(name)
    residue_names = [modeled[number] for number in sorted(modeled)]
    entity_id = reference.block["entity_poly"]["entity_id"].as_item()
    reference.block["entity_poly_seq"] = CIFCategory(
        {
            "entity_id": [entity_id] * (2 * len(residue_names)),
            "num": [str(number) for number in range(1, 2 * len(residue_names) + 1)],
            "mon_id": residue_names * 2,
            "hetero": ["n"] * (2 * len(residue_names)),
        }
    )

    with pytest.raises(ValueError, match="reference entity sequence match"):
        carry_polymer_entity_categories(output, reference)

    for category in ("entity", "entity_poly", "entity_poly_seq", "struct_asym"):
        assert category not in output.block


def test_carry_polymer_entity_categories_normalizes_selenomethionine(resources_dir):
    """Accept deposited MSE only when the modeled residue is canonical MET."""
    output = _multi_model_cif(
        resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    )
    atom_site = output.block["atom_site"]
    names = atom_site["label_comp_id"].as_array(str)
    atom_site["label_comp_id"] = np.where(names == "MSE", "MET", names)

    carry_polymer_entity_categories(output, resources_dir / "1vme" / "1vme_final.cif")

    carried_names = output.block["entity_poly_seq"]["mon_id"].as_array(str)
    assert "MSE" not in carried_names
    assert "MET" in carried_names


def test_carry_polymer_entity_categories_copies_deposited_chem_comp_rows(resources_dir):
    """Carry complete deposited chemical-component rows required by modeled residues."""
    output = _multi_model_cif(
        resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    )
    reference = CIFFile.read(str(resources_dir / "1vme" / "1vme_final.cif"))
    component_ids = sorted(set(output.block["atom_site"]["label_comp_id"].as_array(str)))
    reference.block["chem_comp"] = CIFCategory(
        {
            "id": component_ids,
            "type": ["L-peptide linking"] * len(component_ids),
            "name": [f"DEPOSITED {component_id}" for component_id in component_ids],
        }
    )

    categories = carry_polymer_entity_categories(output, reference)

    assert categories[-1] == "chem_comp"
    carried = output.block["chem_comp"]
    assert set(carried["id"].as_array(str)) == set(component_ids)
    assert all(name.startswith("DEPOSITED ") for name in carried["name"].as_array(str))


def test_carry_polymer_entity_categories_keeps_polymer_beside_non_polymer(resources_dir):
    """A ligand or water entity does not stop the polymer entity from being carried."""
    output = _multi_model_cif(
        resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    )
    atom_site = output.block["atom_site"]
    entity_ids = atom_site["label_entity_id"].as_array(str)
    polymer_id = entity_ids[0]
    # Re-label the last row of each model as a water in its own entity and chain.
    rows_per_model = atom_site.row_count // 2
    water_rows = [rows_per_model - 1, atom_site.row_count - 1]
    for column, value in (
        ("label_entity_id", "9"),
        ("label_asym_id", "Z"),
        ("label_seq_id", "."),
        ("label_comp_id", "HOH"),
    ):
        values = atom_site[column].as_array(str)
        values[water_rows] = value
        atom_site[column] = values
    reference = CIFFile.read(str(resources_dir / "1vme" / "1vme_final.cif"))

    carry_polymer_entity_categories(output, reference)

    assert output.block["entity"]["id"].as_array(str).tolist() == [polymer_id]
    struct_asym = output.block["struct_asym"]
    assert "Z" not in struct_asym["id"].as_array(str)
    assert set(struct_asym["entity_id"].as_array(str)) == {polymer_id}


def test_carry_polymer_entity_categories_reports_unconvertible_monomers(resources_dir):
    """A monomer without a one-letter code fails as ValueError, not KeyError."""
    output = _multi_model_cif(
        resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    )
    atom_site = output.block["atom_site"]
    names = atom_site["label_comp_id"].as_array(str)
    substituted = names[0]
    atom_site["label_comp_id"] = np.where(names == substituted, "SEP", names)
    # Relabel the same monomer in the deposit, so the sequence still matches and the only
    # possible failure is the one-letter conversion.
    reference = CIFFile.read(str(resources_dir / "1vme" / "1vme_final.cif"))
    sequence = reference.block["entity_poly_seq"]
    reference_names = sequence["mon_id"].as_array(str)
    sequence["mon_id"] = np.where(reference_names == substituted, "SEP", reference_names)

    with pytest.raises(ValueError, match="no one-letter representation"):
        carry_polymer_entity_categories(output, reference)

    assert "entity_poly_seq" not in output.block


def test_save_everything_writes_cifs_when_entity_carry_fails(
    resources_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A failed entity carry warns but never costs a completed run its coordinates."""
    structure_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    atom_array = load_any(
        structure_path,
        altloc="first",
        extra_fields=["occupancy", "b_factor", "atom_id"],
    )
    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    coords = torch.from_numpy(np.stack([atom_array.coord, atom_array.coord])).float()
    # A reference whose sequence cannot match the output, so the carry fails.
    reference = CIFFile.read(str(resources_dir / "1vme" / "1vme_final.cif"))
    sequence = reference.block["entity_poly_seq"]
    sequence["mon_id"] = np.full(sequence.row_count, "ZZZ")

    args = GuidanceConfig(
        protein="1vme_0.5occA_0.5occB",
        structure=structure_path,
        density=Path("dummy"),
        model_name="boltz2",
        guidance_type="pure_guidance",
        log_path="dummy",
        output_dir=str(tmp_path),
    )
    save_everything(
        args,
        losses=[],
        refined_structure={"asym_unit": atom_array},
        traj_denoised=[coords],
        traj_next_step=[coords],
        scaler_type="pure_guidance",
        final_state=coords,
        reference_cif=reference,
    )

    assert "without polymer entity categories" in caplog.text
    for output_path in (
        tmp_path / "refined.cif",
        tmp_path / "trajectory" / "denoised" / "trajectory_0.cif",
        tmp_path / "trajectory" / "next_step" / "trajectory_0.cif",
    ):
        block = CIFFile.read(str(output_path)).block
        atom_ids = block["atom_site"]["id"].as_array(int)
        assert np.array_equal(atom_ids, np.arange(1, len(atom_ids) + 1))
        assert "entity_poly_seq" not in block
