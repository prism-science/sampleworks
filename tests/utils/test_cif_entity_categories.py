"""Tests for carrying deposited polymer entity categories into output CIFs."""

from pathlib import Path

import numpy as np
import pytest
import torch
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack, stack
from biotite.structure.io.pdbx import set_structure
from biotite.structure.io.pdbx.cif import CIFBlock, CIFCategory, CIFFile
from sampleworks.utils.cif_utils import carry_polymer_entity_categories
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import save_everything


# Deposited entity 1 (author chain P, label chain A), author-numbered from -3. Position 10 is
# microheterogeneous (SER/SEP), and MSE/LYR are modified residues written as HETATM.
_SEQUENCE = ["MET", "GLY", "HIS", "HIS", "HIS", "HIS", "MSE", "PRO", "LYR", "SER"]
_AUTHOR_OFFSET = -4  # pdb_seq_num = seq_id + _AUTHOR_OFFSET
_HETERO = {"MSE", "LYR", "HOH"}


def _reference(chem_comp_ids: list[str] | None = None) -> CIFFile:
    """Build an RCSB-style deposit with a second, unmodeled entity (author Q, label B).

    Parameters
    ----------
    chem_comp_ids : list[str] | None
        Component ids to list in ``chem_comp``; omitted when ``None``.

    Returns
    -------
    CIFFile
        Single-block deposit with polymer categories and ``pdbx_poly_seq_scheme``.
    """
    rows = [("1", "A", "P", i + 1, name) for i, name in enumerate(_SEQUENCE)]
    rows.append(("1", "A", "P", len(_SEQUENCE), "SEP"))
    rows += [("2", "B", "Q", i + 1, "GLY") for i in range(2)]
    scheme = {
        "asym_id": [asym for _, asym, _, _, _ in rows],
        "entity_id": [entity for entity, _, _, _, _ in rows],
        "seq_id": [str(seq) for _, _, _, seq, _ in rows],
        "mon_id": [name for _, _, _, _, name in rows],
        "pdb_seq_num": [str(seq + _AUTHOR_OFFSET) for _, _, _, seq, _ in rows],
        "auth_seq_num": [str(seq + _AUTHOR_OFFSET) for _, _, _, seq, _ in rows],
        "auth_mon_id": [name for _, _, _, _, name in rows],
        "pdb_strand_id": [strand for _, _, strand, _, _ in rows],
        "pdb_ins_code": ["."] * len(rows),
    }
    categories = {
        "entity": {"id": ["1", "2"], "type": ["polymer", "polymer"]},
        "entity_poly": {"entity_id": ["1", "2"], "pdbx_strand_id": ["P", "Q"]},
        "entity_poly_seq": {
            "entity_id": scheme["entity_id"],
            "num": scheme["seq_id"],
            "mon_id": scheme["mon_id"],
        },
        "struct_asym": {"id": ["A", "B"], "entity_id": ["1", "2"]},
        "pdbx_poly_seq_scheme": scheme,
    }
    if chem_comp_ids is not None:
        categories["chem_comp"] = {
            "id": chem_comp_ids,
            "name": [f"DEPOSITED {component_id}" for component_id in chem_comp_ids],
        }
    reference = CIFFile()
    reference["deposit"] = CIFBlock(
        {name: CIFCategory(columns) for name, columns in categories.items()}
    )
    return reference


def _output(positions: list[int], extra: tuple[tuple[str, int, str], ...] = ()) -> CIFFile:
    """Write a two-model, one-CA-per-residue output in the deposit's author numbering.

    Parameters
    ----------
    positions : list[int]
        0-based positions in ``_SEQUENCE`` that the output models.
    extra : tuple[tuple[str, int, str], ...]
        Additional ``(chain, author number, name)`` residues, e.g. waters.

    Returns
    -------
    CIFFile
        Output as written by ``set_structure``: label ids hold author values.
    """
    residues = [("P", p + 1 + _AUTHOR_OFFSET, _SEQUENCE[p]) for p in positions] + list(extra)
    atom_array = AtomArray(len(residues))
    atom_array.chain_id[:] = [chain for chain, _, _ in residues]
    atom_array.res_id[:] = [number for _, number, _ in residues]
    atom_array.res_name[:] = [name for _, _, name in residues]
    atom_array.hetero[:] = [name in _HETERO for _, _, name in residues]
    atom_array.atom_name[:] = "CA"
    atom_array.element[:] = "C"
    output = CIFFile()
    set_structure(output, stack([atom_array, atom_array]))
    return output


def _column(output: CIFFile, category: str, column: str) -> list[str]:
    """Return one output column as strings."""
    return output.block[category][column].as_array(str).tolist()


def test_carry_takes_label_ids_from_deposited_scheme():
    """Author P/-2..6 maps to label A/2..10, modified residues included, across both models."""
    positions = list(range(1, len(_SEQUENCE)))  # MET -3 is unmodeled
    output = _output(positions)

    categories = carry_polymer_entity_categories(output, _reference())

    assert categories == (
        "entity",
        "entity_poly",
        "entity_poly_seq",
        "struct_asym",
        "pdbx_poly_seq_scheme",
    )
    assert set(_column(output, "atom_site", "auth_asym_id")) == {"P"}
    assert set(_column(output, "atom_site", "label_asym_id")) == {"A"}
    assert set(_column(output, "atom_site", "label_entity_id")) == {"1"}
    assert _column(output, "atom_site", "label_seq_id") == [str(p + 1) for p in positions] * 2
    assert _column(output, "entity", "id") == ["1"]
    assert _column(output, "struct_asym", "id") == ["A"]

    scheme_auth = dict(
        zip(
            zip(
                _column(output, "pdbx_poly_seq_scheme", "asym_id"),
                _column(output, "pdbx_poly_seq_scheme", "seq_id"),
            ),
            _column(output, "pdbx_poly_seq_scheme", "auth_seq_num"),
        )
    )
    atom_rows = zip(
        _column(output, "atom_site", "label_asym_id"),
        _column(output, "atom_site", "label_seq_id"),
        _column(output, "atom_site", "auth_seq_id"),
    )
    assert all(scheme_auth[(asym, seq)] == auth for asym, seq, auth in atom_rows)
    assert scheme_auth[("A", "1")] == "?"


def test_carry_maps_gap_beside_repeat_by_number():
    """A HIS missing from the tag is ambiguous by sequence but exact by author number."""
    positions = [2, 3, 5, 6]  # HIS HIS _ HIS MSE
    output = _output(positions)

    carry_polymer_entity_categories(output, _reference())

    assert _column(output, "atom_site", "label_seq_id") == ["3", "4", "6", "7"] * 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("auth_seq_id", "99"), "no deposited pdbx_poly_seq_scheme row"),
        (("label_comp_id", "TRP"), "disagree with the deposit"),
    ],
)
def test_carry_does_not_mutate_on_failed_match(mutation, message):
    """An unknown author number or a renamed residue fails before anything is written."""
    output = _output([1, 2, 3])
    atom_site = output.block["atom_site"]
    column, value = mutation
    values = atom_site[column].as_array(str)
    values[0] = value
    atom_site[column] = values
    label_seq_ids = _column(output, "atom_site", "label_seq_id")

    with pytest.raises(ValueError, match=message):
        carry_polymer_entity_categories(output, _reference())

    assert _column(output, "atom_site", "label_seq_id") == label_seq_ids
    for category in ("entity", "entity_poly", "entity_poly_seq", "struct_asym"):
        assert category not in output.block


def test_carry_accepts_met_for_deposited_selenomethionine():
    """A model that writes MET where the deposit has MSE still maps, and the deposit is kept."""
    output = _output([5, 6, 7])
    atom_site = output.block["atom_site"]
    names = atom_site["label_comp_id"].as_array(str)
    atom_site["label_comp_id"] = np.where(names == "MSE", "MET", names)

    carry_polymer_entity_categories(output, _reference())

    assert "MSE" in _column(output, "entity_poly_seq", "mon_id")
    assert _column(output, "atom_site", "label_seq_id") == ["6", "7", "8"] * 2


def test_carry_copies_deposited_chem_comp_rows():
    """Carry the deposited chemical-component rows the output uses."""
    output = _output([1, 2, 8])

    categories = carry_polymer_entity_categories(
        output, _reference(chem_comp_ids=["GLY", "HIS", "LYR", "TRP"])
    )

    assert categories[-1] == "chem_comp"
    assert sorted(_column(output, "chem_comp", "id")) == ["GLY", "HIS", "LYR"]
    assert all(name.startswith("DEPOSITED ") for name in _column(output, "chem_comp", "name"))


def test_carry_keeps_polymer_beside_non_polymer():
    """A water outside the scheme keeps its labels and does not enter the carried categories."""
    output = _output([1, 2, 3], extra=(("P", 101, "HOH"),))
    water = [
        i for i, name in enumerate(_column(output, "atom_site", "label_comp_id")) if name == "HOH"
    ]
    water_labels = [_column(output, "atom_site", "label_asym_id")[i] for i in water]

    carry_polymer_entity_categories(output, _reference())

    assert [_column(output, "atom_site", "label_asym_id")[i] for i in water] == water_labels
    assert _column(output, "struct_asym", "id") == ["A"]
    assert _column(output, "entity", "id") == ["1"]


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
