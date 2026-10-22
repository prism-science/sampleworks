"""Tests for carrying deposited polymer entity categories into output CIFs."""

from pathlib import Path

import numpy as np
import pytest
import torch
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack, BondList, BondType, stack
from biotite.structure.io.pdbx import set_structure
from biotite.structure.io.pdbx.cif import CIFBlock, CIFCategory, CIFFile
from sampleworks.utils.cif_utils import (
    _canonical_monomer,
    _matching_entity,
    _unique_subsequence_indices,
    carry_polymer_entity_categories,
)
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import save_everything


# Deposited entity 1 (author chain P, label chain A), author-numbered from -3. Position 10 is
# microheterogeneous (CYS/CSO), and MSE/LYR/CSO are modified residues written as HETATM, as
# are the water, the glycerol ligand and the made-up ligand XYZ1 some tests add.
_SEQUENCE = ["MET", "GLY", "HIS", "HIS", "HIS", "HIS", "MSE", "CYS", "LYR", "CYS"]
_AUTHOR_OFFSET = -4  # pdb_seq_num = seq_id + _AUTHOR_OFFSET
_HETERO = {"MSE", "LYR", "CSO", "HOH", "GOL", "XYZ1"}


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
    rows.append(("1", "A", "P", len(_SEQUENCE), "CSO"))
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
            "mon_nstd_flag": ["y"] * len(chem_comp_ids),  # a column the CCD does not have
        }
    reference = CIFFile()
    reference["deposit"] = CIFBlock(
        {name: CIFCategory(columns) for name, columns in categories.items()}
    )
    return reference


def _output(
    positions: list[int],
    extra: tuple[tuple[str, int, str], ...] = (),
    renumber: bool = False,
    bonds: tuple[tuple[int, int], ...] = (),
) -> CIFFile:
    """Write a two-model, one-CA-per-residue output in the deposit's author numbering.

    Parameters
    ----------
    positions : list[int]
        0-based positions in ``_SEQUENCE`` that the output models.
    extra : tuple[tuple[str, int, str], ...]
        Additional ``(chain, author number, name)`` residues, e.g. waters.
    renumber : bool
        Write chain ``A`` numbered 1..n instead, as a model wrapper does without #414.
    bonds : tuple[tuple[int, int], ...]
        Bonded pairs of output residues, by index; ``set_structure`` writes them to
        ``struct_conn``.

    Returns
    -------
    CIFFile
        Output as written by ``set_structure``: label ids hold author values.
    """
    residues = [("P", p + 1 + _AUTHOR_OFFSET, _SEQUENCE[p]) for p in positions] + list(extra)
    if renumber:
        residues = [("A", number, name) for number, (_, _, name) in enumerate(residues, start=1)]
    atom_array = AtomArray(len(residues))
    atom_array.chain_id[:] = [chain for chain, _, _ in residues]
    atom_array.res_id[:] = [number for _, number, _ in residues]
    atom_array.res_name[:] = [name for _, _, name in residues]
    atom_array.hetero[:] = [name in _HETERO for _, _, name in residues]
    atom_array.atom_name[:] = "CA"
    atom_array.element[:] = "C"
    bond_rows = [(i, j, BondType.SINGLE) for i, j in bonds]
    atom_array.bonds = BondList(len(residues), np.array(bond_rows, dtype=int).reshape(-1, 3))
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


def test_carry_does_not_mutate_on_failed_match():
    """A residue neither the lookup nor the alignment can place fails before anything is written."""
    output = _output([1, 2, 3])
    atom_site = output.block["atom_site"]
    names = atom_site["label_comp_id"].as_array(str)
    names[0] = "TRP"
    atom_site["label_comp_id"] = names
    label_seq_ids = _column(output, "atom_site", "label_seq_id")

    with pytest.raises(ValueError, match="author lookup: .*; sequence alignment: .*"):
        carry_polymer_entity_categories(output, _reference())

    assert _column(output, "atom_site", "label_seq_id") == label_seq_ids
    for category in ("entity", "entity_poly", "entity_poly_seq", "struct_asym"):
        assert category not in output.block


def test_carry_accepts_canonical_parent_of_deposited_modified_residue():
    """A model that writes CSO where the deposit has CYS (or CYS for CSO) still maps."""
    output = _output([6, 7])
    atom_site = output.block["atom_site"]
    names = atom_site["label_comp_id"].as_array(str)
    atom_site["label_comp_id"] = np.where(names == "CYS", "CSO", names)

    carry_polymer_entity_categories(output, _reference())

    assert _column(output, "atom_site", "label_seq_id") == ["7", "8"] * 2


def test_carry_falls_back_to_alignment_for_renumbered_output():
    """A model-renumbered output (chain A, 1..n) maps onto the deposited numbering."""
    positions = [1, 2, 3, 4, 5, 6, 8, 9]  # skip MET -3 and CYS 4: one unambiguous placement
    output = _output(positions, renumber=True)

    categories = carry_polymer_entity_categories(output, _reference())

    assert "pdbx_poly_seq_scheme" not in categories
    assert _column(output, "atom_site", "label_seq_id") == [str(p + 1) for p in positions] * 2
    assert set(_column(output, "atom_site", "label_entity_id")) == {"1"}
    assert _column(output, "struct_asym", "id") == ["A"]
    assert _column(output, "entity", "id") == ["1"]


def test_carry_accepts_met_for_deposited_selenomethionine():
    """A model that writes MET where the deposit has MSE maps, and MET joins the sequence.

    The deposit's MSE row is kept and the modeled MET is added beside it as a heterogeneity
    row, so ``atom_site`` keeps its ``entity_poly_seq`` parent and ``chem_comp`` lists both.
    """
    output = _output([5, 6, 7])
    atom_site = output.block["atom_site"]
    names = atom_site["label_comp_id"].as_array(str)
    atom_site["label_comp_id"] = np.where(names == "MSE", "MET", names)

    carry_polymer_entity_categories(
        output, _reference(chem_comp_ids=["CYS", "HIS", "LYR", "MET", "MSE"])
    )

    rows = list(
        zip(
            _column(output, "entity_poly_seq", "num"),
            _column(output, "entity_poly_seq", "mon_id"),
            _column(output, "entity_poly_seq", "hetero"),
            strict=True,
        )
    )
    assert rows[5:9] == [("6", "HIS", "n"), ("7", "MSE", "y"), ("7", "MET", "y"), ("8", "CYS", "n")]
    assert len(rows) == len(_SEQUENCE) + 2  # the deposit's CSO hetero row plus the new MET row
    assert _column(output, "atom_site", "label_seq_id") == ["6", "7", "8"] * 2
    assert {"MET", "MSE"} <= set(_column(output, "chem_comp", "id"))


def test_carry_fills_chem_comp_row_the_deposit_lacks_from_the_ccd(
    caplog: pytest.LogCaptureFixture,
):
    """A modeled MET the deposit does not list gets its chem_comp row from the CCD."""
    output = _output([5, 6, 7])
    atom_site = output.block["atom_site"]
    names = atom_site["label_comp_id"].as_array(str)
    atom_site["label_comp_id"] = np.where(names == "MSE", "MET", names)

    carry_polymer_entity_categories(
        output, _reference(chem_comp_ids=["CSO", "CYS", "GLY", "HIS", "LYR", "MSE"])
    )

    rows = {
        component: (name, flag)
        for component, name, flag in zip(
            _column(output, "chem_comp", "id"),
            _column(output, "chem_comp", "name"),
            _column(output, "chem_comp", "mon_nstd_flag"),
            strict=True,
        )
    }
    assert sorted(rows) == ["CSO", "CYS", "GLY", "HIS", "LYR", "MET", "MSE"]
    assert rows["MET"] == ("METHIONINE", "?")  # the CCD has no mon_nstd_flag
    assert rows["MSE"] == ("DEPOSITED MSE", "y")  # the deposit's row is kept as it is
    assert "lacks chem_comp rows" not in caplog.text


def test_carry_copies_deposited_chem_comp_rows(caplog: pytest.LogCaptureFixture):
    """Carry the deposited chem_comp rows, fill the rest from the CCD, warn about unknown ids."""
    output = _output([1, 2, 8], extra=(("P", 101, "XYZ1"),))

    categories = carry_polymer_entity_categories(
        output, _reference(chem_comp_ids=["GLY", "HIS", "TRP"])
    )

    assert categories[-1] == "chem_comp"
    assert sorted(_column(output, "chem_comp", "id")) == [
        "CSO",
        "CYS",
        "GLY",
        "HIS",
        "LYR",
        "MET",
        "MSE",
    ]  # the carried sequence names these; TRP, which nothing names, is not carried
    deposited_names = [
        name for name in _column(output, "chem_comp", "name") if name.startswith("DEPOSITED ")
    ]
    assert deposited_names == ["DEPOSITED GLY", "DEPOSITED HIS"]
    assert "Reference CIF lacks chem_comp rows for ['XYZ1']" in caplog.text
    assert _column(output, "atom_site", "label_seq_id") == ["2", "3", "9", "."] * 2


@pytest.mark.parametrize(
    ("positions", "ligand", "renumber"),
    [
        ([1, 2, 3], ("P", 101, "HOH"), False),  # a water numbered outside the scheme
        ([1, 2, 3], ("P", 5, "GOL"), False),  # a ligand whose number is a sequence position
        ([1, 2, 3, 4, 5, 6, 8, 9], ("P", 5, "GOL"), True),  # the same through the alignment path
    ],
)
def test_carry_keeps_polymer_beside_non_polymer(positions, ligand, renumber):
    """A ligand or water keeps its chain, gets no sequence position, stays out of the carry.

    6b8x numbers its glycerols 301-303 inside chain A's 1-321, so a non-polymer row whose
    author number is also a deposited sequence position must not be read as a renamed
    residue, on either path.
    """
    output = _output(positions, extra=(ligand,), renumber=renumber)
    ligand_rows = [
        i
        for i, name in enumerate(_column(output, "atom_site", "label_comp_id"))
        if name == ligand[2]
    ]
    ligand_labels = [_column(output, "atom_site", "label_asym_id")[i] for i in ligand_rows]

    carry_polymer_entity_categories(output, _reference())

    assert [_column(output, "atom_site", "label_asym_id")[i] for i in ligand_rows] == ligand_labels
    assert {_column(output, "atom_site", "label_seq_id")[i] for i in ligand_rows} == {"."}
    assert {_column(output, "atom_site", "label_entity_id")[i] for i in ligand_rows} == {"1"}
    polymer_seq = [
        seq
        for i, seq in enumerate(_column(output, "atom_site", "label_seq_id"))
        if i not in ligand_rows
    ]
    assert polymer_seq == [str(p + 1) for p in positions] * 2
    assert _column(output, "struct_asym", "id") == ["A"]
    assert _column(output, "entity", "id") == ["1"]


@pytest.mark.parametrize("renumber", [False, True])
def test_carry_relabels_struct_conn_partners(renumber):
    """Bond partners in struct_conn follow their atoms to the deposit's labels, on both paths."""
    positions = [1, 2, 3, 4, 5, 6, 8, 9]  # one unambiguous placement when renumbered
    # MSE (author 3) to CYS (author 6), and HIS (author -1) to the glycerol.
    output = _output(
        positions, extra=(("P", 101, "GOL"),), renumber=renumber, bonds=((5, 7), (1, 8))
    )

    carry_polymer_entity_categories(output, _reference())

    labels = [
        zip(
            _column(output, "struct_conn", f"ptnr{partner}_label_asym_id"),
            _column(output, "struct_conn", f"ptnr{partner}_label_seq_id"),
            strict=True,
        )
        for partner in (1, 2)
    ]
    ligand_chain = "A" if renumber else "P"
    assert set(zip(*labels, strict=True)) == {
        (("A", "7"), ("A", "10")),
        (("A", "3"), (ligand_chain, ".")),
    }
    atom_rows = set(
        zip(
            *(
                _column(output, "atom_site", f"label_{name}")
                for name in ("asym_id", "comp_id", "seq_id", "atom_id")
            ),
            strict=True,
        )
    )
    for partner in (1, 2):
        keys = zip(
            *(
                _column(output, "struct_conn", f"ptnr{partner}_label_{name}")
                for name in ("asym_id", "comp_id", "seq_id", "atom_id")
            ),
            strict=True,
        )
        assert all(key in atom_rows for key in keys)


@pytest.fixture(scope="module")
def carved_1vme(resources_dir: Path) -> AtomArray:
    """Load the carved 1vme input once; it is numbered 1..410 by position, not by author."""
    atom_array = load_any(
        resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif",
        altloc="first",
        extra_fields=["occupancy", "b_factor"],
    )
    return atom_array[0] if isinstance(atom_array, AtomArrayStack) else atom_array


def test_carry_aligns_position_numbered_structure_to_deposit(carved_1vme, resources_dir):
    """Real data: the fallback maps each residue to the deposit's num, with matching monomers."""
    output = CIFFile()
    set_structure(output, stack([carved_1vme, carved_1vme]))
    reference = CIFFile.read(str(resources_dir / "1vme" / "1vme_rcsb.cif"))

    categories = carry_polymer_entity_categories(output, reference)

    assert "pdbx_poly_seq_scheme" not in categories
    atom_site = output.block["atom_site"]
    # Position numbering is the deposit's entity numbering for 1VME.
    assert _column(output, "atom_site", "label_seq_id") == _column(
        output, "atom_site", "auth_seq_id"
    )
    carried = dict(
        zip(
            zip(
                _column(output, "entity_poly_seq", "entity_id"),
                _column(output, "entity_poly_seq", "num"),
            ),
            _column(output, "entity_poly_seq", "mon_id"),
        )
    )
    modeled = zip(
        atom_site["label_entity_id"].as_array(str),
        atom_site["label_seq_id"].as_array(str),
        atom_site["label_comp_id"].as_array(str),
    )
    assert all(
        _canonical_monomer(carried[(entity, seq)]) == _canonical_monomer(name)
        for entity, seq, name in modeled
    )
    # The carried sequence is the deposit's, whole and in order, in both its forms.
    deposit_sequence = reference.block["entity_poly_seq"]
    deposited = [
        name
        for entity, name in zip(
            deposit_sequence["entity_id"].as_array(str),
            deposit_sequence["mon_id"].as_array(str),
            strict=True,
        )
        if entity == "1"
    ]
    assert _column(output, "entity_poly_seq", "mon_id") == deposited
    for column in ("pdbx_seq_one_letter_code", "pdbx_seq_one_letter_code_can"):
        assert "".join(_column(output, "entity_poly", column)[0].split()) == "".join(
            reference.block["entity_poly"][column].as_array(str)[0].split()
        )


def test_unique_subsequence_indices():
    """Exact matches, the fewest deletion runs, canonical parents, ambiguity, no embedding."""
    assert _unique_subsequence_indices(["GLY", "HIS", "SER"], ["GLY", "HIS", "SER"]) == (
        [0, 1, 2],
        None,
    )
    # One run of four deletions (cost 1) beats two separate runs (cost 2).
    reference = ["ALA", "GLY", "SER", "ALA", "THR", "ALA"]
    assert _unique_subsequence_indices(reference, ["ALA", "ALA"]) == ([0, 5], None)
    assert _unique_subsequence_indices(["MSE", "CSO"], ["MET", "CYS"]) == ([0, 1], None)
    # Two of three alanines: three equally good placements, disagreeing from the first residue.
    assert _unique_subsequence_indices(["ALA"] * 3, ["ALA", "ALA"]) == (None, 0)
    # The ambiguity is reported where it starts, not at the start of the sequence.
    assert _unique_subsequence_indices(
        ["GLY", "HIS", "ALA", "ALA", "ALA"], ["GLY", "HIS", "ALA", "ALA"]
    ) == (None, 2)
    assert _unique_subsequence_indices(["ALA"], ["GLY"]) == (None, None)


def test_matching_entity_reports_ambiguous_placement():
    """Two of three alanines can sit in three equally good places; say so instead of guessing."""
    block = CIFBlock(
        {
            "entity_poly": CIFCategory({"entity_id": ["1"]}),
            "entity_poly_seq": CIFCategory(
                {"entity_id": ["1"] * 3, "num": ["1", "2", "3"], "mon_id": ["ALA"] * 3}
            ),
        }
    )

    with pytest.raises(ValueError, match="entity 1 around deposited residue 1 \\(ALA\\)"):
        _matching_entity(block, ["ALA", "ALA"])
    with pytest.raises(ValueError, match="No deposited entity contains"):
        _matching_entity(block, ["GLY"])


def test_matching_entity_accepts_any_deposited_alternative_at_a_position():
    """A point mutation deposited as two rows at one num matches either residue."""
    block = CIFBlock(
        {
            "entity_poly": CIFCategory({"entity_id": ["1"]}),
            "entity_poly_seq": CIFCategory(
                {
                    "entity_id": ["1"] * 4,
                    "num": ["1", "2", "2", "3"],
                    "mon_id": ["GLY", "ASN", "ASP", "HIS"],
                }
            ),
        }
    )

    assert _matching_entity(block, ["GLY", "ASP", "HIS"]) == ("1", ["1", "2", "3"])
    assert _matching_entity(block, ["GLY", "ASN", "HIS"]) == ("1", ["1", "2", "3"])
    assert _unique_subsequence_indices(["GLY", ["ASN", "ASP"], "HIS"], ["ASP", "HIS"]) == (
        [1, 2],
        None,
    )


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
