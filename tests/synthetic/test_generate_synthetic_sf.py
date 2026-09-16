"""Tests for synthetic_utils: atomarray_to_gemmi (real 6b8x structure) and
resolve_mtz_column (in-memory datasets)."""

import logging
from pathlib import Path

import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs
import torch
from atomworks.io.transforms.atom_array import remove_waters
from biotite.structure import AtomArray
from reciprocalspaceship.dtypes.base import MTZDtype
from sampleworks.synthetic.synthetic_utils import (
    assign_occupancies,
    atomarray_to_gemmi,
    resolve_mtz_column,
)
from sampleworks.utils.atom_array_utils import (
    BLANK_ALTLOC_IDS,
    detect_altlocs,
    find_all_altloc_ids,
    keep_amino_acids,
    keep_polymer,
    load_structure_with_altlocs,
    parse_structure,
    remove_hydrogens,
)
from sampleworks.utils.structure_utils import get_asym_unit_from_structure
from SFC_Torch import SFcalculator
from SFC_Torch.io import PDBParser
from SFC_Torch.utils import assert_numpy


DMIN = 2.0

# chain_info fields the model wrappers read: chain_type (all), the canonical sequence
# (Boltz/Protpardelle polymer YAML), and res_name (Boltz ligand CCD code).
CROSS_MODEL_CHAIN_INFO_KEYS = ("chain_type", "processed_entity_canonical_sequence", "res_name")


@pytest.fixture(scope="module")
def stripped_gemmi(resources_dir: Path) -> gemmi.Structure:
    """gemmi.Structure with hydrogens, ligands, and waters removed using gemmi methods."""
    s = gemmi.read_structure(str(resources_dir / "6b8x" / "6b8x_final.pdb"))
    s.remove_hydrogens()
    s.remove_ligands_and_waters()
    return s


@pytest.fixture(scope="module")
def stripped_atom_array(resources_dir: Path) -> AtomArray:
    """AtomArray with hydrogens, waters, and non-polymer/non-amino-acid atoms
    removed using existing utils."""
    arr = load_structure_with_altlocs(resources_dir / "6b8x" / "6b8x_final.pdb")
    arr = remove_hydrogens(arr)
    arr = remove_waters(arr)
    arr = keep_polymer(keep_amino_acids(arr))
    assert isinstance(arr, AtomArray)
    return arr


@pytest.fixture(scope="module")
def gemmi_structure_from_atomarray(
    stripped_atom_array, stripped_gemmi: gemmi.Structure
) -> gemmi.Structure:
    """gemmi.Structure converted from the stripped AtomArray."""
    return atomarray_to_gemmi(
        stripped_atom_array, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm
    )


def _compute_fprotein(gemmi_structure: gemmi.Structure, device: torch.device) -> np.ndarray:
    """Compute |Fprotein| amplitudes from a gemmi structure via SFcalculator at ``DMIN``
    resolution. The final assert_numpy converts any tensor or list to a numpy array."""
    sfc = SFcalculator(
        PDBParser(gemmi_structure),
        mtzdata=None,
        dmin=DMIN,
        mode="xray",
        anomalous=False,
        set_experiment=False,
        device=device,
    )
    sfc.calc_fprotein()
    return assert_numpy(sfc.Fprotein_asu)


def _build_atom_array(
    *,
    chain_id: list[str],
    res_id: list[int],
    res_name: list[str],
    atom_name: list[str],
    hetero: list[bool] | None = None,
    altloc_id: list[str] | None = None,
) -> AtomArray:
    """Build a minimal AtomArray for residue-span validation tests.

    Only the fields a test varies need to be passed; coords are distinct per atom and
    element/b_factor/occupancy are uniform, since none of them participate in residue
    grouping or span validation. An omitted ``altloc_id`` exercises the missing-annotation
    path in ``_resolve_altlocs_for_gemmi``.

    Parameters
    ----------
    chain_id : list of str
        Per-atom chain identifiers.
    res_id : list of int
        Per-atom residue identifiers.
    res_name : list of str
        Per-atom residue names.
    atom_name : list of str
        Per-atom atom names.
    hetero : list of bool, optional
        Per-atom hetero flags. If omitted, Biotite's all-false default is retained.
    altloc_id : list of str, optional
        Per-atom alternate-location identifiers. If omitted, the annotation remains
        unset rather than being populated with blank identifiers.

    Returns
    -------
    AtomArray
        Array with deterministic coordinates and the annotations needed for conversion.
    """
    n = len(atom_name)
    arr = AtomArray(n)
    arr.coord = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    arr.chain_id = np.array(chain_id)
    arr.res_id = np.array(res_id)
    arr.res_name = np.array(res_name)
    arr.atom_name = np.array(atom_name)
    arr.element = np.array(["C"] * n)
    if hetero is not None:
        arr.hetero = np.array(hetero)
    if altloc_id is not None:
        arr.set_annotation("altloc_id", np.array(altloc_id))
    arr.set_annotation("b_factor", np.full(n, 20.0))
    arr.set_annotation("occupancy", np.ones(n))
    return arr


def _dataset_with_columns(columns: dict[str, MTZDtype]) -> rs.DataSet:
    """Build a minimal rs.DataSet with each column cast to its MTZ dtype.

    Each column name maps to a dummy column cast to its MTZ dtype. The reflection index
    is irrelevant to column resolution, so it is left default.
    """
    ds = rs.DataSet({name: [1.0, 2.0, 3.0] for name in columns})
    for name, dtype in columns.items():
        ds[name] = ds[name].astype(dtype)
    return ds


class TestAtomArrayToGemmi:
    """Tests that the atomarray→gemmi conversion is faithful, using the 6b8x structure."""

    def test_cell_matches_pdb(self, gemmi_structure_from_atomarray, stripped_gemmi):
        """Unit cell parameters are preserved through the biotite→gemmi conversion."""
        result = gemmi_structure_from_atomarray.cell
        expected = stripped_gemmi.cell
        assert result.a == pytest.approx(expected.a)
        assert result.b == pytest.approx(expected.b)
        assert result.c == pytest.approx(expected.c)
        assert result.alpha == pytest.approx(expected.alpha)
        assert result.beta == pytest.approx(expected.beta)
        assert result.gamma == pytest.approx(expected.gamma)

    def test_space_group_matches_pdb(self, gemmi_structure_from_atomarray, stripped_gemmi):
        """Space group is preserved through the biotite→gemmi conversion."""
        assert gemmi_structure_from_atomarray.spacegroup_hm == stripped_gemmi.spacegroup_hm

    def test_atoms_match_pdb(self, gemmi_structure_from_atomarray, stripped_gemmi):
        """Atom names and positions match the original PDB, in the same order."""
        # atom order is preserved: biotite keeps PDB file order, and atomarray_to_gemmi
        # emits atoms in that same order
        parser_from_atomarray = PDBParser(gemmi_structure_from_atomarray)
        parser_from_gemmi = PDBParser(stripped_gemmi)
        assert np.array_equal(parser_from_atomarray.atom_name, parser_from_gemmi.atom_name)
        np.testing.assert_allclose(
            parser_from_atomarray.atom_pos, parser_from_gemmi.atom_pos, atol=1e-3
        )

    def test_occupancy_change_is_applied(self, stripped_atom_array, stripped_gemmi):
        """Custom occupancy values are correctly written to each altloc group."""
        occ_values = [0.2, 0.8, 0.0]
        altloc_info = detect_altlocs(stripped_atom_array)
        arr = assign_occupancies(stripped_atom_array, altloc_info, "custom", occ_values)
        parser = PDBParser(
            atomarray_to_gemmi(arr, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm)
        )
        for altloc, expected in zip(altloc_info.altloc_ids, occ_values):
            assert np.allclose(parser.atom_occ[altloc_info.atom_masks[altloc]], expected)

    def test_fprotein_matches_direct_gemmi(
        self, gemmi_structure_from_atomarray, stripped_gemmi, device
    ):
        """Fprotein amplitudes from the converted structure match those from
        the original gemmi structure."""
        f_atomarray = _compute_fprotein(gemmi_structure_from_atomarray, device)
        f_direct = _compute_fprotein(stripped_gemmi, device)
        np.testing.assert_allclose(np.abs(f_atomarray), np.abs(f_direct), atol=1e-3)

    def test_saved_structure_round_trips_annotations(
        self, stripped_atom_array, stripped_gemmi, tmp_path
    ):
        """Test that reloading Gemmi's CIF output returns the original atom array used to construct
        the Gemmi Structure object. This ensures we don't corrupt or lose information in building
        the Gemmi Structure that would affect our structure factor calculations.

        Fields (coords, element, b_factor, occupancy) drive the structure factors; fields
        (chain_id, res_id, res_name, atom_name, altloc_id) carry topology and alignment.
        res_id round-trips only because atomarray_to_gemmi writes label_seq_id (not just the
        author seqid), so atomworks' label scheme reads back the real residue numbers instead
        of collapsing every residue to -1. ins_code is intentionally ignored (see Issue #306).
        """
        ref = stripped_atom_array
        save_cif_path = tmp_path / "saved.cif"
        gemmi_structure = atomarray_to_gemmi(ref, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm)
        gemmi_structure.make_mmcif_document().write_file(str(save_cif_path))
        loaded = load_structure_with_altlocs(save_cif_path)

        assert len(loaded) == len(ref)

        # Extra fields can exist depending on the structure file. For example, cif vs pdb would
        # have an extra annotation is_polymer. Here, we specify the key fields we care about.
        important_annotations = {
            "chain_id",
            "res_id",
            "res_name",
            "atom_name",
            "element",
            "hetero",
            "b_factor",
            "occupancy",
            "altloc_id",
        }
        assert important_annotations <= set(ref.get_annotation_categories())
        assert important_annotations <= set(loaded.get_annotation_categories())

        # Check altloc ids survive the round-trip
        assert find_all_altloc_ids(ref)  # sanity: the reference input actually has altlocs
        blanks = list(BLANK_ALTLOC_IDS)
        loaded_altloc = np.where(np.isin(loaded.altloc_id, blanks), "", loaded.altloc_id)
        ref_altloc = np.where(np.isin(ref.altloc_id, blanks), "", ref.altloc_id)
        assert np.array_equal(loaded_altloc, ref_altloc)

        # Check the non-float annotation columns are identical
        for category in sorted(important_annotations - {"b_factor", "occupancy", "altloc_id"}):
            assert np.array_equal(loaded.get_annotation(category), ref.get_annotation(category)), (
                f"annotation {category!r} did not round-trip"
            )

        assert set(np.unique(loaded.res_id)) != {-1}  # not collapsed to the degenerate -1
        assert len(np.unique(loaded.element)) > 1  # not collapsed to a single/blank symbol

        # Check the coordinates and float annotation are within write precision
        np.testing.assert_allclose(loaded.coord, ref.coord, atol=1e-3)
        np.testing.assert_allclose(loaded.b_factor, ref.b_factor, atol=1e-2)
        np.testing.assert_allclose(loaded.occupancy, ref.occupancy, atol=1e-2)

    def test_saved_structure_round_trips_chain_info(self, resources_dir, stripped_gemmi, tmp_path):
        """Test that a cif written by atomarray_to_gemmi parses back to the same chain_info
        the source file does, so generated cifs can feed model featurization.

        The load_any test above covers _atom_site fidelity. This one covers the layer above
        it: atomworks derives chain_info from the entity block when one is present and from
        _atom_site when it is not. Emitting a partial entity block (an _entity/_entity_poly
        with no _entity_poly_seq, which gemmi's setup_entities() would produce) leads to the
        first path with nothing to read, so chain_info comes back carrying
        unprocessed_entity_canonical_sequence and every wrapper that reads
        processed_entity_canonical_sequence raises KeyError.
        """
        source = parse_structure(resources_dir / "6b8x" / "6b8x_final.pdb")
        ref = get_asym_unit_from_structure(source, 0)  # parse returns a stack; take the first model

        save_cif_path = tmp_path / "saved.cif"
        gemmi_structure = atomarray_to_gemmi(ref, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm)
        gemmi_structure.make_mmcif_document().write_file(str(save_cif_path))
        written = parse_structure(save_cif_path)

        source_info, written_info = source["chain_info"], written["chain_info"]
        assert set(written_info) == set(source_info)
        for chain_id, expected in source_info.items():
            actual = written_info[chain_id]
            for key in CROSS_MODEL_CHAIN_INFO_KEYS:
                assert key in expected, f"source chain {chain_id} has no {key!r}"
                assert key in actual, f"round-tripped chain {chain_id} lost {key!r}"
                assert np.array_equal(np.asarray(actual[key]), np.asarray(expected[key])), (
                    f"chain {chain_id} field {key!r} did not round-trip"
                )

        # The canonical sequence is per-residue, but parse's processing could drop or renumber
        # atoms without disturbing the sequence. Here we check that on a per-atom level the
        # identity is preserved. Other annotation fields (b_factor/occupancy/element) are checked
        # in the annotations round-trip test above.
        loaded = get_asym_unit_from_structure(written, 0)
        assert len(loaded) == len(ref)
        for category in ("chain_id", "res_id", "atom_name"):
            assert np.array_equal(loaded.get_annotation(category), ref.get_annotation(category)), (
                f"annotation {category!r} did not survive the parse round-trip"
            )

    def test_multichain_shared_res_ids_not_merged_in_gemmi(self):
        """Test that atomarray_to_gemmi splits shared res_ids into separate residues per chain
        in the Gemmi Structure object.
        """
        arr = _build_atom_array(
            chain_id=["A", "A", "B", "B"],
            res_id=[1, 2, 2, 3],
            res_name=["ALA", "GLY", "GLY", "ALA"],
            atom_name=["CA", "CA", "CA", "CA"],
        )
        model = atomarray_to_gemmi(arr)[0]

        chains = list(model)
        expected_chain_ids = list(dict.fromkeys(arr.chain_id.tolist()))  # unique, input order
        assert [chain.name for chain in chains] == expected_chain_ids
        for chain in chains:
            mask = arr.chain_id == chain.name
            residues = list(chain)
            # this fixture is one atom per residue, so per chain the residue seqids and atom
            # names must equal the chain's atoms one-to-one. A res_id-only merge would fuse
            # chain A's residue 2 with chain B's residue 2 (shared res_id, adjacent) into one
            # residue, changing chain A's atom count and dropping chain B's atom -- breaking these.
            assert [res.seqid.num for res in residues] == arr.res_id[mask].tolist()
            assert [res.name for res in residues] == arr.res_name[mask].tolist()
            assert [atom.name for res in residues for atom in res] == arr.atom_name[mask].tolist()

    def test_empty_atom_array_raises(self):
        """An empty AtomArray fails fast rather than yielding a chain-less structure."""
        with pytest.raises(ValueError, match="empty AtomArray"):
            atomarray_to_gemmi(AtomArray(0))

    def test_fprotein_changes_with_occupancy(self, stripped_atom_array, stripped_gemmi, device):
        """Fprotein amplitudes differ when occupancies changes from uniform to custom values."""
        altloc_info = detect_altlocs(stripped_atom_array)

        arr_uniform = assign_occupancies(stripped_atom_array, altloc_info, "uniform")
        f_uniform = _compute_fprotein(
            atomarray_to_gemmi(arr_uniform, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm),
            device,
        )

        arr_custom = assign_occupancies(stripped_atom_array, altloc_info, "custom", [0.2, 0.8, 0.0])
        f_custom = _compute_fprotein(
            atomarray_to_gemmi(arr_custom, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm),
            device,
        )

        assert not np.allclose(np.abs(f_uniform), np.abs(f_custom), atol=1e-3)


class TestGemmiHierarchyValidation:
    """Tests that atomarray_to_gemmi rejects atom arrays whose hierarchy gemmi cannot
    represent faithfully, and accepts the legitimate shapes that resemble them.

    A gemmi structure has the following hierarchy: first model is structure[0], first chain
    is model[0], first residue is chain[0], first atom is residue[0].
    """

    def test_same_atom_name_with_distinct_altlocs_is_valid(self):
        """Atoms that differ only by altloc are treated as distinct."""
        arr = _build_atom_array(
            chain_id=["A", "A"],
            res_id=[1, 1],
            res_name=["ALA", "ALA"],
            atom_name=["CA", "CA"],
            altloc_id=["A", "B"],
        )
        chains = list(atomarray_to_gemmi(arr)[0])
        assert [chain.name for chain in chains] == ["A"]
        residues = list(chains[0])
        assert len(residues) == 1
        assert [atom.altloc for atom in residues[0]] == ["A", "B"]

    def test_span_with_mixed_res_name_raises(self):
        """Atoms in one residue disagreeing on res_name are rejected."""
        arr = _build_atom_array(
            chain_id=["A", "A"],
            res_id=[1, 1],
            res_name=["ALA", "GLY"],
            atom_name=["N", "CA"],
        )
        with pytest.raises(ValueError, match="identifies each residue"):
            atomarray_to_gemmi(arr)

    def test_span_with_mixed_hetero_raises(self):
        """Atoms in one residue disagreeing on hetero are rejected."""
        arr = _build_atom_array(
            chain_id=["A", "A"],
            res_id=[1, 1],
            res_name=["MSE", "MSE"],
            atom_name=["N", "SE"],
            hetero=[False, True],
        )
        with pytest.raises(ValueError, match="disagree on hetero"):
            atomarray_to_gemmi(arr)

    def test_hetero_change_between_residues_is_valid(self):
        """hetero may change at a residue boundary, and reaches gemmi as the het_flag that
        marks the residue HETATM ('H') or ATOM ('A')."""
        arr = _build_atom_array(
            chain_id=["A", "A", "A"],
            res_id=[1, 1, 2],
            res_name=["ALA", "ALA", "MSE"],
            atom_name=["N", "CA", "SE"],
            hetero=[False, False, True],
        )
        residues = list(atomarray_to_gemmi(arr)[0][0])
        assert [res.het_flag for res in residues] == ["A", "H"]

    @pytest.mark.parametrize(
        "altloc_id",
        [None, ["", "."]],
        ids=["annotation_absent", "distinct_blank_altloc_ids"],
    )
    def test_duplicate_atom_name_in_residue_raises(self, altloc_id):
        """Atoms in a residue sharing the name and with no or blank altloc are rejected."""
        arr = _build_atom_array(
            chain_id=["A", "A"],
            res_id=[1, 1],
            res_name=["ALA", "ALA"],
            atom_name=["CA", "CA"],
            altloc_id=altloc_id,
        )
        with pytest.raises(ValueError, match="identifies each atom"):
            atomarray_to_gemmi(arr)

    def test_noncontiguous_residue_raises(self):
        """Atoms of a residue must be contiguous: a res_id that reappears after another
        residue intervenes is rejected."""
        arr = _build_atom_array(
            chain_id=["A", "A", "A"],
            res_id=[1, 2, 1],
            res_name=["ALA", "GLY", "ALA"],
            atom_name=["CA", "CA", "CB"],
        )
        with pytest.raises(ValueError, match="identifies each residue"):
            atomarray_to_gemmi(arr)

    def test_noncontiguous_chain_raises(self):
        """Atoms of a chain must be contiguous: a chain_id that reappears after another
        chain intervenes is rejected."""
        arr = _build_atom_array(
            chain_id=["A", "B", "A"],
            res_id=[1, 2, 3],
            res_name=["ALA", "GLY", "SER"],
            atom_name=["CA", "CA", "CA"],
        )
        with pytest.raises(ValueError, match="identifies each chain"):
            atomarray_to_gemmi(arr)


class TestAssignOccupancies:
    """Tests for assign_occupancies value handling, using the 6b8x altlocs."""

    def test_occupancy_warns_on_extra_values(self, stripped_atom_array, caplog):
        """A warning is logged when more occupancy values are provided than there are altlocs."""
        altloc_info = detect_altlocs(stripped_atom_array)
        with caplog.at_level(logging.WARNING):
            assign_occupancies(stripped_atom_array, altloc_info, "custom", [0.2, 0.8, 0.0, 0.0])
        assert "Extra values will be ignored" in caplog.text

    def test_occupancy_warns_on_missing_values(self, stripped_atom_array, caplog):
        """A warning is logged when fewer occupancy values are provided than there are altlocs."""
        altloc_info = detect_altlocs(stripped_atom_array)
        with caplog.at_level(logging.WARNING):
            assign_occupancies(stripped_atom_array, altloc_info, "custom", [0.5, 0.5])
        assert "Missing values are automatically set to 0" in caplog.text

    def test_occupancy_raises_on_out_of_range(self, stripped_atom_array):
        """ValueError is raised when an occupancy value is outside [0.0, 1.0]."""
        altloc_info = detect_altlocs(stripped_atom_array)
        with pytest.raises(ValueError, match="out of range"):
            assign_occupancies(stripped_atom_array, altloc_info, "custom", [1.5, 0.0, 0.0])

    def test_occupancy_raises_on_bad_sum(self, stripped_atom_array):
        """ValueError is raised when occupancy values do not sum to 1.0."""
        altloc_info = detect_altlocs(stripped_atom_array)
        with pytest.raises(ValueError, match="sum to 1.0"):
            assign_occupancies(stripped_atom_array, altloc_info, "custom", [0.3, 0.3, 0.3])


class TestResolveMtzColumn:
    """Tests for resolve_mtz_column column-selection logic."""

    @pytest.fixture
    def dataset_with_amplitude_and_phase(self) -> rs.DataSet:
        """One amplitude column (FP) and one phase column (PHI)."""
        return _dataset_with_columns(
            {"FP": rs.StructureFactorAmplitudeDtype(), "PHI": rs.PhaseDtype()}
        )

    @pytest.fixture
    def dataset_with_single_amplitude(self) -> rs.DataSet:
        """A sole amplitude column (FP), with no phase column present."""
        return _dataset_with_columns({"FP": rs.StructureFactorAmplitudeDtype()})

    @pytest.fixture
    def dataset_with_two_amplitudes(self) -> rs.DataSet:
        """Two amplitude columns (FP, FC) of the same dtype."""
        return _dataset_with_columns(
            {"FP": rs.StructureFactorAmplitudeDtype(), "FC": rs.StructureFactorAmplitudeDtype()}
        )

    def test_returns_sole_candidate(self, dataset_with_amplitude_and_phase):
        """With exactly one column of the dtype and no explicit choice, it is returned."""
        assert (
            resolve_mtz_column(dataset_with_amplitude_and_phase, rs.StructureFactorAmplitudeDtype())
            == "FP"
        )

    def test_no_candidate_raises(self, dataset_with_single_amplitude):
        """A dtype absent from the dataset fails fast."""
        with pytest.raises(ValueError, match="column found"):
            resolve_mtz_column(dataset_with_single_amplitude, rs.PhaseDtype())

    def test_ambiguous_candidates_raise(self, dataset_with_two_amplitudes):
        """Two columns of the dtype with no explicit choice is ambiguous."""
        with pytest.raises(ValueError, match="Multiple"):
            resolve_mtz_column(dataset_with_two_amplitudes, rs.StructureFactorAmplitudeDtype())

    def test_explicit_column_disambiguates(self, dataset_with_two_amplitudes):
        """An explicit column among the candidates is returned verbatim."""
        assert (
            resolve_mtz_column(
                dataset_with_two_amplitudes, rs.StructureFactorAmplitudeDtype(), column="FC"
            )
            == "FC"
        )

    def test_explicit_column_of_wrong_dtype_raises(self, dataset_with_amplitude_and_phase):
        """An explicit column that is not of the requested dtype is rejected."""
        with pytest.raises(ValueError, match="not among"):
            resolve_mtz_column(
                dataset_with_amplitude_and_phase, rs.StructureFactorAmplitudeDtype(), column="PHI"
            )
