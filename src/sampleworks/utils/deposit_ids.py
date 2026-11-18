"""Write model outputs with the chain, residue, and entity ids of the PDB entry they came from.

Model wrappers often renumber residues, build residues the input lacks, and drop entity
metadata, so an output written as-is cannot be compared residue by residue with the deposited
mmCIF (the entry as deposited in the PDB). This module finds each residue in the deposited
mmCIF and writes the output with the deposited author ids (``auth_asym_id``, ``auth_seq_id``,
``pdbx_PDB_ins_code``), the label ids (``label_asym_id``, ``label_seq_id``,
``label_entity_id``, the PDB numbering of chains, sequence positions and entities), and the
categories that describe its chains, entities and ligands.
"""

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Self, TypeVar

import numpy as np
from atomworks.io.transforms.categories import category_to_dict
from biotite.structure import (
    AtomArray,
    AtomArrayStack,
    CellList,
    get_residue_positions,
    get_residue_starts,
)
from biotite.structure.info import get_from_ccd
from biotite.structure.io.pdbx import set_structure
from biotite.structure.io.pdbx.cif import CIFBlock, CIFCategory, CIFFile
from loguru import logger

from sampleworks.utils.atom_array_utils import closest_canonical_amino_acid
from sampleworks.utils.cif_utils import add_category_to_cif
from sampleworks.utils.sequence import align_to_sequence


_Structure = TypeVar("_Structure", AtomArray, AtomArrayStack)

# mmCIF category names, spelled once: a mistyped key would silently skip a category instead of
# raising.
_ATOM_SITE = "atom_site"
_ENTITY = "entity"
_ENTITY_POLY = "entity_poly"
_ENTITY_POLY_SEQ = "entity_poly_seq"
_STRUCT_ASYM = "struct_asym"
_PDBX_POLY_SEQ_SCHEME = "pdbx_poly_seq_scheme"
_CHEM_COMP = "chem_comp"
_STRUCT_CONN = "struct_conn"
_PDBX_NONPOLY_SCHEME = "pdbx_nonpoly_scheme"
# The dictionary's value for an item that does not apply, e.g. label_seq_id of a ligand atom.
_INAPPLICABLE = "."
# entity_poly_seq.hetero marks positions with more than one monomer; absent, it defaults to no.
_HETERO = "hetero"
_HETERO_YES = "y"
_HETERO_NO = "n"

# Atom pairs that link consecutive polymer residues: peptide C-N and nucleic acid O3'-P.
_BACKBONE_LINKS = (("C", "N"), ("O3'", "P"))
# Bond length range in angstroms, as in biotite's check_backbone_continuity.
_MIN_BOND_LENGTH = 1.2
_MAX_BOND_LENGTH = 1.8

_POLYMER_ENTITY_CATEGORIES = (
    _ENTITY,
    _ENTITY_POLY,
    _ENTITY_POLY_SEQ,
    _STRUCT_ASYM,
    _PDBX_POLY_SEQ_SCHEME,
)


def _canonical_monomer(name: str) -> str:
    """Return the canonical name used for conservative sequence matching.

    Parameters
    ----------
    name : str
        Three-letter monomer name.

    Returns
    -------
    str
        Canonical parent amino acid (e.g. ``MSE -> MET``, ``CSO -> CYS``), or ``name`` itself
        when it is not an amino acid.
    """
    return closest_canonical_amino_acid(name) or name


def _rows_where(category: CIFCategory, column_name: str, values: set[str]) -> dict[str, list[str]]:
    """Select category rows whose key column is one of ``values``, keeping their order.

    Parameters
    ----------
    category : CIFCategory
        Source CIF category.
    column_name : str
        Name of the column used to select rows.
    values : set[str]
        Values to keep.

    Returns
    -------
    dict[str, list[str]]
        Selected rows represented as column lists.
    """
    mask = np.isin(category[column_name].as_array(str), list(values))
    return {name: list(np.asarray(category[name].as_array(str))[mask]) for name in category}


def _blank_to_empty(values: np.ndarray) -> np.ndarray:
    """Map CIF null markers (``.``/``?``) to empty strings so insertion codes compare equal."""
    return np.where(np.isin(values, [".", "?"]), "", values)


_Label = tuple[str, str, str]  # deposited (label_asym_id, label_seq_id, label_entity_id)

# Deposited label ids of each atom; the deposit_ prefix keeps them apart from label_*
# annotations that readers load from an input file.
DEPOSIT_LABEL_ANNOTATIONS = (
    "deposit_label_asym_id",
    "deposit_label_seq_id",
    "deposit_label_entity_id",
)
# Row of pdbx_nonpoly_scheme matched to each atom; -1 when no deposited ligand or water matches.
_DEPOSIT_NONPOLY_ROW = "deposit_nonpoly_row"


@dataclass(frozen=True)
class _PolySeqSchemePosition:
    """One deposited polymer position: the ``pdbx_poly_seq_scheme`` row(s) of one residue.

    Fields are named after the ``pdbx_poly_seq_scheme`` columns. A residue at this position
    takes ``asym_id``, ``seq_id`` and ``entity_id`` as its ``label_asym_id``, ``label_seq_id``
    and ``label_entity_id``, and ``pdb_strand_id``, ``pdb_seq_num`` and ``pdb_ins_code`` as its
    ``auth_asym_id``, ``auth_seq_id`` and ``pdbx_PDB_ins_code``.

    Attributes
    ----------
    asym_id, entity_id : str
        Label chain and entity.
    seq_id : str
        Position in the ``entity_poly_seq`` of the entity (its ``num``), counting from 1. Label
        numbering has no insertion codes; only author numbering does.
    pdb_strand_id, pdb_seq_num, pdb_ins_code : str
        Author chain, residue number and insertion code (``""`` when absent).
    monomers : frozenset[str]
        Canonical names of the monomers allowed here. A microheterogeneity site (e.g.
        CYS/CSO) has one scheme row per monomer, all merged into this position.
    """

    asym_id: str
    seq_id: str
    entity_id: str
    pdb_strand_id: str
    pdb_seq_num: str
    pdb_ins_code: str
    monomers: frozenset[str]

    @property
    def label(self) -> _Label:
        """``(label_asym_id, label_seq_id, label_entity_id)`` of the position."""
        return (self.asym_id, self.seq_id, self.entity_id)

    @property
    def author_key(self) -> tuple[str, str, str]:
        """``(auth_asym_id, auth_seq_id, pdbx_PDB_ins_code)`` of the position."""
        return (self.pdb_strand_id, self.pdb_seq_num, self.pdb_ins_code)

    def with_monomer(self, name: str) -> Self:
        """Return a copy that also allows monomer ``name``.

        Parameters
        ----------
        name : str
            Monomer name; it is stored canonicalized (``MSE -> MET``).

        Returns
        -------
        Self
            The position with ``name`` added to ``monomers``.
        """
        return replace(self, monomers=self.monomers | {_canonical_monomer(name)})


def _scheme_positions(block: CIFBlock) -> tuple[_PolySeqSchemePosition, ...]:
    """List every polymer position in the deposited ``pdbx_poly_seq_scheme``, in row order.

    Positions the deposited model leaves unobserved are included.

    Parameters
    ----------
    block : CIFBlock
        Block of the deposited mmCIF, with ``pdbx_poly_seq_scheme``.

    Returns
    -------
    tuple[_PolySeqSchemePosition, ...]
        One entry per ``(asym_id, seq_id)``; microheterogeneity rows merge into ``monomers``.
    """
    scheme = category_to_dict(block, _PDBX_POLY_SEQ_SCHEME)
    positions: dict[tuple[str, str], _PolySeqSchemePosition] = {}
    for asym, seq, entity, strand, number, ins_code, monomer in zip(
        scheme["asym_id"],
        scheme["seq_id"],
        scheme["entity_id"],
        scheme["pdb_strand_id"],
        scheme["pdb_seq_num"],
        _blank_to_empty(scheme["pdb_ins_code"]),
        scheme["mon_id"],
        strict=True,
    ):
        position = positions.get((asym, seq)) or _PolySeqSchemePosition(
            asym, seq, entity, strand, number, ins_code, frozenset()
        )
        positions[(asym, seq)] = position.with_monomer(monomer)
    return tuple(positions.values())


@dataclass(frozen=True)
class Deposit:
    """A deposited mmCIF, read once and reused to label many output structures.

    Attributes
    ----------
    block : CIFBlock
        Data block of the deposited mmCIF; its chain, entity and ligand categories are copied
        into outputs.
    monomers : frozenset[str]
        Canonical names of the monomers listed in ``entity_poly_seq``, used to tell modified
        residues from ligands.
    scheme : tuple[_PolySeqSchemePosition, ...]
        Polymer positions from ``pdbx_poly_seq_scheme``, in row order (``_scheme_positions``).
    nonpoly : dict[str, Any]
        Columns of ``pdbx_nonpoly_scheme``, which lists the deposited ligands and waters; empty
        when the file has none.
    """

    block: CIFBlock
    monomers: frozenset[str]
    scheme: tuple[_PolySeqSchemePosition, ...]
    nonpoly: dict[str, Any]

    @classmethod
    def from_cif(cls, reference: CIFFile) -> Self:
        """Read a deposited mmCIF, checking that it has the polymer categories labeling needs.

        Parameters
        ----------
        reference : CIFFile
            Deposited mmCIF, e.g. as downloaded from RCSB.

        Returns
        -------
        Self
            The deposited mmCIF, ready to label outputs.

        Raises
        ------
        ValueError
            If the file has no block or several blocks, or lacks ``entity``, ``entity_poly``,
            ``entity_poly_seq``, ``struct_asym`` or ``pdbx_poly_seq_scheme``.
        """
        block = reference.block
        missing = [name for name in _POLYMER_ENTITY_CATEGORIES if name not in block]
        if missing:
            raise ValueError(f"Reference CIF lacks required categories: {missing}")
        monomers = frozenset(
            _canonical_monomer(str(name))
            for name in block[_ENTITY_POLY_SEQ]["mon_id"].as_array(str)
        )
        return cls(
            block, monomers, _scheme_positions(block), category_to_dict(block, _PDBX_NONPOLY_SCHEME)
        )


def _residue_atoms(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Return the first atom of each residue, whose annotations describe the residue.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to describe.

    Returns
    -------
    AtomArray | AtomArrayStack
        One atom per residue, in residue order.
    """
    return atom_array[..., get_residue_starts(atom_array)]


def _backbone_linked(atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
    """Find the residues bonded to a given residue in the same chain through the backbone.

    A residue is linked when its ``C`` lies within bond length of the ``N`` of another
    residue in its chain, or its ``N`` within bond length of another residue's ``C`` (``O3'``
    and ``P`` for nucleic acids). Distances come from the first model, and atoms with
    non-finite coordinates are skipped.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to inspect.

    Returns
    -------
    np.ndarray
        Whether each residue is linked, shape ``(n_residues,)``.
    """
    frame = atom_array[0] if isinstance(atom_array, AtomArrayStack) else atom_array
    residue_of_atom = get_residue_positions(frame, np.arange(frame.array_length()))
    finite = np.isfinite(frame.coord).all(axis=-1)
    linked = np.zeros(len(get_residue_starts(frame)), dtype=bool)
    for head_name, tail_name in _BACKBONE_LINKS:
        heads = np.flatnonzero((frame.atom_name == head_name) & finite)
        tails = np.flatnonzero((frame.atom_name == tail_name) & finite)
        if heads.size == 0 or tails.size == 0:
            continue
        cell_list = CellList(frame.coord[tails], cell_size=_MAX_BOND_LENGTH)
        candidates = cell_list.get_atoms(frame.coord[heads], radius=_MAX_BOND_LENGTH)
        head_of_pair, tail_column = np.nonzero(candidates >= 0)  # -1 pads the candidate lists
        head = heads[head_of_pair]
        tail = tails[candidates[head_of_pair, tail_column]]
        distance = np.linalg.norm(frame.coord[head] - frame.coord[tail], axis=-1)
        bonded = (
            (distance >= _MIN_BOND_LENGTH)
            & (residue_of_atom[head] != residue_of_atom[tail])
            & (frame.chain_id[head] == frame.chain_id[tail])
        )
        linked[residue_of_atom[head[bonded]]] = True
        linked[residue_of_atom[tail[bonded]]] = True
    return linked


def _polymer_residues(atom_array: AtomArray | AtomArrayStack, deposit: Deposit) -> np.ndarray:
    """Decide which residues of a structure belong to a polymer chain.

    A residue is polymer when it is written as ``ATOM``, or when it is a ``HETATM`` whose
    canonical name is listed in the deposited ``entity_poly_seq`` and that is bonded to its
    chain through the backbone (``_backbone_linked``) (e.g. like a modified residue such as MSE is).
    A free amino acid (a GLU ligand beside a chain containing GLU) has a listed name but no
    such bond, so it stays a ligand. A modified residue whose neighbors are both unobserved
    has no bond to test: we still link it to the chain when its ids name a position in the
    deposited sequence (``_positions_by_lookup``), but the sequence alignment does not link it.

    Only input structures are classified this way: model wrappers may write every residue as
    ``ATOM``, so model residues follow the input residues they were matched to
    (``place_in_deposit``).

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to describe.
    deposit : Deposit
        Deposited mmCIF the residues will be placed in.

    Returns
    -------
    np.ndarray
        Whether each residue is polymer, shape ``(n_residues,)``.
    """
    residue_atoms = _residue_atoms(atom_array)
    listed = np.array(
        [_canonical_monomer(str(name)) in deposit.monomers for name in residue_atoms.res_name],
        dtype=bool,
    )
    return ~residue_atoms.hetero | (listed & _backbone_linked(atom_array))


def _positions_by_lookup(
    keys: list[tuple[str, ...]],
    lookup: dict[tuple[str, ...], _PolySeqSchemePosition],
    residue_atoms: AtomArray | AtomArrayStack,
    polymer: np.ndarray,
) -> list[_PolySeqSchemePosition | None]:
    """Place residues by looking up a key for each residue among the deposited positions.

    The key holds the author ids or the label ids of the residue
    (``_resolve_deposit_positions``).

    Parameters
    ----------
    keys : list[tuple[str, ...]]
        Lookup key of each residue.
    lookup : dict[tuple[str, ...], _PolySeqSchemePosition]
        Deposited positions, by the same kind of key.
    residue_atoms : AtomArray | AtomArrayStack
        One atom per residue that ``keys`` describe (``_residue_atoms``).
    polymer : np.ndarray
        Whether each residue is polymer (``_polymer_residues``), shape ``(n_residues,)``. A
        ``HETATM`` residues that match a deposited sequence position are considered polymer.

    Returns
    -------
    list[_PolySeqSchemePosition | None]
        Deposited position of each residue, ``None`` for ligands and waters.

    Raises
    ------
    ValueError
        If a polymer residue has no position or its name is not among the monomers of its
        sequence position, or no residue is placed.
    """
    sequenced = np.array(
        [
            key in lookup and _canonical_monomer(str(name)) in lookup[key].monomers
            for key, name in zip(keys, residue_atoms.res_name, strict=True)
        ],
        dtype=bool,
    )
    polymer = polymer | sequenced
    unmatched = {key for key, is_polymer in zip(keys, polymer) if is_polymer and key not in lookup}
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} residue(s) have no deposited pdbx_poly_seq_scheme row, "
            f"e.g. {sorted(unmatched)[:3]}"
        )
    renamed = {
        (key, name)
        for key, name, is_polymer in zip(keys, residue_atoms.res_name, polymer)
        if is_polymer and _canonical_monomer(name) not in lookup[key].monomers
    }
    if renamed:
        raise ValueError(f"Residue names disagree with the deposit, e.g. {sorted(renamed)[:3]}")
    positions = [lookup[key] if is_polymer else None for key, is_polymer in zip(keys, polymer)]
    if not any(positions):
        raise ValueError("Output has no polymer residues")
    return positions


def _positions_by_alignment(
    residue_atoms: AtomArray | AtomArrayStack,
    polymer: np.ndarray,
    scheme: tuple[_PolySeqSchemePosition, ...],
) -> list[_PolySeqSchemePosition | None]:
    """Place residues by aligning the sequence of each output chain to a deposited chain.

    Used when residues follow neither the deposited author numbering nor the label numbering,
    e.g. a model that numbers from 1. Each output chain is aligned to the deposited chains whose
    ``asym_id`` or ``pdb_strand_id`` in ``pdbx_poly_seq_scheme`` equals its name, and must fit
    exactly one. Either id can appear, depending on whether the input was read by label or
    author ids. Names are compared after canonicalizing (``MSE -> MET``), and any monomer of a
    microheterogeneity site (several monomers at one position) matches.

    Chains are aligned one at a time, not pooled per entity: a model numbers each chain from 1,
    so two copies of one entity observed over different ranges share numbers but not residues.

    Parameters
    ----------
    residue_atoms : AtomArray | AtomArrayStack
        One atom per residue to place (``_residue_atoms``).
    polymer : np.ndarray
        Whether each residue is polymer (``_polymer_residues``), shape ``(n_residues,)``.
    scheme : tuple[_PolySeqSchemePosition, ...]
        Deposited positions (``Deposit.scheme``).

    Returns
    -------
    list[_PolySeqSchemePosition | None]
        Deposited position of each residue, ``None`` for ligands and waters.

    Raises
    ------
    ValueError
        If no chain has polymer residues, a chain gives two names to one residue number, a
        chain fits a deposited chain in several equally good ways, fits no deposited chain of
        its name or several, or two output chains fit the same deposited chain.
    """
    chain_positions: dict[str, list[_PolySeqSchemePosition]] = {}  # asym_id -> scheme order
    for position in scheme:
        chain_positions.setdefault(position.asym_id, []).append(position)
    positions: list[_PolySeqSchemePosition | None] = [None] * residue_atoms.array_length()
    output_chain_by_asym_id: dict[str, str] = {}
    for chain in dict.fromkeys(residue_atoms.chain_id):
        rows = np.flatnonzero((residue_atoms.chain_id == chain) & polymer)
        if rows.size == 0:
            continue  # ligands and waters
        names: dict[tuple[int, str], str] = {}  # (number, insertion code) -> name
        for row in rows:
            key = (int(residue_atoms.res_id[row]), str(residue_atoms.ins_code[row]))
            if key in names and names[key] != residue_atoms.res_name[row]:
                raise ValueError(f"Output chain {chain} has conflicting names for residue {key}")
            names[key] = str(residue_atoms.res_name[row])
        # Sort by number only: insertion codes keep input order, which need not be alphabetical
        # (thrombin runs 1H, 1G, ..., 1).
        sequence_keys = sorted(names, key=lambda key: key[0])
        observed = [_canonical_monomer(names[key]) for key in sequence_keys]
        fits: dict[str, np.ndarray] = {}  # asym_id -> index in its chain of each residue
        ambiguous: dict[str, str] = {}  # asym_id -> deposited position we're unsure about
        for asym_id, deposited in chain_positions.items():
            # A chain name can be the label id of one deposited chain and the author id of
            # another, so both are tried.
            if chain not in (asym_id, deposited[0].pdb_strand_id):
                continue
            # Output residue numbers may not follow the deposited numbering; align by name only.
            indices, ambiguous_at = align_to_sequence(
                observed, [position.monomers for position in deposited], None, None
            )
            if np.any(indices < 0):  # -1: no embedding in this chain
                continue
            if ambiguous_at is None:
                fits[asym_id] = indices
            else:
                ambiguous[asym_id] = deposited[ambiguous_at].seq_id
        if not fits and len(ambiguous) == 1:
            ((asym_id, seq_id),) = ambiguous.items()
            raise ValueError(
                f"Output chain {chain} fits the deposited sequence in several equally good ways "
                f"(chain {asym_id} around position {seq_id}): a gap inside a run of repeated "
                "residues cannot be placed from the sequence alone"
            )
        # A chain fit ambiguously still contains the sequence, so it competes with a unique fit.
        if len(fits) + len(ambiguous) != 1:
            raise ValueError(
                f"Output chain {chain} names deposited chains {sorted(fits | ambiguous)} that "
                "contain its sequence, not exactly one"
            )
        ((asym_id, indices),) = fits.items()
        if asym_id in output_chain_by_asym_id:
            raise ValueError(
                f"Output chains {output_chain_by_asym_id[asym_id]} and {chain} both name "
                f"chain {asym_id}"
            )
        output_chain_by_asym_id[asym_id] = str(chain)
        position_of = dict(
            zip(sequence_keys, (chain_positions[asym_id][index] for index in indices), strict=True)
        )
        for row in rows:
            positions[row] = position_of[
                (int(residue_atoms.res_id[row]), str(residue_atoms.ins_code[row]))
            ]
    if not output_chain_by_asym_id:
        raise ValueError("Output has no polymer chains")
    return positions


def _one_residue_per_position(
    positions: list[_PolySeqSchemePosition | None], atom_array: AtomArray | AtomArrayStack
) -> list[_PolySeqSchemePosition | None]:
    """Check that no two residues were given the same deposited position and monomer.

    A match can give one position to two residues it cannot tell apart, such as a residue the
    model built that is numbered like an observed one; their atoms then share position, monomer
    and atom name. Different monomers at one position (microheterogeneity, such as a CYS/CSO
    site) are allowed, and so are altlocs.

    Parameters
    ----------
    positions : list[_PolySeqSchemePosition | None]
        Deposited position of each residue of ``atom_array``.
    atom_array : AtomArray | AtomArrayStack
        The placed structure.

    Returns
    -------
    list[_PolySeqSchemePosition | None]
        ``positions``, unchanged.

    Raises
    ------
    ValueError
        If two atoms share a position, monomer, atom name and altloc.
    """
    residue_of_atom = get_residue_positions(atom_array, np.arange(atom_array.array_length()))
    altloc_ids = (
        atom_array.altloc_id
        if "altloc_id" in atom_array.get_annotation_categories()
        else np.full(atom_array.array_length(), "")
    )
    atoms = Counter(
        (positions[residue].label, name, atom_name, altloc_id)
        for residue, name, atom_name, altloc_id in zip(
            residue_of_atom, atom_array.res_name, atom_array.atom_name, altloc_ids, strict=True
        )
        if positions[residue] is not None
    )
    shared = sorted({atom[:2] for atom, count in atoms.items() if count > 1})
    if shared:
        raise ValueError(
            f"{len(shared)} deposited position(s) name several residues, e.g. {shared[:3]}"
        )
    return positions


def _resolve_deposit_positions(
    atom_array: AtomArray | AtomArrayStack, deposit: Deposit
) -> list[_PolySeqSchemePosition | None]:
    """Find the deposited polymer position of each residue in a structure.

    Three methods are tried in turn: ``(chain_id, res_id, ins_code)`` as deposited author ids,
    ``(chain_id, res_id)`` as deposited label chain and sequence position, then sequence
    alignment (``_positions_by_alignment``). The first that places every polymer residue, with
    no two at one position (``_one_residue_per_position``), is used.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to place.
    deposit : Deposit
        Deposited mmCIF to place it in.

    Returns
    -------
    list[_PolySeqSchemePosition | None]
        Deposited position of each residue, ``None`` for ligands and waters.

    Raises
    ------
    ValueError
        If no method places every polymer residue; the message gives the reason each failed.

    Notes
    -----
    Which ids the residues carry is inferred, not recorded: inputs are loaded without saying
    whether they were read by label or author ids. Author ids are tried first, so residues
    whose label ids also read as valid author ids, with matching names, are placed by author
    ids.
    """
    residue_atoms = _residue_atoms(atom_array)
    polymer = _polymer_residues(atom_array, deposit)
    chain_ids, res_ids = residue_atoms.chain_id, residue_atoms.res_id.astype(str)
    scheme = deposit.scheme
    errors: list[str] = []
    try:
        keys = list(zip(chain_ids, res_ids, residue_atoms.ins_code, strict=True))
        positions = _positions_by_lookup(
            keys, {p.author_key: p for p in scheme}, residue_atoms, polymer
        )
        return _one_residue_per_position(positions, atom_array)
    except ValueError as error:
        errors.append(f"author lookup: {error}")
    try:
        keys = list(zip(chain_ids, res_ids, strict=True))
        positions = _positions_by_lookup(
            keys, {(p.asym_id, p.seq_id): p for p in scheme}, residue_atoms, polymer
        )
        return _one_residue_per_position(positions, atom_array)
    except ValueError as error:
        errors.append(f"label lookup: {error}")
    try:
        positions = _positions_by_alignment(residue_atoms, polymer, scheme)
        return _one_residue_per_position(positions, atom_array)
    except ValueError as error:
        errors.append(f"sequence alignment: {error}")
    raise ValueError("; ".join(errors))


def _walk_deposited_positions(
    atom_array: AtomArray | AtomArrayStack,
    known: list[_PolySeqSchemePosition | None],
    polymer: np.ndarray,
    deposit: Deposit,
) -> list[_PolySeqSchemePosition | None]:
    """Place polymer residues that have no position by counting from a placed neighbor.

    Models build residues the input lacks, such as unobserved loops and termini. A residue ``k``
    polymer residues after the closest placed residue before it in its chain (or ``k`` before
    the first placed one, when none precedes it) takes the position ``k`` rows away in the
    ``pdbx_poly_seq_scheme`` of that deposited chain. Counting rows rather than residue numbers
    follows the deposited numbering through gaps and insertion codes (149A-149E in thrombin).

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure that ``known`` describes, e.g. a model with built residues.
    known : list[_PolySeqSchemePosition | None]
        Deposited position of each residue, ``None`` where unknown.
    polymer : np.ndarray
        Whether each residue is polymer, shape ``(n_residues,)``; only these are placed.
    deposit : Deposit
        Deposited mmCIF that ``known`` refers to.

    Returns
    -------
    list[_PolySeqSchemePosition | None]
        Deposited position of each residue, ``None`` for ligands and waters.

    Raises
    ------
    ValueError
        If a chain has no placed residue, a counted position falls outside the deposited chain
        or lists a different monomer, or two residues share a position
        (``_one_residue_per_position``).
    """
    residue_atoms = _residue_atoms(atom_array)
    chain_positions: dict[str, list[_PolySeqSchemePosition]] = {}  # asym_id -> scheme order
    for position in deposit.scheme:
        chain_positions.setdefault(position.asym_id, []).append(position)
    index_of = {
        (position.asym_id, position.seq_id): index
        for positions in chain_positions.values()
        for index, position in enumerate(positions)
    }
    positions = list(known)
    for chain in dict.fromkeys(residue_atoms.chain_id):
        rows = np.flatnonzero((residue_atoms.chain_id == chain) & polymer)
        if rows.size == 0:
            continue  # ligands and waters
        placed = np.flatnonzero([positions[row] is not None for row in rows])  # ranks in rows
        if placed.size == 0:
            raise ValueError(f"Chain {chain} has no residue placed in the deposit")
        nearest = placed[
            np.clip(np.searchsorted(placed, np.arange(rows.size), "right") - 1, 0, None)
        ]
        for rank, (row, anchor_rank) in enumerate(zip(rows, nearest, strict=True)):
            if positions[row] is not None:
                continue
            anchor = positions[rows[anchor_rank]]
            assert anchor is not None
            deposited = chain_positions[anchor.asym_id]
            index = index_of[(anchor.asym_id, anchor.seq_id)] + rank - anchor_rank
            name = str(residue_atoms.res_name[row])
            residue = f"{name} {chain} {residue_atoms.res_id[row]}{residue_atoms.ins_code[row]}"
            if not 0 <= index < len(deposited):
                raise ValueError(f"{residue} lies outside deposited chain {anchor.asym_id}")
            if _canonical_monomer(name) not in deposited[index].monomers:
                raise ValueError(
                    f"{residue} lands on deposited {sorted(deposited[index].monomers)} at "
                    f"{anchor.asym_id} {deposited[index].seq_id}"
                )
            positions[row] = deposited[index]
    return _one_residue_per_position(positions, atom_array)


def _match_deposited_ligands(
    residue_atoms: AtomArray | AtomArrayStack,
    positions: list[_PolySeqSchemePosition | None],
    nonpoly: dict[str, Any],
) -> dict[int, int]:
    """Find the deposited ``pdbx_nonpoly_scheme`` row of each ligand and water.

    A residue without a polymer position matches a row when its ``res_id``, ``ins_code`` and
    ``res_name`` equal ``pdb_seq_num``, ``pdb_ins_code`` and ``mon_id`` in that row, and its
    ``chain_id`` equals ``asym_id`` or ``pdb_strand_id`` (inputs may carry either). Residues
    without a match, such as a ligand missing from the deposited mmCIF, are left out.

    Parameters
    ----------
    residue_atoms : AtomArray | AtomArrayStack
        One atom per residue to match (``_residue_atoms``).
    positions : list[_PolySeqSchemePosition | None]
        Deposited polymer position of each residue; only residues without one are matched.
    nonpoly : dict[str, Any]
        Columns of the deposited ``pdbx_nonpoly_scheme``, empty when there is none.

    Returns
    -------
    dict[int, int]
        Residue index -> row index in ``pdbx_nonpoly_scheme``.

    Raises
    ------
    ValueError
        If a residue matches several scheme rows.
    """
    if not nonpoly:
        return {}
    candidates: dict[tuple[str, ...], set[int]] = {}  # (chain, number, code, monomer) -> rows
    for index, (asym, strand, number, code, monomer) in enumerate(
        zip(
            nonpoly["asym_id"],
            nonpoly["pdb_strand_id"],
            nonpoly["pdb_seq_num"],
            _blank_to_empty(nonpoly["pdb_ins_code"]),
            nonpoly["mon_id"],
            strict=True,
        )
    ):
        for chain in (asym, strand):
            candidates.setdefault((chain, number, code, monomer), set()).add(index)
    ligands: dict[int, int] = {}
    for residue, (position, *key) in enumerate(
        zip(
            positions,
            residue_atoms.chain_id,
            residue_atoms.res_id.astype(str),
            residue_atoms.ins_code,
            residue_atoms.res_name,
            strict=True,
        )
    ):
        matches = candidates.get(tuple(key), set()) if position is None else set()
        if len(matches) > 1:
            raise ValueError(f"{key[3]} {key[0]} {key[1]}{key[2]} names several deposited ligands")
        if matches:
            (ligands[residue],) = matches
    return ligands


def _new_entity_ids(reference_block: CIFBlock, residue_names: list[str]) -> dict[str, str]:
    """Assign one new entity id per residue name, numbered after the highest deposited one.

    New ids keep ligands missing from the deposited mmCIF from claiming a deposited entity.

    Parameters
    ----------
    reference_block : CIFBlock
        Block of the deposited mmCIF, with ``entity``.
    residue_names : list[str]
        Names of residues missing from the deposited mmCIF.

    Returns
    -------
    dict[str, str]
        Residue name -> new entity id.
    """
    deposited = [int(entity_id) for entity_id in reference_block[_ENTITY]["id"].as_array(str)]
    first = max(deposited, default=0) + 1
    return {name: str(first + i) for i, name in enumerate(sorted(set(residue_names)))}


def place_in_deposit(
    model_atom_array: AtomArray,
    input_atom_array: AtomArray,
    model_indices: np.ndarray,
    input_indices: np.ndarray,
    deposit: Deposit,
) -> AtomArray:
    """Label a model with the deposited ids of its residues, using the input it was matched to.

    Models renumber residues and build ones the input lacks, so their own numbering cannot be
    looked up in the deposited mmCIF. Instead, the input residues are placed in the deposited
    mmCIF (``_resolve_deposit_positions``), each model residue with a matched atom takes the
    position of its input residue, and residues the model built are placed by counting from
    their neighbors along the deposited chain (``_walk_deposited_positions``). A model residue
    matched to an input ligand stays a ligand, whatever the model wrote as its ``hetero`` flag.

    Parameters
    ----------
    model_atom_array : AtomArray
        Model topology, ``n_model`` atoms.
    input_atom_array : AtomArray
        Input topology the model was matched to, ``n_input`` atoms.
    model_indices, input_indices : np.ndarray
        Matched atoms as indices into each array, shape ``(n_common,)`` (``AtomReconciler``).
    deposit : Deposit
        Deposited mmCIF the input came from.

    Returns
    -------
    AtomArray
        Annotated copy of ``model_atom_array`` (``annotate_deposit_ids``).

    Raises
    ------
    ValueError
        If the input or a model residue cannot be placed.

    Notes
    -----
    A model residue whose atoms match two input residues takes the position of the first,
    without checking that the two agree; ``_one_residue_per_position`` catches only a
    resulting clash.
    """
    input_positions = _resolve_deposit_positions(input_atom_array, deposit)
    model_residue = get_residue_positions(model_atom_array, model_indices)  # (n_common,)
    input_residue = get_residue_positions(input_atom_array, input_indices)  # (n_common,)
    n_model_residues = len(get_residue_starts(model_atom_array))
    known: list[_PolySeqSchemePosition | None] = [None] * n_model_residues
    for residue, matched in zip(model_residue, input_residue, strict=True):
        known[residue] = known[residue] or input_positions[matched]
    input_ligand = np.array([position is None for position in input_positions])
    model_polymer = np.ones(n_model_residues, dtype=bool)
    model_polymer[model_residue[input_ligand[input_residue]]] = False
    positions = _walk_deposited_positions(model_atom_array, known, model_polymer, deposit)
    return annotate_deposit_ids(model_atom_array, deposit, positions)


def annotate_deposit_ids(
    atom_array: _Structure,
    deposit: Deposit,
    positions: list[_PolySeqSchemePosition | None] | None = None,
) -> _Structure:
    """Return a copy of a structure relabeled with the deposited ids of its residues.

    Placing residues can take a sequence alignment, so annotate once and reuse the copy for
    every frame written with ``set_deposited_structure``.

    - A polymer residue takes the author ids of its deposited position as ``chain_id``,
      ``res_id`` and ``ins_code``, and the label ids as the ``DEPOSIT_LABEL_ANNOTATIONS``.
    - A ligand or water listed in the deposited ``pdbx_nonpoly_scheme``
      (``_match_deposited_ligands``) takes the ids of that row the same way, and the row index
      as the ``deposit_nonpoly_row`` annotation.
    - Any other residue keeps its ids and gets a new entity id per residue name
      (``_new_entity_ids``), so it cannot claim a deposited entity.

    Residues outside the polymer have no sequence position (``.``).

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to annotate.
    deposit : Deposit
        Deposited mmCIF to place it in.
    positions : list[_PolySeqSchemePosition | None] | None
        Deposited position of each residue; found with ``_resolve_deposit_positions`` when
        omitted.

    Returns
    -------
    AtomArray | AtomArrayStack
        Annotated copy.

    Raises
    ------
    ValueError
        If ``positions`` is omitted and the residues cannot be placed, or a residue matches
        several deposited ligands.
    """
    if positions is None:
        positions = _resolve_deposit_positions(atom_array, deposit)
    reference_block = deposit.block
    residue_atoms = _residue_atoms(atom_array)
    nonpoly = deposit.nonpoly
    ligands = _match_deposited_ligands(residue_atoms, positions, nonpoly)
    new_entity_ids = _new_entity_ids(
        reference_block,
        [
            str(name)
            for residue, (position, name) in enumerate(zip(positions, residue_atoms.res_name))
            if position is None and residue not in ligands
        ],
    )
    # (author chain, number, insertion code), (label asym, seq, entity) per residue
    residue_ids: list[tuple[tuple[str, int, str], _Label]] = []
    for residue, (position, chain, number, code, name) in enumerate(
        zip(
            positions,
            residue_atoms.chain_id,
            residue_atoms.res_id,
            residue_atoms.ins_code,
            residue_atoms.res_name,
            strict=True,
        )
    ):
        if position is not None:
            author = (position.pdb_strand_id, int(position.pdb_seq_num), position.pdb_ins_code)
            residue_ids.append((author, position.label))
        elif residue in ligands:
            row = ligands[residue]
            author = (
                str(nonpoly["pdb_strand_id"][row]),
                int(nonpoly["pdb_seq_num"][row]),
                str(_blank_to_empty(nonpoly["pdb_ins_code"])[row]),
            )
            label = (str(nonpoly["asym_id"][row]), _INAPPLICABLE, str(nonpoly["entity_id"][row]))
            residue_ids.append((author, label))
        else:
            residue_ids.append(
                (
                    (str(chain), int(number), str(code)),
                    (str(chain), _INAPPLICABLE, new_entity_ids[str(name)]),
                )
            )
    residue_of_atom = get_residue_positions(atom_array, np.arange(atom_array.array_length()))
    annotated = atom_array.copy()
    for index, name in enumerate(("chain_id", "res_id", "ins_code")):
        values = np.array([author[index] for author, _ in residue_ids])
        annotated.set_annotation(
            name, values[residue_of_atom].astype(atom_array.get_annotation(name).dtype)
        )
    for index, name in enumerate(DEPOSIT_LABEL_ANNOTATIONS):
        values = np.array([label[index] for _, label in residue_ids])
        annotated.set_annotation(name, values[residue_of_atom])
    nonpoly_rows = np.array([ligands.get(residue, -1) for residue in range(len(residue_ids))])
    annotated.set_annotation(_DEPOSIT_NONPOLY_ROW, nonpoly_rows[residue_of_atom])
    return annotated


def _carried_categories(
    reference_block: CIFBlock,
    polymer_labels: set[_Label],
    nonpoly: dict[str, Any],
    ligand_rows: list[int],
) -> dict[str, dict[str, list[str]]]:
    """Select the deposited rows that describe the chains, entities and ligands in an output.

    Rows of ``entity``, ``entity_poly`` and ``entity_poly_seq`` are kept for the polymer
    entities in ``polymer_labels``, and rows of ``struct_asym`` and ``pdbx_poly_seq_scheme``
    for their chains. Ligands in ``ligand_rows`` keep their rows of ``pdbx_nonpoly_scheme``,
    ``entity`` and ``struct_asym``. In ``pdbx_poly_seq_scheme``, ``auth_seq_num`` and
    ``auth_mon_id`` are ``?`` at positions the output does not model.

    Parameters
    ----------
    reference_block : CIFBlock
        Block of the deposited mmCIF.
    polymer_labels : set[_Label]
        Deposited label ids of the polymer residues in the output.
    nonpoly : dict[str, Any]
        Columns of the deposited ``pdbx_nonpoly_scheme``, empty when there is none.
    ligand_rows : list[int]
        Sorted rows of ``pdbx_nonpoly_scheme`` matched to ligands and waters in the output.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Category name -> columns to write.
    """
    asym_ids = {asym_id for asym_id, _, _ in polymer_labels}
    entity_ids = {entity_id for _, _, entity_id in polymer_labels}
    ligand_asym_ids = {str(nonpoly["asym_id"][row]) for row in ligand_rows}
    ligand_entity_ids = {str(nonpoly["entity_id"][row]) for row in ligand_rows}
    categories = {
        name: _rows_where(reference_block[name], column, ids_to_keep)
        for name, column, ids_to_keep in (
            (_ENTITY, "id", entity_ids | ligand_entity_ids),
            (_ENTITY_POLY, "entity_id", entity_ids),
            (_ENTITY_POLY_SEQ, "entity_id", entity_ids),
            (_STRUCT_ASYM, "id", asym_ids | ligand_asym_ids),
            (_PDBX_POLY_SEQ_SCHEME, "asym_id", asym_ids),
        )
    }
    if ligand_rows:
        categories[_PDBX_NONPOLY_SCHEME] = {
            column: list(values[ligand_rows]) for column, values in nonpoly.items()
        }
    carried_scheme = categories[_PDBX_POLY_SEQ_SCHEME]
    modeled = {(asym_id, seq_id) for asym_id, seq_id, _ in polymer_labels}
    has_coords = [
        row in modeled
        for row in zip(carried_scheme["asym_id"], carried_scheme["seq_id"], strict=True)
    ]
    for column, source in (("auth_seq_num", "pdb_seq_num"), ("auth_mon_id", "mon_id")):
        carried_scheme[column] = [
            value if present else "?"
            for value, present in zip(carried_scheme[source], has_coords, strict=True)
        ]
    return categories


def _add_modeled_monomers(
    entity_poly_seq: dict[str, list[str]],
    atom_site: CIFCategory,
    labels: list[_Label | None],
) -> None:
    """Add ``entity_poly_seq`` rows for modeled monomers that the deposited sequence lacks.

    Residue matching accepts the canonical parent of a modified residue (MET where the
    deposited sequence has MSE, CYS where it has CSO). mmCIF requires every polymer
    ``atom_site`` row to match an ``entity_poly_seq`` row on ``(entity_id, num, mon_id)``, so
    the modeled monomer is recorded as microheterogeneity (two monomers at one sequence
    position): a row with the modeled name is inserted after the deposited row, and both are
    marked ``hetero = y``.

    Side effect: ``entity_poly_seq`` is modified in place, gaining a ``hetero`` column if it has
    none. Nothing changes when every modeled name is already listed.

    Parameters
    ----------
    entity_poly_seq : dict[str, list[str]]
        Columns of the ``entity_poly_seq`` being written.
    atom_site : CIFCategory
        Output ``atom_site``.
    labels : list[_Label | None]
        Deposited label ids of each ``atom_site`` row, ``None`` for ligand and water atoms.
    """
    keys = list(zip(entity_poly_seq["entity_id"], entity_poly_seq["num"], strict=True))
    deposited = set(zip(keys, entity_poly_seq["mon_id"], strict=True))
    modeled = {
        ((label[2], label[1]), str(name))
        for label, name in zip(labels, atom_site["label_comp_id"].as_array(str), strict=True)
        if label is not None
    }
    added = sorted(modeled - deposited)
    if not added:
        return
    if _HETERO not in entity_poly_seq:
        entity_poly_seq[_HETERO] = [_HETERO_NO] * len(keys)
    last_row = {key: index for index, key in enumerate(keys)}  # the later row wins
    source_rows = np.array([last_row[key] for key, _ in added])
    hetero_keys = {key for key, _ in added}
    is_hetero = np.array([key in hetero_keys for key in keys])
    for column, values in entity_poly_seq.items():
        column_values = np.asarray(values, dtype=object)
        if column == _HETERO:
            column_values = np.where(is_hetero, _HETERO_YES, column_values)
            new_values = np.full(len(added), _HETERO_YES, dtype=object)
        elif column == "mon_id":
            new_values = np.array([mon_id for _, mon_id in added], dtype=object)
        else:
            new_values = column_values[source_rows]
        entity_poly_seq[column] = list(np.insert(column_values, source_rows + 1, new_values))


def _carried_chem_comp(chem_comp: CIFCategory, components: set[str]) -> dict[str, list[str]]:
    """Collect a ``chem_comp`` row for each component, from the deposited mmCIF or the CCD.

    Components missing from the deposited ``chem_comp`` (MET beside a deposited MSE) take
    their row from the CCD bundled with biotite, limited to the deposited columns; columns the
    CCD lacks are ``?``. Components the CCD lacks too are left out with a warning.

    Parameters
    ----------
    chem_comp : CIFCategory
        Deposited ``chem_comp``.
    components : set[str]
        Component names that need a row.

    Returns
    -------
    dict[str, list[str]]
        ``chem_comp`` columns: deposited rows first, then rows from the CCD.
    """
    reference_components = np.asarray(chem_comp["id"].as_array(str))
    component_mask = np.isin(reference_components, list(components))
    chem_comp_rows = {
        column: list(np.asarray(chem_comp[column].as_array(str))[component_mask])
        for column in chem_comp
    }
    unlisted_components = sorted(map(str, components - set(reference_components[component_mask])))
    unknown_components = []
    for component in unlisted_components:
        ccd_row = get_from_ccd(_CHEM_COMP, component)
        if ccd_row is None:
            unknown_components.append(component)
            continue
        for column, values in chem_comp_rows.items():
            values.append(ccd_row[column].as_array(str)[0] if column in ccd_row else "?")
    if unknown_components:
        logger.warning(
            f"Reference CIF lacks chem_comp rows for {unknown_components}, and so does the "
            "CCD; carrying the rows found"
        )
    return chem_comp_rows


def _relabeled_struct_conn_partners(
    struct_conn: CIFCategory, relabel: dict[tuple[str, str, str, str], tuple[str, str]]
) -> dict[str, np.ndarray]:
    """Point ``struct_conn`` bond partners at the deposited label ids of their residues.

    ``set_structure`` names each partner by the label ids it wrote to ``atom_site``, so
    partners are found by ``(label_asym_id, label_seq_id, insertion code, label_comp_id)``.

    Parameters
    ----------
    struct_conn : CIFCategory
        ``struct_conn`` as ``set_structure`` wrote it.
    relabel : dict[tuple[str, str, str, str], tuple[str, str]]
        Maps each residue, by the ids ``set_structure`` wrote, to its deposited
        ``(label_asym_id, label_seq_id)``.

    Returns
    -------
    dict[str, np.ndarray]
        New ``ptnr1_*`` and ``ptnr2_*`` columns for ``label_asym_id`` and ``label_seq_id``.

    Raises
    ------
    KeyError
        If a bond partner is not a residue in ``relabel``.
    """
    partner_labels: dict[str, np.ndarray] = {}
    for partner in (1, 2):
        partners = zip(
            struct_conn[f"ptnr{partner}_label_asym_id"].as_array(str),
            struct_conn[f"ptnr{partner}_label_seq_id"].as_array(str),
            _blank_to_empty(struct_conn[f"pdbx_ptnr{partner}_PDB_ins_code"].as_array(str)),
            struct_conn[f"ptnr{partner}_label_comp_id"].as_array(str),
            strict=True,
        )
        asym_ids, seq_ids = zip(*(relabel[key] for key in partners), strict=True)
        partner_labels[f"ptnr{partner}_label_asym_id"] = np.array(asym_ids)
        partner_labels[f"ptnr{partner}_label_seq_id"] = np.array(seq_ids)
    return partner_labels


def set_deposited_structure(
    cif_file: CIFFile, atom_array: AtomArray | AtomArrayStack, deposit: Deposit
) -> tuple[str, ...]:
    """Write a structure to a CIF with the chain, residue and entity ids of its PDB entry.

    The written file can be compared residue by residue with the deposited mmCIF and carries
    the chain, entity and ligand metadata of the entry. ``set_structure`` writes the atoms of a
    structure annotated by ``annotate_deposit_ids``, with the deposited author ids it stored in
    ``chain_id``, ``res_id`` and ``ins_code``; then

    - ``label_asym_id``, ``label_seq_id`` and ``label_entity_id`` in ``atom_site`` are set to
      the deposited ids, and ``struct_conn`` bond partners are updated to match;
    - the deposited rows for the chains, entities and ligands present are copied
      (``_carried_categories``);
    - a modeled monomer that differs from the deposited one at its position (MET for MSE) is
      added to ``entity_poly_seq`` (``_add_modeled_monomers``);
    - when the deposited mmCIF has ``chem_comp``, a row is copied for every component in the
      output (``_carried_chem_comp``).

    Parameters
    ----------
    cif_file : CIFFile
        Output CIF. Side effect: the structure and categories are written into it. If this
        function raises, ``cif_file`` may be partly written and should be discarded.
    atom_array : AtomArray | AtomArrayStack
        Structure annotated by ``annotate_deposit_ids``.
    deposit : Deposit
        Deposited mmCIF used for the annotations.

    Returns
    -------
    tuple[str, ...]
        Names of the categories copied from the deposited mmCIF.

    Raises
    ------
    ValueError
        If ``atom_array`` lacks the annotations ``annotate_deposit_ids`` sets, or has no
        deposited polymer residue.
    KeyError
        If a ``struct_conn`` bond partner is not a written residue.
    """
    missing = {*DEPOSIT_LABEL_ANNOTATIONS, _DEPOSIT_NONPOLY_ROW} - set(
        atom_array.get_annotation_categories()
    )
    if missing:
        raise ValueError(f"Atom array lacks deposit annotations {sorted(missing)}")
    reference_block = deposit.block
    set_structure(cif_file, atom_array)
    block = cif_file.block
    atom_site = block[_ATOM_SITE]
    models = atom_site.row_count // atom_array.array_length()
    asym_ids, seq_ids, entity_ids = (
        np.tile(atom_array.get_annotation(name).astype(str), models)
        for name in DEPOSIT_LABEL_ANNOTATIONS
    )
    labels: list[_Label | None] = [
        None if seq == _INAPPLICABLE else (asym, seq, entity)
        for asym, seq, entity in zip(asym_ids, seq_ids, entity_ids, strict=True)
    ]
    polymer_labels = {label for label in labels if label is not None}
    if not polymer_labels:
        raise ValueError("Output has no deposited polymer residues")

    ligand_rows = sorted(set(atom_array.get_annotation(_DEPOSIT_NONPOLY_ROW).tolist()) - {-1})

    # Copy the deposited rows for the chains and ligands present, plus monomers the model adds.
    categories = _carried_categories(reference_block, polymer_labels, deposit.nonpoly, ligand_rows)
    _add_modeled_monomers(categories[_ENTITY_POLY_SEQ], atom_site, labels)
    if _CHEM_COMP in reference_block:
        components = set(atom_site["label_comp_id"].as_array(str))
        components |= set(categories[_ENTITY_POLY_SEQ]["mon_id"])
        categories[_CHEM_COMP] = _carried_chem_comp(reference_block[_CHEM_COMP], components)

    # set_structure named struct_conn partners by its own label ids; point them at deposited ones.
    if _STRUCT_CONN in block:
        written = zip(
            atom_site["label_asym_id"].as_array(str),
            atom_site["label_seq_id"].as_array(str),
            _blank_to_empty(atom_site["pdbx_PDB_ins_code"].as_array(str)),
            atom_site["label_comp_id"].as_array(str),
            strict=True,
        )
        relabel = dict(zip(written, zip(asym_ids, seq_ids, strict=True), strict=True))
        for column, values in _relabeled_struct_conn_partners(block[_STRUCT_CONN], relabel).items():
            block[_STRUCT_CONN][column] = values

    atom_site["label_asym_id"] = asym_ids
    atom_site["label_seq_id"] = seq_ids
    atom_site["label_entity_id"] = entity_ids
    for name, data in categories.items():
        add_category_to_cif(cif_file, data, name, overwrite=True)
    return tuple(categories)
