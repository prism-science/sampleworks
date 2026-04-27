"""Place structures in their deposited mmCIF's numbering and carry its polymer categories."""

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, NamedTuple, Self, TypeVar

import numpy as np
from atomworks.io.transforms.categories import category_to_dict
from biotite.structure import (
    AtomArray,
    AtomArrayStack,
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


# mmCIF category names used by the entity carry, spelled once: a mistyped key would silently
# skip a category instead of raising.
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

_POLYMER_ENTITY_CATEGORIES = (
    _ENTITY,
    _ENTITY_POLY,
    _ENTITY_POLY_SEQ,
    _STRUCT_ASYM,
    _PDBX_POLY_SEQ_SCHEME,
)


def _single_block(cif_file: CIFFile) -> CIFBlock:
    """Return the sole data block from a CIF file.

    Parameters
    ----------
    cif_file : CIFFile
        CIF file expected to contain exactly one data block.

    Returns
    -------
    CIFBlock
        The sole data block.

    Raises
    ------
    ValueError
        If the CIF file does not contain exactly one data block.
    """
    block_names = list(cif_file.keys())
    if len(block_names) != 1:
        raise ValueError(f"Expected one CIF block, found {len(block_names)}")
    return cif_file[block_names[0]]


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


def _matching_entity(reference_block: CIFBlock, residue_names: list[str]) -> tuple[str, list[str]]:
    """Find one deposited entity with a unique ordered modeled sequence match.

    Parameters
    ----------
    reference_block : CIFBlock
        Deposited CIF block containing polymer entity categories.
    residue_names : list[str]
        Output residue names in ``label_seq_id`` order.

    Returns
    -------
    tuple[str, list[str]]
        Reference entity ID and the deposited ``entity_poly_seq.num`` of each output residue.

    Raises
    ------
    ValueError
        If the output is empty, unmatched, or ambiguously matched.
    """
    if not residue_names:
        raise ValueError("Output entity has no polymer residues")
    sequence = reference_block[_ENTITY_POLY_SEQ]
    sequence_ids = np.asarray(sequence["entity_id"].as_array(str))
    sequence_nums = np.asarray(sequence["num"].as_array(str))
    sequence_names = np.asarray(sequence["mon_id"].as_array(str))
    matches: list[tuple[str, list[str]]] = []
    ambiguous: list[str] = []  # "entity 1 around deposited residue 158 (ALA)"
    for entity_id in reference_block[_ENTITY_POLY]["entity_id"].as_array(str):
        rows = np.flatnonzero(sequence_ids == entity_id)
        # Microheterogeneity (several monomers at one residue): any of them may match.
        nums = sorted({str(num) for num in sequence_nums[rows]}, key=lambda num: int(num))
        reference_names = [
            [str(name) for name in sequence_names[rows][sequence_nums[rows] == num]] for num in nums
        ]
        # Output numbering says nothing about the deposit's gaps.
        positions, ambiguous_at = align_to_sequence(
            [_canonical_monomer(name) for name in residue_names],
            [{_canonical_monomer(name) for name in names} for names in reference_names],
            None,
            None,
        )
        if np.any(positions < 0):
            continue  # no embedding in this entity
        if ambiguous_at is None:
            matches.append((str(entity_id), [nums[position] for position in positions]))
        else:
            ambiguous.append(
                f"entity {entity_id} around deposited residue {nums[ambiguous_at]} "
                f"({'/'.join(reference_names[ambiguous_at])})"
            )
    if len(matches) == 1:
        return matches[0]
    if matches:
        entity_ids = [entity_id for entity_id, _ in matches]
        raise ValueError(f"Modeled sequence matches several deposited entities: {entity_ids}")
    if ambiguous:
        raise ValueError(
            "Modeled sequence fits the deposited sequence in several equally good ways "
            f"({'; '.join(ambiguous)}): a gap inside a run of repeated residues cannot be "
            "placed from the sequence alone"
        )
    raise ValueError("No deposited entity contains the modeled sequence")


def _select_category_rows(
    category: CIFCategory,
    column_name: str,
    value: str,
) -> dict[str, list[str]]:
    """Select category rows whose key column equals a value.

    Parameters
    ----------
    category : CIFCategory
        Source CIF category.
    column_name : str
        Name of the column used to select rows.
    value : str
        Value to match.

    Returns
    -------
    dict[str, list[str]]
        Selected rows represented as column lists.
    """
    mask = np.asarray(category[column_name].as_array(str)) == value
    return {name: list(np.asarray(category[name].as_array(str))[mask]) for name in category}


def _blank_to_empty(values: np.ndarray) -> np.ndarray:
    """Map CIF null markers (``.``/``?``) to empty strings so insertion codes compare equal."""
    return np.where(np.isin(values, [".", "?"]), "", values)


def _concatenate_category_rows(rows: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    """Concatenate compatible CIF category row dictionaries.

    Parameters
    ----------
    rows : list[dict[str, list[str]]]
        Row dictionaries sharing the same columns.

    Returns
    -------
    dict[str, list[str]]
        Concatenated category columns.

    Raises
    ------
    KeyError
        If a row dictionary has columns the first one lacks, or lacks columns it has.
    """
    columns = {name: [] for name in rows[0]}
    for row in rows:
        if row.keys() != columns.keys():
            raise KeyError(
                f"Category rows disagree on columns: extra {sorted(row.keys() - columns.keys())}, "
                f"missing {sorted(columns.keys() - row.keys())}"
            )
        for name, values in row.items():
            columns[name].extend(values)
    return columns


_Label = tuple[str, str, str]  # deposited (label_asym_id, label_seq_id, label_entity_id)

# Deposited label ids, written over the label columns set_structure fills from chain_id and
# res_id. Named apart from the label_* fields readers can load as annotations.
DEPOSIT_LABEL_ANNOTATIONS = (
    "deposit_label_asym_id",
    "deposit_label_seq_id",
    "deposit_label_entity_id",
)
# Each atom's deposited pdbx_nonpoly_scheme row, -1 outside the deposit's ligands and waters.
DEPOSIT_NONPOLY_ROW = "deposit_nonpoly_row"


@dataclass(frozen=True)
class PolySeqSchemePosition:
    """One deposited polymer position: the ``pdbx_poly_seq_scheme`` row(s) of one residue.

    Fields are named after the scheme's columns. A residue at this position takes ``asym_id``,
    ``seq_id`` and ``entity_id`` as its ``label_asym_id``, ``label_seq_id`` and
    ``label_entity_id``, and ``pdb_strand_id``, ``pdb_seq_num`` and ``pdb_ins_code`` as its
    ``auth_asym_id``, ``auth_seq_id`` and ``pdbx_PDB_ins_code``.

    Attributes
    ----------
    asym_id, entity_id : str
        Label chain and entity.
    seq_id : str
        Position in the entity's ``entity_poly_seq`` (its ``num``), counting from 1. It has
        no insertion code; only author numbering does.
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


def _scheme_positions(block: CIFBlock) -> tuple[PolySeqSchemePosition, ...]:
    """List every deposited polymer position, resolved or not, in scheme order.

    Parameters
    ----------
    block : CIFBlock
        Deposited block with ``pdbx_poly_seq_scheme``.

    Returns
    -------
    tuple[PolySeqSchemePosition, ...]
        One entry per ``(asym_id, seq_id)``; microheterogeneity rows merge into ``monomers``.
    """
    scheme = category_to_dict(block, _PDBX_POLY_SEQ_SCHEME)
    positions: dict[tuple[str, str], PolySeqSchemePosition] = {}
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
        position = positions.get((asym, seq)) or PolySeqSchemePosition(
            asym, seq, entity, strand, number, ins_code, frozenset()
        )
        positions[(asym, seq)] = position.with_monomer(monomer)
    return tuple(positions.values())


@dataclass(frozen=True)
class Deposit:
    """A deposited mmCIF, parsed once to place structures in it.

    Attributes
    ----------
    block : CIFBlock
        The deposit's data block, whose categories are carried.
    monomers : frozenset[str]
        Canonical names of the monomers its ``entity_poly_seq`` lists.
    scheme : tuple[PolySeqSchemePosition, ...]
        Its polymer positions in ``pdbx_poly_seq_scheme`` order (``_scheme_positions``).
    nonpoly : dict[str, Any]
        Its ``pdbx_nonpoly_scheme`` columns, empty when it has none.
    """

    block: CIFBlock
    monomers: frozenset[str]
    scheme: tuple[PolySeqSchemePosition, ...]
    nonpoly: dict[str, Any]

    @classmethod
    def from_cif(cls, reference: CIFFile) -> Self:
        """Parse a deposited CIF.

        Parameters
        ----------
        reference : CIFFile
            Deposited CIF, e.g. as downloaded from RCSB.

        Returns
        -------
        Self
            The parsed deposit.

        Raises
        ------
        ValueError
            If the file does not have exactly one block, or lacks the polymer entity
            categories.
        """
        block = _single_block(reference)
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


_Structure = TypeVar("_Structure", AtomArray, AtomArrayStack)


class _Residues(NamedTuple):
    """Residue-level annotations of an atom array, one entry per residue.

    Attributes
    ----------
    chain_id, res_id, ins_code, res_name : np.ndarray
        The residue's annotations, ``res_id`` as strings.
    hetero : np.ndarray
        Whether the residue is written as ``HETATM``.
    polymer : np.ndarray
        Whether the residue belongs to a polymer chain.
    """

    chain_id: np.ndarray
    res_id: np.ndarray
    ins_code: np.ndarray
    res_name: np.ndarray
    hetero: np.ndarray
    polymer: np.ndarray


def _residues(atom_array: AtomArray | AtomArrayStack, deposit: Deposit) -> _Residues:
    """Collect the residue-level annotations of ``atom_array``.

    A residue is polymer when it is not ``HETATM``, or when the deposit's ``entity_poly_seq``
    lists its monomer (a modified residue such as MSE or LYR), compared after canonicalizing
    both sides. Ligands and waters are outside the polymer whatever their number, which can
    coincide with a deposited sequence position (6b8x numbers its glycerols 301-303, inside
    chain A's 1-321).

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to describe.
    deposit : Deposit
        The deposit the residues are placed in.

    Returns
    -------
    _Residues
        One entry per residue, in residue order.
    """
    starts = get_residue_starts(atom_array)
    res_name = atom_array.res_name[starts]
    hetero = atom_array.hetero[starts]
    listed = np.array(
        [_canonical_monomer(str(name)) in deposit.monomers for name in res_name], dtype=bool
    )
    return _Residues(
        chain_id=atom_array.chain_id[starts],
        res_id=atom_array.res_id[starts].astype(str),
        ins_code=atom_array.ins_code[starts],
        res_name=res_name,
        hetero=hetero,
        polymer=~hetero | listed,
    )


def _positions_by_lookup(
    keys: list[tuple[str, ...]],
    lookup: dict[tuple[str, ...], PolySeqSchemePosition],
    residues: _Residues,
) -> list[PolySeqSchemePosition | None]:
    """Place residues by looking their keys up among the deposited positions.

    Parameters
    ----------
    keys : list[tuple[str, ...]]
        Lookup key per residue.
    lookup : dict[tuple[str, ...], PolySeqSchemePosition]
        Deposited positions by key.
    residues : _Residues
        The residues ``keys`` describe.

    Returns
    -------
    list[PolySeqSchemePosition | None]
        Deposited position per residue, ``None`` for residues outside the scheme (e.g.
        ligands).

    Raises
    ------
    ValueError
        If a non-``HETATM`` residue has no position, a residue's name is not one of its
        position's monomers, or no residue is placed.
    """
    unmatched = {
        key for key, hetero in zip(keys, residues.hetero) if not hetero and key not in lookup
    }
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} residue(s) have no deposited pdbx_poly_seq_scheme row, "
            f"e.g. {sorted(unmatched)[:3]}"
        )
    renamed = {
        (key, name)
        for key, name, polymer in zip(keys, residues.res_name, residues.polymer)
        if polymer and key in lookup and _canonical_monomer(name) not in lookup[key].monomers
    }
    if renamed:
        raise ValueError(f"Residue names disagree with the deposit, e.g. {sorted(renamed)[:3]}")
    positions = [
        lookup.get(key) if polymer else None for key, polymer in zip(keys, residues.polymer)
    ]
    if not any(positions):
        raise ValueError("Output has no polymer residues")
    return positions


def _positions_by_author_ids(
    residues: _Residues, scheme: tuple[PolySeqSchemePosition, ...]
) -> list[PolySeqSchemePosition | None]:
    """Place residues whose ``(chain_id, res_id, ins_code)`` are the deposit's author ids.

    Parameters
    ----------
    residues : _Residues
        Residues to place.
    scheme : tuple[PolySeqSchemePosition, ...]
        Deposited positions (``Deposit.scheme``).

    Returns
    -------
    list[PolySeqSchemePosition | None]
        Deposited position per residue, ``None`` for non-polymer residues.

    Raises
    ------
    ValueError
        As ``_positions_by_lookup``.
    """
    keys = list(zip(residues.chain_id, residues.res_id, residues.ins_code, strict=True))
    return _positions_by_lookup(keys, {p.author_key: p for p in scheme}, residues)


def _positions_by_label_ids(
    residues: _Residues, scheme: tuple[PolySeqSchemePosition, ...]
) -> list[PolySeqSchemePosition | None]:
    """Place residues whose ``(chain_id, res_id)`` are the deposit's label chain and position.

    Parameters
    ----------
    residues : _Residues
        Residues to place.
    scheme : tuple[PolySeqSchemePosition, ...]
        Deposited positions (``Deposit.scheme``).

    Returns
    -------
    list[PolySeqSchemePosition | None]
        Deposited position per residue, ``None`` for non-polymer residues.

    Raises
    ------
    ValueError
        As ``_positions_by_lookup``.
    """
    keys = list(zip(residues.chain_id, residues.res_id, strict=True))
    return _positions_by_lookup(keys, {(p.asym_id, p.seq_id): p for p in scheme}, residues)


def _positions_by_alignment(
    residues: _Residues, scheme: tuple[PolySeqSchemePosition, ...], reference_block: CIFBlock
) -> list[PolySeqSchemePosition | None]:
    """Place residues by aligning each chain's sequence to the deposit's.

    The fallback when the residues keep neither the deposit's author nor its label numbering,
    e.g. a model that numbers from 1. For each chain:

    1. Align its residue names to each deposited entity's ``entity_poly_seq`` and take the
       single entity that matches (``_matching_entity``).
    2. Pick the deposited chain of that entity whose ``pdbx_poly_seq_scheme.asym_id`` or
       ``pdb_strand_id`` equals the chain's name. Either can appear: an input read from an
       RCSB CIF brings label chain ids, a density input author ones.
    3. Give each residue that chain's position at the ``entity_poly_seq.num`` it aligned to.

    Chains are taken one by one, not merged per entity: a model numbers each chain from 1, so
    two copies of one entity resolved over different ranges share numbers but not residues.

    Parameters
    ----------
    residues : _Residues
        Residues to place.
    scheme : tuple[PolySeqSchemePosition, ...]
        Deposited positions (``Deposit.scheme``).
    reference_block : CIFBlock
        Deposited block with ``entity_poly`` and ``entity_poly_seq``.

    Returns
    -------
    list[PolySeqSchemePosition | None]
        Deposited position per residue, ``None`` for non-polymer residues.

    Raises
    ------
    ValueError
        If no chain has polymer residues, a chain has conflicting names for one residue, a
        chain has no unique match among the deposited entities, does not name exactly one
        deposited chain of its entity, or two chains name the same one.
    """
    position_at = {(p.asym_id, p.seq_id): p for p in scheme}
    deposited_chains = {(p.asym_id, p.pdb_strand_id, p.entity_id) for p in scheme}
    positions: list[PolySeqSchemePosition | None] = [None] * len(residues.chain_id)
    output_chain_by_asym_id: dict[str, str] = {}
    for chain in dict.fromkeys(residues.chain_id):
        rows = np.flatnonzero((residues.chain_id == chain) & residues.polymer)
        if rows.size == 0:
            continue  # ligands and waters
        names: dict[tuple[int, str], str] = {}  # (number, insertion code) -> name
        for row in rows:
            key = (int(residues.res_id[row]), str(residues.ins_code[row]))
            if key in names and names[key] != residues.res_name[row]:
                raise ValueError(f"Output chain {chain} has conflicting names for residue {key}")
            names[key] = str(residues.res_name[row])
        # Stable: residues sharing a number keep their order (14, 14A, 14B; thrombin's 1H ... 1).
        sequence_keys = sorted(names, key=lambda key: key[0])
        entity_id, seq_ids = _matching_entity(reference_block, [names[k] for k in sequence_keys])
        # Copies of one entity are told apart by chain name. The name is ambiguous when it is
        # one copy's label id and another's author id.
        asym_ids = {
            asym_id
            for asym_id, strand_id, entity in deposited_chains
            if entity == entity_id and chain in (asym_id, strand_id)
        }
        if len(asym_ids) != 1:
            raise ValueError(
                f"Output chain {chain} names deposited chains {sorted(asym_ids)} of entity "
                f"{entity_id}, not exactly one"
            )
        asym_id = asym_ids.pop()
        if asym_id in output_chain_by_asym_id:
            raise ValueError(
                f"Output chains {output_chain_by_asym_id[asym_id]} and {chain} both name "
                f"chain {asym_id}"
            )
        output_chain_by_asym_id[asym_id] = str(chain)
        seq_id_of = dict(zip(sequence_keys, seq_ids, strict=True))
        for row in rows:
            key = (int(residues.res_id[row]), str(residues.ins_code[row]))
            positions[row] = position_at[(asym_id, seq_id_of[key])]
    if not output_chain_by_asym_id:
        raise ValueError("Output has no polymer chains")
    return positions


def _one_residue_per_position(
    positions: list[PolySeqSchemePosition | None], atom_array: AtomArray | AtomArrayStack
) -> list[PolySeqSchemePosition | None]:
    """Check that no two residues take the same deposited position and monomer.

    Two residues a match cannot tell apart (e.g. one the model built, numbered like a resolved
    one) give one position and monomer to two atoms of the same name. Different monomers at
    one position are the deposit's microheterogeneity (a CYS/CSO site) and are allowed, as are
    altlocs.

    Parameters
    ----------
    positions : list[PolySeqSchemePosition | None]
        Deposited position per residue of ``atom_array``.
    atom_array : AtomArray | AtomArrayStack
        The placed structure.

    Returns
    -------
    list[PolySeqSchemePosition | None]
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


def resolve_deposit_positions(
    atom_array: AtomArray | AtomArrayStack, deposit: Deposit
) -> list[PolySeqSchemePosition | None]:
    """Find each residue's deposited ``pdbx_poly_seq_scheme`` position.

    Three matches are tried in turn: author ids (``_positions_by_author_ids``), label ids
    (``_positions_by_label_ids``), then sequence alignment (``_positions_by_alignment``). The
    first that places every polymer residue, and no two at one position
    (``_one_residue_per_position``), wins.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to place.
    deposit : Deposit
        The deposit to place them in.

    Returns
    -------
    list[PolySeqSchemePosition | None]
        Deposited position per residue, ``None`` for non-polymer residues.

    Raises
    ------
    ValueError
        If no match places every residue; the message gives each one's reason.

    Notes
    -----
    Which ids the residues carry is inferred, not recorded: inputs are loaded without saying
    whether they were read by label or author ids. Author ids are tried first, so residues
    whose label ids also read as valid author ids, with matching names, are placed by author
    ids.
    """
    reference_block = deposit.block
    residues = _residues(atom_array, deposit)
    scheme = deposit.scheme
    errors: list[str] = []
    try:
        positions = _positions_by_author_ids(residues, scheme)
        return _one_residue_per_position(positions, atom_array)
    except ValueError as error:
        errors.append(f"author lookup: {error}")
    try:
        positions = _positions_by_label_ids(residues, scheme)
        return _one_residue_per_position(positions, atom_array)
    except ValueError as error:
        errors.append(f"label lookup: {error}")
    try:
        positions = _positions_by_alignment(residues, scheme, reference_block)
        return _one_residue_per_position(positions, atom_array)
    except ValueError as error:
        errors.append(f"sequence alignment: {error}")
    raise ValueError("; ".join(errors))


def _walk_deposited_positions(
    atom_array: AtomArray | AtomArrayStack,
    known: list[PolySeqSchemePosition | None],
    deposit: Deposit,
) -> list[PolySeqSchemePosition | None]:
    """Place each polymer residue without a position from the nearest placed one in its chain.

    A residue ``k`` polymer residues from the nearest placed residue of its chain (before it,
    else after) takes the position ``k`` steps from that residue's in its deposited chain's
    ``pdbx_poly_seq_scheme`` order. That order follows the deposit's numbering through skips
    and insertion codes (thrombin's 149A-149E), which counting residue numbers does not.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure whose residues ``known`` describes, e.g. a model with built residues.
    known : list[PolySeqSchemePosition | None]
        Deposited position per residue, ``None`` where unknown.
    deposit : Deposit
        The deposit to place them in.

    Returns
    -------
    list[PolySeqSchemePosition | None]
        Deposited position per residue, ``None`` for non-polymer residues.

    Raises
    ------
    ValueError
        If a chain has no placed residue, a residue would fall outside its deposited chain or
        names a monomer its position does not list, or two residues share a position
        (``_one_residue_per_position``).
    """
    residues = _residues(atom_array, deposit)
    chain_positions: dict[str, list[PolySeqSchemePosition]] = {}  # asym_id -> scheme order
    for position in deposit.scheme:
        chain_positions.setdefault(position.asym_id, []).append(position)
    index_of = {
        (position.asym_id, position.seq_id): index
        for positions in chain_positions.values()
        for index, position in enumerate(positions)
    }
    positions = list(known)
    for chain in dict.fromkeys(residues.chain_id):
        rows = np.flatnonzero((residues.chain_id == chain) & residues.polymer)
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
            name = str(residues.res_name[row])
            residue = f"{name} {chain} {residues.res_id[row]}{residues.ins_code[row]}"
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
    residues: _Residues, positions: list[PolySeqSchemePosition | None], nonpoly: dict[str, Any]
) -> dict[int, int]:
    """Match residues without a polymer position to the deposit's ``pdbx_nonpoly_scheme`` rows.

    A residue matches a scheme row when its ``res_id``, ``ins_code`` and ``res_name`` equal the
    row's ``pdb_seq_num``, ``pdb_ins_code`` and ``mon_id``, and its ``chain_id`` is the row's
    ``asym_id`` or ``pdb_strand_id`` (the input may carry either). Residues without a match,
    such as a ligand the deposit lacks, are left out.

    Parameters
    ----------
    residues : _Residues
        Residues to match.
    positions : list[PolySeqSchemePosition | None]
        Deposited polymer position per residue; only residues without one are matched.
    nonpoly : dict[str, Any]
        Deposited ``pdbx_nonpoly_scheme`` columns, empty when the deposit has none.

    Returns
    -------
    dict[int, int]
        Residue index -> ``pdbx_nonpoly_scheme`` row index.

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
            residues.chain_id,
            residues.res_id,
            residues.ins_code,
            residues.res_name,
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
    """Number one new entity per residue name, after the deposit's highest entity id.

    Parameters
    ----------
    reference_block : CIFBlock
        Deposited block with ``entity``.
    residue_names : list[str]
        Names of residues the deposit does not list.

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
    """Annotate a model with deposited ids through the input it was matched to.

    The input's residues are placed in the deposit (``resolve_deposit_positions``), each model
    residue with a matched atom takes its input residue's position, and the residues the model
    built are placed from their neighbors along the deposited chain
    (``_walk_deposited_positions``).

    Parameters
    ----------
    model_atom_array : AtomArray
        Model topology, ``n_model`` atoms.
    input_atom_array : AtomArray
        Input topology the model was matched to, ``n_input`` atoms.
    model_indices, input_indices : np.ndarray
        Matched atoms, ``(n_common,)`` indices into each array (``AtomReconciler``).
    deposit : Deposit
        The deposit the input came from.

    Returns
    -------
    AtomArray
        Annotated copy of ``model_atom_array`` (``annotate_deposit_ids``).

    Raises
    ------
    ValueError
        If the input cannot be placed, or a model residue cannot.

    Notes
    -----
    A model residue whose atoms match two input residues takes the position of the first,
    without checking that the two agree; only the one-residue-per-position check
    (``_one_residue_per_position``) would catch a resulting clash.
    """
    input_positions = resolve_deposit_positions(input_atom_array, deposit)
    model_residue = get_residue_positions(model_atom_array, model_indices)  # (n_common,)
    input_residue = get_residue_positions(input_atom_array, input_indices)  # (n_common,)
    known: list[PolySeqSchemePosition | None] = [None] * len(get_residue_starts(model_atom_array))
    for residue, matched in zip(model_residue, input_residue, strict=True):
        # A model residue matched to atoms of two input residues keeps the first one's position.
        known[residue] = known[residue] or input_positions[matched]
    positions = _walk_deposited_positions(model_atom_array, known, deposit)
    return annotate_deposit_ids(model_atom_array, deposit, positions)


def annotate_deposit_ids(
    atom_array: _Structure,
    deposit: Deposit,
    positions: list[PolySeqSchemePosition | None] | None = None,
) -> _Structure:
    """Return a copy of ``atom_array`` carrying each residue's deposited ids.

    A polymer residue takes its position's author ids as ``chain_id``, ``res_id`` and
    ``ins_code``, and its label ids as the ``DEPOSIT_LABEL_ANNOTATIONS``. A ligand or water the
    deposit's ``pdbx_nonpoly_scheme`` lists (``_match_deposited_ligands``) takes that row's ids
    the same way, and its row as ``DEPOSIT_NONPOLY_ROW``. Any other residue keeps its ids under
    a new entity per residue name (``_new_entity_ids``), so it cannot claim a deposited entity.
    Residues outside the polymer have no sequence position (``.``).

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to annotate.
    deposit : Deposit
        The deposit to place them in.
    positions : list[PolySeqSchemePosition | None] | None
        Deposited position per residue; found with ``resolve_deposit_positions`` when omitted.

    Returns
    -------
    AtomArray | AtomArrayStack
        Annotated copy.

    Raises
    ------
    ValueError
        If ``positions`` is omitted and the residues cannot be placed, or a ligand matches
        several deposited ones.
    """
    if positions is None:
        positions = resolve_deposit_positions(atom_array, deposit)
    reference_block = deposit.block
    residues = _residues(atom_array, deposit)
    nonpoly = deposit.nonpoly
    ligands = _match_deposited_ligands(residues, positions, nonpoly)
    new_entity_ids = _new_entity_ids(
        reference_block,
        [
            str(name)
            for residue, (position, name) in enumerate(zip(positions, residues.res_name))
            if position is None and residue not in ligands
        ],
    )
    # (author chain, number, insertion code), (label asym, seq, entity) per residue
    residue_ids: list[tuple[tuple[str, int, str], _Label]] = []
    for residue, (position, chain, number, code, name) in enumerate(
        zip(
            positions,
            residues.chain_id,
            residues.res_id,
            residues.ins_code,
            residues.res_name,
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
    annotated.set_annotation(DEPOSIT_NONPOLY_ROW, nonpoly_rows[residue_of_atom])
    return annotated


def _carried_categories(
    reference_block: CIFBlock,
    polymer_labels: set[_Label],
    nonpoly: dict[str, Any],
    ligand_rows: list[int],
) -> dict[str, dict[str, list[str]]]:
    """Select the deposit's category rows for the polymer chains and ligands present.

    ``entity``, ``entity_poly`` and ``entity_poly_seq`` rows are kept for the entities in
    ``polymer_labels``, and ``struct_asym`` and ``pdbx_poly_seq_scheme`` rows for its chains;
    ``entity`` and ``struct_asym`` also for the ligands, whose ``pdbx_nonpoly_scheme`` rows are
    kept too. The scheme's ``auth_seq_num`` and ``auth_mon_id`` are filled at modeled
    positions and ``?`` elsewhere.

    Parameters
    ----------
    reference_block : CIFBlock
        Deposited block with the polymer entity categories.
    polymer_labels : set[_Label]
        Deposited labels of the output's polymer residues.
    nonpoly : dict[str, Any]
        Deposited ``pdbx_nonpoly_scheme`` columns, empty when the deposit has none.
    ligand_rows : list[int]
        Sorted ``pdbx_nonpoly_scheme`` rows of the output's deposited ligands.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Category name -> carried columns.
    """
    asym_ids = {asym_id for asym_id, _, _ in polymer_labels}
    entity_ids = {entity_id for _, _, entity_id in polymer_labels}
    ligand_asym_ids = {str(nonpoly["asym_id"][row]) for row in ligand_rows}
    ligand_entity_ids = {str(nonpoly["entity_id"][row]) for row in ligand_rows}
    categories = {
        name: _concatenate_category_rows(
            [
                _select_category_rows(reference_block[name], column, value)
                for value in sorted(ids_to_keep)
            ]
        )
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
    """Add each modeled monomer that differs from the deposit's as a heterogeneity row.

    The alignment accepts a canonical parent for a modified residue (MET for MSE, CYS for
    CSO), but ``atom_site`` is a child of ``entity_poly_seq`` on ``(entity_id, num,
    mon_id)``, so a modeled MET at a deposited MSE position has no parent row. The mmCIF
    dictionary's form for two monomers at one position (microheterogeneity, e.g. a CYS/CSO
    site) is two rows sharing ``num`` with ``_entity_poly_seq.hetero`` set, and that is what
    is written: the deposit's row is kept, its flag set, and the modeled name inserted after
    it. On 1VME position 13::

        entity_id  num  mon_id  hetero          entity_id  num  mon_id  hetero
        1          13   MSE     n        ->     1          13   MSE     y
                                                1          13   MET     y

    Side effect: ``entity_poly_seq`` is rewritten in place; a ``hetero`` column is created
    when the deposit has none. Nothing changes when every modeled name is the deposit's.

    Parameters
    ----------
    entity_poly_seq : dict[str, list[str]]
        Carried ``entity_poly_seq`` columns.
    atom_site : CIFCategory
        Output ``atom_site``.
    labels : list[_Label | None]
        Deposit label per ``atom_site`` row, ``None`` for rows outside the polymer entities.
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
    """Collect a ``chem_comp`` row for each component, from the deposit or else the CCD.

    A component the deposit does not list (MET beside a deposited MSE) takes its row from
    biotite's bundled CCD, in the deposit's columns; a column the CCD lacks is ``?``. One the
    CCD lacks too is warned about and left out.

    Parameters
    ----------
    chem_comp : CIFCategory
        Deposited ``chem_comp``.
    components : set[str]
        Monomer names that need a row.

    Returns
    -------
    dict[str, list[str]]
        ``chem_comp`` columns: the deposit's rows for ``components``, then CCD rows.
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
    """Map ``struct_conn`` partners to their atoms' deposited label ids.

    ``set_structure`` names partners by the ``atom_site`` label ids it wrote, so each partner
    is found by those: ``(label_asym_id, label_seq_id, insertion code, label_comp_id)``.

    Parameters
    ----------
    struct_conn : CIFCategory
        ``struct_conn`` as ``set_structure`` wrote it.
    relabel : dict[tuple[str, str, str, str], tuple[str, str]]
        Written residue key -> deposited ``(label_asym_id, label_seq_id)``.

    Returns
    -------
    dict[str, np.ndarray]
        New values of each partner's ``label_asym_id`` and ``label_seq_id`` columns.
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
    """Write a structure annotated by ``annotate_deposit_ids`` with the deposit's metadata.

    ``set_structure`` writes the atoms; the label chain, sequence position and entity then come
    from the annotations, ``struct_conn`` partners follow them, and the deposit's polymer
    categories are carried for the chains and entities present (``_carried_categories``).
    Where the model writes the canonical parent of a deposited modified residue (MET for
    MSE), the modeled monomer is added to ``entity_poly_seq`` (``_add_modeled_monomers``), and
    ``chem_comp`` rows are carried for every monomer named (``_carried_chem_comp``).

    Parameters
    ----------
    cif_file : CIFFile
        Output CIF. Side effect: the structure and carried categories are written into it.
    atom_array : AtomArray | AtomArrayStack
        Structure annotated by ``annotate_deposit_ids``.
    deposit : Deposit
        The deposit the annotations came from.

    Returns
    -------
    tuple[str, ...]
        Names of the carried categories.

    Raises
    ------
    ValueError
        If ``atom_array`` lacks the annotations or has no deposited polymer residue.
    """
    missing = {*DEPOSIT_LABEL_ANNOTATIONS, DEPOSIT_NONPOLY_ROW} - set(
        atom_array.get_annotation_categories()
    )
    if missing:
        raise ValueError(f"Atom array lacks deposit annotations {sorted(missing)}")
    reference_block = deposit.block
    set_structure(cif_file, atom_array)
    block = _single_block(cif_file)
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

    ligand_rows = sorted(set(atom_array.get_annotation(DEPOSIT_NONPOLY_ROW).tolist()) - {-1})

    # Select the deposit's rows for the chains present, plus the monomers the model adds.
    categories = _carried_categories(reference_block, polymer_labels, deposit.nonpoly, ligand_rows)
    _add_modeled_monomers(categories[_ENTITY_POLY_SEQ], atom_site, labels)
    if _CHEM_COMP in reference_block:
        components = set(atom_site["label_comp_id"].as_array(str))
        components |= set(categories[_ENTITY_POLY_SEQ]["mon_id"])
        categories[_CHEM_COMP] = _carried_chem_comp(reference_block[_CHEM_COMP], components)

    # Point struct_conn partners, written under set_structure's label ids, at the deposit's.
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
