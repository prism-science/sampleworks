"""Place structures in their deposited mmCIF's numbering and carry its polymer categories."""

from dataclasses import dataclass
from typing import NamedTuple, Self, TypeVar

import numpy as np
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
# The dictionary's value for an item that does not apply, e.g. label_seq_id of a ligand atom.
_INAPPLICABLE = "."
# entity_poly_seq.hetero marks positions with more than one monomer; absent, it defaults to no.
_HETERO = "hetero"
_HETERO_YES = "y"
_HETERO_NO = "n"

_POLYMER_ENTITY_CATEGORIES = (_ENTITY, _ENTITY_POLY, _ENTITY_POLY_SEQ)
# The author-number lookup additionally needs these; without them the carry falls back to
# sequence alignment.
_SCHEME_CATEGORIES = (_STRUCT_ASYM, _PDBX_POLY_SEQ_SCHEME)


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


@dataclass(frozen=True)
class Deposit:
    """A deposited mmCIF, parsed once to place structures in it.

    Attributes
    ----------
    block : CIFBlock
        The deposit's data block, whose categories are carried.
    monomers : frozenset[str]
        Canonical names of the monomers its ``entity_poly_seq`` lists.
    """

    block: CIFBlock
    monomers: frozenset[str]

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
        return cls(block, monomers)


_Label = tuple[str, str, str]  # deposited (label_asym_id, label_seq_id, label_entity_id)

# Deposited label ids, written over the label columns set_structure fills from chain_id and
# res_id. Named apart from the label_* fields readers can load as annotations.
DEPOSIT_LABEL_ANNOTATIONS = (
    "deposit_label_asym_id",
    "deposit_label_seq_id",
    "deposit_label_entity_id",
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
        Whether the residue belongs to a polymer chain (``_polymer_residues``).
    entity_id : np.ndarray
        Entity id ``set_structure`` would write. Under atomworks' patch of it, that is the
        ``chain_entity`` annotation, or one entity per chain.
    """

    chain_id: np.ndarray
    res_id: np.ndarray
    ins_code: np.ndarray
    res_name: np.ndarray
    hetero: np.ndarray
    polymer: np.ndarray
    entity_id: np.ndarray


def _chain_entity_ids(chain_ids: np.ndarray) -> np.ndarray:
    """Number entities one per chain, in order of appearance, as ``set_structure`` does."""
    numbers = {chain: str(number) for number, chain in enumerate(dict.fromkeys(chain_ids), 1)}
    return np.array([numbers[chain] for chain in chain_ids])


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
    entity_ids = (
        atom_array.chain_entity.astype(str)
        if "chain_entity" in atom_array.get_annotation_categories()
        else _chain_entity_ids(atom_array.chain_id)
    )
    return _Residues(
        chain_id=atom_array.chain_id[starts],
        res_id=atom_array.res_id[starts].astype(str),
        ins_code=atom_array.ins_code[starts],
        res_name=res_name,
        hetero=hetero,
        polymer=~hetero | listed,
        entity_id=entity_ids[starts],
    )


def _labels_by_author_ids(residues: _Residues, reference_block: CIFBlock) -> list[_Label | None]:
    """Label residues by looking their ids up as author ids in the deposit's scheme.

    Parameters
    ----------
    residues : _Residues
        Residues whose ``chain_id``, ``res_id`` and ``ins_code`` are the deposit's
        ``pdb_strand_id``, ``pdb_seq_num`` and ``pdb_ins_code``.
    reference_block : CIFBlock
        Deposited block with ``pdbx_poly_seq_scheme`` and ``struct_asym``.

    Returns
    -------
    list[_Label | None]
        Deposited label per residue, ``None`` for residues outside the scheme (e.g. ligands).

    Raises
    ------
    ValueError
        If scheme categories are absent, a non-``HETATM`` residue has no scheme row, a
        residue's name disagrees with its scheme row, or no residue is labeled.
    """
    missing = [name for name in _SCHEME_CATEGORIES if name not in reference_block]
    if missing:
        raise ValueError(f"Reference CIF lacks {missing}")
    scheme = reference_block[_PDBX_POLY_SEQ_SCHEME]
    scheme_keys = zip(
        scheme["pdb_strand_id"].as_array(str),
        scheme["pdb_seq_num"].as_array(str),
        _blank_to_empty(scheme["pdb_ins_code"].as_array(str)),
    )
    scheme_labels = zip(
        scheme["asym_id"].as_array(str),
        scheme["seq_id"].as_array(str),
        scheme["entity_id"].as_array(str),
    )
    lookup: dict[tuple[str, str, str], _Label] = {}
    monomers: dict[tuple[str, str, str], set[str]] = {}
    for key, label, monomer in zip(scheme_keys, scheme_labels, scheme["mon_id"].as_array(str)):
        lookup.setdefault(key, label)  # microheterogeneity rows (e.g. CYS/CSO) share a seq_id
        monomers.setdefault(key, set()).add(_canonical_monomer(monomer))

    keys = list(zip(residues.chain_id, residues.res_id, residues.ins_code, strict=True))
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
        if polymer and key in lookup and _canonical_monomer(name) not in monomers[key]
    }
    if renamed:
        raise ValueError(f"Residue names disagree with the deposit, e.g. {sorted(renamed)[:3]}")
    labels = [lookup.get(key) if polymer else None for key, polymer in zip(keys, residues.polymer)]
    if not any(labels):
        raise ValueError("Output has no polymer residues")
    return labels


def _labels_by_alignment(residues: _Residues, reference_block: CIFBlock) -> list[_Label | None]:
    """Label residues by aligning each entity's sequence to the deposit's.

    Used when the residues do not carry the deposit's author numbering (a model that
    renumbers residues). Each residue gets the deposited ``entity_poly_seq.num`` it aligns to
    as its ``label_seq_id`` and the deposited entity id; its ``label_asym_id`` stays its chain.

    Parameters
    ----------
    residues : _Residues
        Residues to label.
    reference_block : CIFBlock
        Deposited block with ``entity``, ``entity_poly`` and ``entity_poly_seq``.

    Returns
    -------
    list[_Label | None]
        Deposited label per residue, ``None`` for non-polymer residues.

    Raises
    ------
    ValueError
        If no entity has polymer residues, an entity has conflicting names for one number,
        or an entity has no unique match among the deposited entities.
    """
    remap: dict[tuple[str, str], tuple[str, str]] = {}  # (entity, number) -> (entity, num)
    for entity_id in dict.fromkeys(residues.entity_id):
        names: dict[int, str] = {}
        for entity, number, name, polymer in zip(
            residues.entity_id, residues.res_id, residues.res_name, residues.polymer, strict=True
        ):
            if entity != entity_id or not polymer:
                continue
            if int(number) in names and names[int(number)] != name:
                raise ValueError(f"Output entity {entity_id} has conflicting names for {number}")
            names[int(number)] = str(name)
        numbers = sorted(names)
        if not numbers:
            continue  # ligands and waters
        deposited_id, nums = _matching_entity(reference_block, [names[n] for n in numbers])
        for number, num in zip(numbers, nums, strict=True):
            remap[(str(entity_id), str(number))] = (deposited_id, num)
    if not remap:
        raise ValueError("Output has no polymer entities")
    labels: list[_Label | None] = []
    for chain, entity, number, polymer in zip(
        residues.chain_id, residues.entity_id, residues.res_id, residues.polymer, strict=True
    ):
        match = remap.get((str(entity), str(number))) if polymer else None
        labels.append(None if match is None else (str(chain), match[1], match[0]))
    return labels


def resolve_deposit_labels(
    atom_array: AtomArray | AtomArrayStack, deposit: Deposit
) -> list[_Label | None]:
    """Find each residue's deposited label.

    Residues are looked up by their ids as the deposit's author ids, or, failing that, each
    entity's sequence is aligned to the deposit's (``_labels_by_alignment``).

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to place.
    deposit : Deposit
        The deposit to place them in.

    Returns
    -------
    list[_Label | None]
        Deposited label per residue, ``None`` for non-polymer residues.

    Raises
    ------
    ValueError
        If neither match places every polymer residue.
    """
    reference_block = deposit.block
    residues = _residues(atom_array, deposit)
    try:
        return _labels_by_author_ids(residues, reference_block)
    except ValueError as lookup_error:
        try:
            return _labels_by_alignment(residues, reference_block)
        except ValueError as alignment_error:
            raise ValueError(
                f"author lookup: {lookup_error}; sequence alignment: {alignment_error}"
            ) from alignment_error


def annotate_deposit_ids(
    atom_array: _Structure,
    deposit: Deposit,
    labels: list[_Label | None] | None = None,
) -> _Structure:
    """Return a copy of ``atom_array`` annotated with each residue's deposited label ids.

    Sets the ``DEPOSIT_LABEL_ANNOTATIONS``. A residue without a deposited label (a ligand or
    water) keeps its chain and entity and has no sequence position (``.``).

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to annotate.
    deposit : Deposit
        The deposit to place it in.
    labels : list[_Label | None] | None
        Deposited label per residue; found with ``resolve_deposit_labels`` when omitted.

    Returns
    -------
    AtomArray | AtomArrayStack
        Annotated copy.

    Raises
    ------
    ValueError
        If ``labels`` is omitted and the residues cannot be placed.
    """
    if labels is None:
        labels = resolve_deposit_labels(atom_array, deposit)
    residues = _residues(atom_array, deposit)
    residue_ids = [
        label if label is not None else (str(chain), _INAPPLICABLE, str(entity))
        for label, chain, entity in zip(labels, residues.chain_id, residues.entity_id, strict=True)
    ]
    residue_of_atom = get_residue_positions(atom_array, np.arange(atom_array.array_length()))
    annotated = atom_array.copy()
    for index, name in enumerate(DEPOSIT_LABEL_ANNOTATIONS):
        values = np.array([ids[index] for ids in residue_ids], dtype=object)
        annotated.set_annotation(name, values[residue_of_atom].astype(str))
    return annotated


def _carried_categories(
    reference_block: CIFBlock, polymer_labels: set[_Label], with_scheme: bool
) -> dict[str, dict[str, list[str]]]:
    """Select the deposit's category rows for the polymer chains and entities present.

    ``entity``, ``entity_poly`` and ``entity_poly_seq`` rows are kept for the entities in
    ``polymer_labels``. With ``with_scheme``, the chains' ``struct_asym`` and
    ``pdbx_poly_seq_scheme`` rows are kept too, the scheme's ``auth_seq_num`` and
    ``auth_mon_id`` filled at modeled positions and ``?`` elsewhere; otherwise ``struct_asym``
    lists the chains present with their entities.

    Parameters
    ----------
    reference_block : CIFBlock
        Deposited block with the polymer entity categories.
    polymer_labels : set[_Label]
        Deposited labels of the output's polymer residues.
    with_scheme : bool
        Whether the output's chains are the deposit's and keep its author ids.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Category name -> carried columns.
    """
    asym_ids = {asym_id for asym_id, _, _ in polymer_labels}
    entity_ids = {entity_id for _, _, entity_id in polymer_labels}
    categories = {
        name: _concatenate_category_rows(
            [
                _select_category_rows(reference_block[name], column, value)
                for value in sorted(ids_to_keep)
            ]
        )
        for name, column, ids_to_keep in (
            (_ENTITY, "id", entity_ids),
            (_ENTITY_POLY, "entity_id", entity_ids),
            (_ENTITY_POLY_SEQ, "entity_id", entity_ids),
        )
    }
    if not with_scheme:
        chain_entities = {asym_id: entity_id for asym_id, _, entity_id in polymer_labels}
        categories[_STRUCT_ASYM] = {
            "id": sorted(chain_entities),
            "entity_id": [chain_entities[chain] for chain in sorted(chain_entities)],
        }
        return categories
    for name, column in ((_STRUCT_ASYM, "id"), (_PDBX_POLY_SEQ_SCHEME, "asym_id")):
        categories[name] = _concatenate_category_rows(
            [
                _select_category_rows(reference_block[name], column, value)
                for value in sorted(asym_ids)
            ]
        )
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


def _keeps_deposited_author_ids(
    atom_site: CIFCategory, labels: list[_Label | None], reference_block: CIFBlock
) -> bool:
    """Check that every labeled row's author ids are its deposited scheme row's.

    Parameters
    ----------
    atom_site : CIFCategory
        Output ``atom_site``.
    labels : list[_Label | None]
        Deposited label per row, ``None`` for non-polymer rows.
    reference_block : CIFBlock
        Deposited block.

    Returns
    -------
    bool
        ``False`` also when the deposit lacks ``struct_asym`` or ``pdbx_poly_seq_scheme``.
    """
    if not all(name in reference_block for name in _SCHEME_CATEGORIES):
        return False
    scheme = reference_block[_PDBX_POLY_SEQ_SCHEME]
    deposited = dict(
        zip(
            zip(scheme["asym_id"].as_array(str), scheme["seq_id"].as_array(str), strict=True),
            zip(
                scheme["pdb_strand_id"].as_array(str),
                scheme["pdb_seq_num"].as_array(str),
                _blank_to_empty(scheme["pdb_ins_code"].as_array(str)),
                strict=True,
            ),
            strict=True,
        )
    )
    written = zip(
        atom_site["auth_asym_id"].as_array(str),
        atom_site["auth_seq_id"].as_array(str),
        _blank_to_empty(atom_site["pdbx_PDB_ins_code"].as_array(str)),
        strict=True,
    )
    return all(
        label is None or deposited.get(label[:2]) == author_ids
        for label, author_ids in zip(labels, written, strict=True)
    )


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
        Structure with the ``DEPOSIT_LABEL_ANNOTATIONS``.
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
    missing = set(DEPOSIT_LABEL_ANNOTATIONS) - set(atom_array.get_annotation_categories())
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

    # Select the deposit's rows for the chains present, plus the monomers the model adds.
    with_scheme = _keeps_deposited_author_ids(atom_site, labels, reference_block)
    categories = _carried_categories(reference_block, polymer_labels, with_scheme)
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
