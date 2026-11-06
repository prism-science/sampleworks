"""Write model outputs with the chain, residue, and entity ids of the PDB entry they came from.

Model wrappers often renumber residues and drop entity metadata, so an output written as-is
cannot be compared residue by residue with the deposited mmCIF (the entry as deposited in the
PDB). This module finds each residue in the deposited mmCIF and writes the output with the
deposited label ids (``label_asym_id``, ``label_seq_id`` and ``label_entity_id``, the PDB
numbering of chains, sequence positions and entities) and the categories that describe its
polymer chains.
"""

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
# The dictionary's value for an item that does not apply, e.g. label_seq_id of a ligand atom.
_INAPPLICABLE = "."
# entity_poly_seq.hetero marks positions with more than one monomer; absent, it defaults to no.
_HETERO = "hetero"
_HETERO_YES = "y"
_HETERO_NO = "n"

_POLYMER_ENTITY_CATEGORIES = (_ENTITY, _ENTITY_POLY, _ENTITY_POLY_SEQ)
# Looking residues up by author id also needs these; without them only sequence alignment can
# place residues.
_SCHEME_CATEGORIES = (_STRUCT_ASYM, _PDBX_POLY_SEQ_SCHEME)


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


def _matching_entity_and_poly_seq_num(
    reference_block: CIFBlock, residue_names: list[str]
) -> tuple[str, list[str]]:
    """Find the deposited entity that contains a modeled sequence, and where each residue sits.

    Parameters
    ----------
    reference_block : CIFBlock
        Block of the deposited mmCIF, with ``entity_poly`` and ``entity_poly_seq``.
    residue_names : list[str]
        Names of the polymer residues of one output entity, sorted by residue number.

    Returns
    -------
    tuple[str, list[str]]
        Id of the matching deposited entity, and the ``entity_poly_seq.num`` each residue
        aligns to.

    Raises
    ------
    ValueError
        If ``residue_names`` is empty, no deposited entity contains the sequence, several do,
        or the sequence fits one entity in several equally good ways.
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
        # Output residue numbers may not follow the deposited numbering, so align by name only.
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


@dataclass(frozen=True)
class Deposit:
    """A deposited mmCIF, read once and reused to label many output structures.

    Attributes
    ----------
    block : CIFBlock
        Data block of the deposited mmCIF; its polymer categories are copied into outputs.
    monomers : frozenset[str]
        Canonical names of the monomers listed in ``entity_poly_seq``, used to tell modified
        residues from ligands.
    """

    block: CIFBlock
    monomers: frozenset[str]

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
            If the file has no block or several blocks, or lacks ``entity``, ``entity_poly`` or
            ``entity_poly_seq``.
        """
        block = reference.block
        missing = [name for name in _POLYMER_ENTITY_CATEGORIES if name not in block]
        if missing:
            raise ValueError(f"Reference CIF lacks required categories: {missing}")
        monomers = frozenset(
            _canonical_monomer(str(name))
            for name in block[_ENTITY_POLY_SEQ]["mon_id"].as_array(str)
        )
        return cls(block, monomers)


_Label = tuple[str, str, str]  # deposited (label_asym_id, label_seq_id, label_entity_id)

# Deposited label ids of each atom; the deposit_ prefix keeps them apart from label_*
# annotations that readers load from an input file.
DEPOSIT_LABEL_ANNOTATIONS = (
    "deposit_label_asym_id",
    "deposit_label_seq_id",
    "deposit_label_entity_id",
)


_Structure = TypeVar("_Structure", AtomArray, AtomArrayStack)


class _Residues(NamedTuple):
    """Annotations of each residue in an atom array, used to find it in the deposited mmCIF.

    Every field has shape ``(n_residues,)``.

    Attributes
    ----------
    chain_id, res_id, ins_code, res_name : np.ndarray
        Residue annotations, with ``res_id`` as strings.
    hetero : np.ndarray
        Whether the residue is written as ``HETATM``.
    polymer : np.ndarray
        Whether the residue belongs to a polymer chain (see ``_residues``).
    entity_id : np.ndarray
        Entity id that ``set_structure`` writes: the ``chain_entity`` annotation when present
        (atomworks patches ``set_structure`` to use it), otherwise one entity per chain.
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
    """Collect the annotations of each residue and decide which residues are polymer.

    A residue is polymer when it is written as ``ATOM``, or when it is a ``HETATM`` whose
    canonical name is listed in the deposited ``entity_poly_seq`` (a modified residue such as
    MSE). Membership is decided by name, not residue number, because ligand numbers can fall
    inside the sequence range of a chain.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to describe.
    deposit : Deposit
        Deposited mmCIF the residues will be placed in.

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
    """Find deposited label ids by reading residue ids as deposited author ids.

    Works when the output keeps the author numbering of the deposited mmCIF, as structures read
    from it do: ``chain_id``, ``res_id`` and ``ins_code`` are looked up as ``pdb_strand_id``,
    ``pdb_seq_num`` and ``pdb_ins_code`` in ``pdbx_poly_seq_scheme``. A polymer ``HETATM``
    residue with no row there is treated as a ligand, since a free amino acid (a GLU ligand
    beside a chain containing GLU) looks like a modified residue by name.

    Parameters
    ----------
    residues : _Residues
        Residues to label.
    reference_block : CIFBlock
        Block of the deposited mmCIF, with ``pdbx_poly_seq_scheme`` and ``struct_asym``.

    Returns
    -------
    list[_Label | None]
        Deposited label ids of each residue, ``None`` for ligands and waters.

    Raises
    ------
    ValueError
        If ``pdbx_poly_seq_scheme`` or ``struct_asym`` is missing, an ``ATOM`` residue has no
        row in ``pdbx_poly_seq_scheme``, a polymer residue name differs from the deposited
        name at that position, or no residue is labeled.
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
    """Find deposited label ids by aligning each output entity sequence to the deposited ones.

    Used when residues are not numbered as in the deposited mmCIF, e.g. after a model wrapper
    renumbers them in ``featurize``. All chains of one output entity share one alignment, so
    their residues are pooled by residue number and insertion code. Each residue takes the
    deposited ``entity_poly_seq.num`` it aligns to as ``label_seq_id`` and the matching
    deposited entity id as ``label_entity_id``; ``label_asym_id`` stays the output chain id.

    Parameters
    ----------
    residues : _Residues
        Residues to label.
    reference_block : CIFBlock
        Block of the deposited mmCIF, with ``entity_poly`` and ``entity_poly_seq``.

    Returns
    -------
    list[_Label | None]
        Deposited label ids of each residue, ``None`` for ligands and waters.

    Raises
    ------
    ValueError
        If no residue is polymer, two residues of one entity share a number and insertion code
        but differ in name, or an entity does not fit exactly one deposited entity in exactly
        one way.
    """
    # (output entity, res_id, ins_code) -> (deposited entity, entity_poly_seq.num)
    remap: dict[tuple[str, str, str], tuple[str, str]] = {}
    for entity_id in dict.fromkeys(residues.entity_id):
        names: dict[tuple[int, str], str] = {}  # (res_id, ins_code) -> res_name
        for entity, number, ins_code, name, polymer in zip(
            residues.entity_id,
            residues.res_id,
            residues.ins_code,
            residues.res_name,
            residues.polymer,
            strict=True,
        ):
            if entity != entity_id or not polymer:
                continue
            key = (int(number), str(ins_code))
            if key in names and names[key] != name:
                raise ValueError(
                    f"Output entity {entity_id} has conflicting names for {number}{ins_code}"
                )
            names[key] = str(name)
        residue_keys = sorted(names)
        if not residue_keys:
            continue  # ligands and waters
        deposited_id, nums = _matching_entity_and_poly_seq_num(
            reference_block, [names[key] for key in residue_keys]
        )
        for (number, ins_code), num in zip(residue_keys, nums, strict=True):
            remap[(str(entity_id), str(number), ins_code)] = (deposited_id, num)
    if not remap:
        raise ValueError("Output has no polymer entities")
    labels: list[_Label | None] = []
    for chain, entity, number, ins_code, polymer in zip(
        residues.chain_id,
        residues.entity_id,
        residues.res_id,
        residues.ins_code,
        residues.polymer,
        strict=True,
    ):
        match = remap.get((str(entity), str(number), str(ins_code))) if polymer else None
        labels.append(None if match is None else (str(chain), match[1], match[0]))
    return labels


def resolve_deposit_labels(
    atom_array: AtomArray | AtomArrayStack, deposit: Deposit
) -> list[_Label | None]:
    """Find where each residue of a structure sits in its deposited mmCIF.

    This maps model output, which may be renumbered, onto the chain, sequence position and
    entity ids of the PDB entry. Residues are first looked up by their ids as deposited author
    ids (``_labels_by_author_ids``). If that fails for any reason, including a residue whose
    name differs from the deposited one at the same number, each output entity sequence is
    aligned to the deposited sequences instead (``_labels_by_alignment``).

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to place.
    deposit : Deposit
        Deposited mmCIF to place it in.

    Returns
    -------
    list[_Label | None]
        Deposited ``(label_asym_id, label_seq_id, label_entity_id)`` of each residue, ``None``
        for ligands and waters.

    Raises
    ------
    ValueError
        If neither method places every polymer residue; the message gives both reasons.
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
    """Return a copy of a structure annotated with the deposited label ids of every atom.

    Placing residues can take a sequence alignment, so annotate once and reuse the copy for
    every frame written with ``set_deposited_structure``. The ids are stored in the
    ``DEPOSIT_LABEL_ANNOTATIONS``. Ligands and waters keep their chain id, have no sequence
    position (``.``), and keep the entity id that ``set_structure`` writes, which is not
    necessarily a deposited entity id.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack
        Structure to annotate.
    deposit : Deposit
        Deposited mmCIF to place it in.
    labels : list[_Label | None] | None
        Deposited label ids of each residue, from ``resolve_deposit_labels``; resolved here
        when omitted.

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
    """Select the deposited rows that describe the polymer chains and entities in an output.

    Rows of ``entity``, ``entity_poly`` and ``entity_poly_seq`` are kept for the entities in
    ``polymer_labels``. With ``with_scheme``, rows of ``struct_asym`` and
    ``pdbx_poly_seq_scheme`` are kept for the chains in ``polymer_labels``, and the scheme
    columns ``auth_seq_num`` and ``auth_mon_id`` are ``?`` at positions the output does not
    model. Without it, ``struct_asym`` is rebuilt to list each polymer chain of the output
    with its entity.

    Parameters
    ----------
    reference_block : CIFBlock
        Block of the deposited mmCIF.
    polymer_labels : set[_Label]
        Deposited label ids of the polymer residues in the output.
    with_scheme : bool
        Whether the output keeps the deposited chains and author ids, so the deposited
        ``pdbx_poly_seq_scheme`` still describes it.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Category name -> carried columns.
    """
    asym_ids = {asym_id for asym_id, _, _ in polymer_labels}
    entity_ids = {entity_id for _, _, entity_id in polymer_labels}
    categories = {
        name: _rows_where(reference_block[name], column, ids_to_keep)
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
        categories[name] = _rows_where(reference_block[name], column, asym_ids)
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


def _keeps_deposited_author_ids(
    atom_site: CIFCategory, labels: list[_Label | None], reference_block: CIFBlock
) -> bool:
    """Check whether every polymer residue was written with its deposited author ids.

    If so, the deposited ``pdbx_poly_seq_scheme``, which maps label ids to author ids, still
    describes the output and can be copied into it.

    Parameters
    ----------
    atom_site : CIFCategory
        Output ``atom_site``.
    labels : list[_Label | None]
        Deposited label ids of each ``atom_site`` row, ``None`` for ligand and water atoms.
    reference_block : CIFBlock
        Block of the deposited mmCIF.

    Returns
    -------
    bool
        ``True`` when the author ids written for every polymer atom are the ones
        ``pdbx_poly_seq_scheme`` lists for its label ids. ``False`` otherwise, or when the
        deposited mmCIF lacks ``struct_asym`` or ``pdbx_poly_seq_scheme``.
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
    """Write a structure to a CIF with the chain, residue and entity ids of its PDB entry.

    The written file can be compared residue by residue with the deposited mmCIF and carries
    the polymer metadata of the entry. ``set_structure`` writes the atoms of a structure
    annotated by ``annotate_deposit_ids``; then

    - ``label_asym_id``, ``label_seq_id`` and ``label_entity_id`` in ``atom_site`` are set to
      the deposited ids, and ``struct_conn`` bond partners are updated to match;
    - the deposited rows for the chains and entities present are copied
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
        If ``atom_array`` lacks the ``DEPOSIT_LABEL_ANNOTATIONS`` or has no deposited polymer
        residue.
    KeyError
        If a ``struct_conn`` bond partner is not a written residue.
    """
    missing = set(DEPOSIT_LABEL_ANNOTATIONS) - set(atom_array.get_annotation_categories())
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

    # Copy the deposited rows for the chains present, plus monomers the model adds.
    with_scheme = _keeps_deposited_author_ids(atom_site, labels, reference_block)
    categories = _carried_categories(reference_block, polymer_labels, with_scheme)
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
