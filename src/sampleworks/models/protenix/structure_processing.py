import copy
import functools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from atomworks.enums import ChainType
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from biotite.structure import AtomArray, BondType, get_residue_starts
from jaxtyping import Bool, Int
from loguru import logger
from protenix.data.constants import STD_RESIDUES

from sampleworks.utils.sequence import annotate_sequence_positions
from sampleworks.utils.structure_utils import get_valid_atom_mask


POLYMER_ENTITY_TYPES = {
    ChainType.POLYPEPTIDE_L: "proteinChain",
    ChainType.POLYPEPTIDE_D: "proteinChain",
    ChainType.DNA: "dnaSequence",
    ChainType.RNA: "rnaSequence",
}

# Only these orders are written out; Protenix reads a bond without ``bondOrder`` as single.
EXPLICIT_BOND_ORDERS = (BondType.DOUBLE, BondType.TRIPLE)


class ChainEntity(NamedTuple):
    """Where a chain sits in the Protenix JSON.

    Attributes
    ----------
    entity_index : int
        1-based index of the chain's entry in the JSON ``sequences`` list.
    copy_id : int
        1-based index of the chain among the chains sharing that entry.
    """

    entity_index: int
    copy_id: int


def merge_covalent_bonds(covalent_bonds, all_entity_counts):
    """Merge covalent bonds with same entity and position.

    Parameters
    ----------
    covalent_bonds: list[dict]
        List of covalent bond dictionaries.
    all_entity_counts: dict[str, int]
        Mapping of entity_id to chain count.

    Returns
    -------
    list[dict]
        List of merged covalent bond dictionaries.
    """
    bonds_recorder = defaultdict(list)
    bonds_entity_counts = {}

    for bond_dict in covalent_bonds:
        bond_unique_string = [str(bond_dict.get("bondOrder", 1))]
        entity_counts = (
            all_entity_counts[str(bond_dict["entity1"])],
            all_entity_counts[str(bond_dict["entity2"])],
        )
        for i in range(2):
            for j in ["entity", "position", "atom"]:
                k = f"{j}{i + 1}"
                bond_unique_string.append(str(bond_dict[k]))
        bond_unique_string = "_".join(bond_unique_string)
        bonds_recorder[bond_unique_string].append(bond_dict)
        bonds_entity_counts[bond_unique_string] = entity_counts

    merged_covalent_bonds = []
    for k, v in bonds_recorder.items():
        counts1 = bonds_entity_counts[k][0]
        counts2 = bonds_entity_counts[k][1]
        if counts1 == counts2 == len(v):
            bond_dict_copy = copy.deepcopy(v[0])
            del bond_dict_copy["copy1"]
            del bond_dict_copy["copy2"]
            merged_covalent_bonds.append(bond_dict_copy)
        else:
            merged_covalent_bonds.extend(v)

    return merged_covalent_bonds


@functools.cache
def _build_ccd_suffix_map() -> dict[str, list[str]]:
    """Build a suffix to 5 character CCD code mapping.

    Returns
    -------
    dict[str, list[str]]
        Maps 2 character suffixes to lists of matching 5 character CCD codes.
    """
    from protenix.data.ccd import get_all_ccd_code

    suffix_map: dict[str, list[str]] = {}
    for code in get_all_ccd_code():
        if len(code) == 5:
            suffix_map.setdefault(code[-2:], []).append(code)
    return suffix_map


def _expand_tilde_ccd_code(truncated: str) -> str:
    """Expand a tilde-truncated PDB CCD code.

    PDB format truncates 5 char CCD codes to '~' + last 2 characters
    (e.g., A1AQS is ~QS). This function reverses the truncation by
    searching Protenix's CCD dictionary for matching codes.

    Parameters
    ----------
    truncated : str
        A tilde-prefixed CCD code (e.g., '~QS').

    Returns
    -------
    str
        The full CCD code if a unique match is found, otherwise the
        original truncated code.
    """
    suffix = truncated[1:]  # strip ~
    candidates = _build_ccd_suffix_map().get(suffix, [])

    if len(candidates) == 1:
        logger.info(f"Expanded tilde-truncated CCD code '{truncated}' → '{candidates[0]}'")
        return candidates[0]
    elif len(candidates) > 1:
        msg = (
            f"Tilde-truncated CCD code '{truncated}' is ambiguous, it maps to "
            f"{candidates}. Cannot auto-expand. Please use mmCIF format "
            "input, which preserves the full CCD code."
        )
        logger.error(msg)
        raise ValueError(msg)
    else:
        logger.error(
            f"Tilde-truncated CCD code '{truncated}' has no matching 5 character "
            f"CCD entry. Protenix may not be able to parse this "
            f"component. Please use mmCIF format input."
        )
        return truncated


def _residue_names(chain_atoms: AtomArray) -> list[str]:
    """Return the name of each residue in a chain, in order.

    Parameters
    ----------
    chain_atoms : AtomArray
        Atoms of a single chain.

    Returns
    -------
    list[str]
        One residue name per residue.
    """
    return np.asarray(chain_atoms.res_name)[get_residue_starts(chain_atoms)].tolist()


def _modifications(chain_atoms: AtomArray, entity_type: str) -> list[dict[str, Any]]:
    """List the non-standard residues of a polymer chain in Protenix's format.

    Parameters
    ----------
    chain_atoms : AtomArray
        Atoms of a single polymer chain, annotated with ``seq_idx``.
    entity_type : str
        ``"proteinChain"``, ``"dnaSequence"`` or ``"rnaSequence"``.

    Returns
    -------
    list[dict[str, Any]]
        One entry per observed non-standard residue, positioned 1-based in the deposited
        sequence.
    """
    residue_starts = get_residue_starts(chain_atoms)
    res_names = np.asarray(chain_atoms.res_name)[residue_starts]
    positions = np.asarray(chain_atoms.seq_idx)[residue_starts] + 1
    position_key, type_key = (
        ("ptmPosition", "ptmType")
        if entity_type == "proteinChain"
        else ("basePosition", "modificationType")
    )
    return [
        {position_key: int(position), type_key: f"CCD_{res_name}"}
        for position, res_name in zip(positions, res_names)
        if res_name not in STD_RESIDUES
    ]


def _entity_entry(chain_atoms: AtomArray, info: dict[str, Any]) -> tuple[str, dict] | None:
    """Describe one chain as a Protenix ``sequences`` entry, without its ``count``.

    Parameters
    ----------
    chain_atoms : AtomArray
        Resolved atoms of a single chain, annotated with ``seq_idx``.
    info : dict[str, Any]
        The chain's atomworks ``chain_info`` entry.

    Returns
    -------
    tuple[str, dict] | None
        Entity type and entity dict, or None for a polymer type Protenix does not take.
    """
    chain_type = info["chain_type"]
    if not chain_type.is_polymer():
        ccd_codes = [
            _expand_tilde_ccd_code(name) if name.startswith("~") else name
            for name in _residue_names(chain_atoms)
        ]
        return "ligand", {"ligand": f"CCD_{'_'.join(ccd_codes)}"}

    entity_type = POLYMER_ENTITY_TYPES.get(chain_type)
    if entity_type is None:
        return None
    entity_dict: dict[str, Any] = {"sequence": info["processed_entity_canonical_sequence"]}
    if entity_type == "proteinChain":
        entity_dict["msa"] = {}
    modifications = _modifications(chain_atoms, entity_type)
    if modifications:
        entity_dict["modifications"] = modifications
    return entity_type, entity_dict


def _group_chains_into_entities(
    atom_array: AtomArray, chain_info: dict[str, Any]
) -> tuple[list[dict], dict[str, ChainEntity]]:
    """Build the Protenix ``sequences`` list, one entry per set of identical chains.

    Parameters
    ----------
    atom_array : AtomArray
        Resolved atoms, annotated with ``seq_idx``.
    chain_info : dict[str, Any]
        Atomworks ``chain_info``; chains absent from it are skipped.

    Returns
    -------
    tuple[list[dict], dict[str, ChainEntity]]
        The ``sequences`` list in order of first appearance, and the entity and copy of
        every chain it covers.
    """
    entries: list[tuple[str, dict]] = []
    counts: list[int] = []
    chain_entities: dict[str, ChainEntity] = {}
    chain_ids = np.asarray(atom_array.chain_id)
    for chain_id in dict.fromkeys(chain_ids.tolist()):
        if chain_id not in chain_info:
            continue
        entry = _entity_entry(atom_array[chain_ids == chain_id], chain_info[chain_id])
        if entry is None:
            continue
        if entry not in entries:
            entries.append(entry)
            counts.append(0)
        entity_offset = entries.index(entry)
        counts[entity_offset] += 1
        chain_entities[chain_id] = ChainEntity(entity_offset + 1, counts[entity_offset])
    sequences = [
        {entity_type: {**entity_dict, "count": count}}
        for (entity_type, entity_dict), count in zip(entries, counts)
    ]
    return sequences, chain_entities


def _residue_positions(
    atom_array: AtomArray, is_polymer_atom: Bool[np.ndarray, " atoms"]
) -> Int[np.ndarray, " atoms"]:
    """Return each atom's 1-based residue position as Protenix bonds reference it.

    Parameters
    ----------
    atom_array : AtomArray
        Atoms annotated with ``seq_idx``.
    is_polymer_atom : np.ndarray
        Whether each atom belongs to a polymer chain.

    Returns
    -------
    np.ndarray
        Position in the deposited sequence for polymer atoms, residue ordinal within the
        chain for ligand atoms.
    """
    residue_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    residue_chain_ids = np.asarray(atom_array.chain_id)[residue_starts[:-1]]
    residue_ordinals = np.empty(len(residue_chain_ids), dtype=np.int64)
    for chain_id in np.unique(residue_chain_ids):
        in_chain = residue_chain_ids == chain_id
        residue_ordinals[in_chain] = np.arange(1, in_chain.sum() + 1)
    ligand_positions = np.repeat(residue_ordinals, np.diff(residue_starts))
    return np.where(is_polymer_atom, np.asarray(atom_array.seq_idx) + 1, ligand_positions)


def _select_covalent_bonds(
    atom_array: AtomArray,
    is_polymer_atom: Bool[np.ndarray, " atoms"],
    positions: Int[np.ndarray, " atoms"],
    has_modifications: bool,
) -> Int[np.ndarray, "bonds 3"]:
    """Pick the bonds Protenix cannot infer from its own residue templates.

    Ligand bonds to polymers and between ligand residues are always kept; single-atom
    chains (ions) are left to the model. Polymer bonds between chains or non-adjacent
    residues are kept only when some polymer carries a modification.

    Parameters
    ----------
    atom_array : AtomArray
        Atoms with a ``BondList``.
    is_polymer_atom : np.ndarray
        Whether each atom belongs to a polymer chain.
    positions : np.ndarray
        Each atom's residue position, from :func:`_residue_positions`.
    has_modifications : bool
        Whether any polymer entity has a modification.

    Returns
    -------
    np.ndarray
        Selected rows of ``atom_array.bonds.as_array()`` (atom_i, atom_j, bond_type).
    """
    bonds = atom_array.bonds.as_array()
    atom_i, atom_j = bonds[:, 0], bonds[:, 1]
    chain_ids = np.asarray(atom_array.chain_id)
    res_ids = np.asarray(atom_array.res_id)
    atoms_per_chain = Counter(chain_ids.tolist())
    is_ion = np.array([atoms_per_chain[chain_id] == 1 for chain_id in chain_ids], dtype=bool)
    is_ligand = ~is_polymer_atom & ~is_ion

    other_chain = chain_ids[atom_i] != chain_ids[atom_j]
    other_residue = other_chain | (res_ids[atom_i] != res_ids[atom_j])
    ligand_polymer = (is_ligand[atom_i] & is_polymer_atom[atom_j]) | (
        is_polymer_atom[atom_i] & is_ligand[atom_j]
    )
    keep = ligand_polymer | (is_ligand[atom_i] & is_ligand[atom_j] & other_residue)
    if has_modifications:
        non_adjacent = np.abs(positions[atom_i] - positions[atom_j]) > 1
        keep |= is_polymer_atom[atom_i] & is_polymer_atom[atom_j] & (other_chain | non_adjacent)
    return bonds[keep]


def _bond_to_json(
    bond: Int[np.ndarray, " 3"],
    atom_array: AtomArray,
    positions: Int[np.ndarray, " atoms"],
    chain_entities: dict[str, ChainEntity],
) -> dict[str, Any]:
    """Describe one bond by entity, residue position, atom name and copy.

    Parameters
    ----------
    bond : np.ndarray
        One row of a ``BondList`` array (atom_i, atom_j, bond_type).
    atom_array : AtomArray
        Atoms the bond indexes into.
    positions : np.ndarray
        Each atom's residue position, from :func:`_residue_positions`.
    chain_entities : dict[str, ChainEntity]
        Entity and copy of every chain.

    Returns
    -------
    dict[str, Any]
        Protenix ``covalent_bonds`` entry, with ``bondOrder`` only for double and triple
        bonds.
    """
    bond_dict: dict[str, Any] = {}
    for endpoint, atom_index in enumerate(bond[:2], start=1):
        chain_entity = chain_entities[str(atom_array.chain_id[atom_index])]
        bond_dict[f"entity{endpoint}"] = chain_entity.entity_index
        bond_dict[f"position{endpoint}"] = int(positions[atom_index])
        bond_dict[f"atom{endpoint}"] = str(atom_array.atom_name[atom_index])
        bond_dict[f"copy{endpoint}"] = chain_entity.copy_id
    if bond[2] in EXPLICIT_BOND_ORDERS:
        bond_dict["bondOrder"] = int(bond[2])
    return bond_dict


def _covalent_bonds(
    atom_array: AtomArray,
    is_polymer_atom: Bool[np.ndarray, " atoms"],
    chain_entities: dict[str, ChainEntity],
    has_modifications: bool,
) -> list[dict[str, Any]]:
    """List the bonds Protenix needs spelled out, one entry per bonded atom pair.

    Parameters
    ----------
    atom_array : AtomArray
        Atoms of the chains in ``chain_entities``, annotated with ``seq_idx``.
    is_polymer_atom : np.ndarray
        Whether each atom belongs to a polymer chain.
    chain_entities : dict[str, ChainEntity]
        Entity and copy of every chain.
    has_modifications : bool
        Whether any polymer entity has a modification.

    Returns
    -------
    list[dict[str, Any]]
        Unmerged Protenix ``covalent_bonds`` entries.
    """
    positions = _residue_positions(atom_array, is_polymer_atom)
    bonds = _select_covalent_bonds(atom_array, is_polymer_atom, positions, has_modifications)
    return [_bond_to_json(bond, atom_array, positions, chain_entities) for bond in bonds]


def structure_to_protenix_json(structure: dict) -> dict[str, Any]:
    """Convert Atomworks structure to Protenix input JSON.

    Polymer chains are described by their full deposited sequence; modifications and
    bonds come from the resolved atoms of the first model.

    Parameters
    ----------
    structure: dict
        Atomworks structure dictionary with 'asym_unit' key containing
        AtomArray or AtomArrayStack.

    Returns
    -------
    dict[str, Any]
        Protenix-compatible JSON dictionary.
    """
    if "asym_unit" not in structure:
        raise ValueError("structure must contain asym_unit key")
    chain_info = structure.get("chain_info", {})

    atom_array = ensure_atom_array_stack(structure["asym_unit"])[0]
    atom_array = annotate_sequence_positions(atom_array, chain_info)
    atom_array = atom_array[get_valid_atom_mask(atom_array)]

    sequences, chain_entities = _group_chains_into_entities(atom_array, chain_info)
    atom_array = atom_array[np.isin(np.asarray(atom_array.chain_id), list(chain_entities))]
    json_dict: dict[str, Any] = {"sequences": sequences}

    is_polymer_atom = np.array(
        [chain_info[chain_id]["chain_type"].is_polymer() for chain_id in atom_array.chain_id],
        dtype=bool,
    )
    has_modifications = any(
        "modifications" in entity_dict for entry in sequences for entity_dict in entry.values()
    )
    covalent_bonds = _covalent_bonds(atom_array, is_polymer_atom, chain_entities, has_modifications)
    if covalent_bonds:
        entity_counts = {
            str(entity_index): entity_dict["count"]
            for entity_index, entry in enumerate(sequences, start=1)
            for entity_dict in entry.values()
        }
        json_dict["covalent_bonds"] = merge_covalent_bonds(covalent_bonds, entity_counts)

    json_dict["name"] = structure.get("metadata", {}).get("name", "sample")

    return json_dict


def create_protenix_input_from_structure(structure: dict, out_dir: str | Path) -> tuple[Path, dict]:
    """Create Protenix input JSON from Atomworks structure.

    Parameters
    ----------
    structure: dict
        Atomworks structure dictionary.
    out_dir: str | Path
        Output directory for saving JSON file.

    Returns
    -------
    tuple[Path, dict]
        Path to saved JSON file and JSON dictionary.
    """
    out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
    out_dir = out_dir.expanduser().resolve()

    json_dict = structure_to_protenix_json(structure)

    protenix_input_path = out_dir / "protenix_input.json"
    protenix_input_path.parent.mkdir(parents=True, exist_ok=True)

    with open(protenix_input_path, "w") as f:
        json.dump([json_dict], f, indent=4)

    return protenix_input_path, json_dict
