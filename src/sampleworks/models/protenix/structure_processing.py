import copy
import functools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from atomworks.enums import ChainType
from biotite.structure import (
    array,
    Atom,
    AtomArray,
    AtomArrayStack,
    concatenate,
    get_chain_starts,
    get_residue_starts,
)
from loguru import logger
from protenix.data.constants import STD_RESIDUES
from protenix.data.utils import (
    get_lig_lig_bonds,
    get_ligand_polymer_bond_mask,
    get_polymer_polymer_bond,
)


def ensure_atom_array(atom_array: AtomArray | AtomArrayStack) -> AtomArray:
    """Convert AtomArrayStack to AtomArray by extracting first frame.

    Parameters
    ----------
    atom_array: AtomArray or AtomArrayStack
        Input structure from atomworks.parse().

    Returns
    -------
    AtomArray
        Single-frame AtomArray. If input is AtomArrayStack, returns first frame.
        If input is AtomArray, returns unchanged.

    Notes
    -----
    atomworks.parse() always returns AtomArrayStack for asym_unit.
    Since annotations (chain_id, entity_id, etc.) are identical across
    all frames, we extract the first frame for metadata processing.
    """
    if isinstance(atom_array, AtomArrayStack):
        return cast(AtomArray, atom_array[0])
    return atom_array


# TODO: Fix this so that we can handle multiple altlocs properly
#  http://github.com/k-chrispens/sampleworks/issues/46
def filter_altloc(atom_array: AtomArray) -> AtomArray:
    """Filter atom array to single conformation per residue.

    Keeps atoms with no alternative location ('') or first alternative ('A').
    This ensures each residue is counted once even when multiple conformations exist.
    """
    if not hasattr(atom_array, "altloc"):
        return atom_array
    altloc = cast(np.ndarray, atom_array.altloc)
    mask = (altloc == "") | (altloc == "A")
    return cast(AtomArray, atom_array[mask])


def filter_zero_occupancy(atom_array: AtomArray) -> AtomArray:
    """Filter out atoms with zero occupancy or NaN coordinates."""
    mask = np.ones(len(atom_array), dtype=bool)

    if hasattr(atom_array, "occupancy"):
        mask &= cast(np.ndarray, atom_array.occupancy) > 0

    coords = cast(np.ndarray, atom_array.coord)
    mask &= ~np.any(np.isnan(coords), axis=-1)

    return cast(AtomArray, atom_array[mask])


def get_valid_residue_positions(atom_array: AtomArray) -> dict[str, set[int]]:
    """Get residue positions that have at least one atom with valid coordinates.

    Parameters
    ----------
    atom_array: AtomArray
        Biotite AtomArray (should be BEFORE filter_zero_occupancy is applied).

    Returns
    -------
    dict[str, set[int]]
        Mapping from chain_id to set of res_id values that have valid coordinates.
    """
    valid_positions: dict[str, set[int]] = defaultdict(set)
    coords = cast(np.ndarray, atom_array.coord)
    chain_ids = cast(np.ndarray, atom_array.chain_id)
    res_ids = cast(np.ndarray, atom_array.res_id)

    valid_mask = ~np.any(np.isnan(coords), axis=-1)
    for i in np.where(valid_mask)[0]:
        valid_positions[str(chain_ids[i])].add(int(res_ids[i]))

    return dict(valid_positions)


def add_terminal_oxt_atoms(atom_array: AtomArray, chain_info: dict) -> AtomArray:
    """Add terminal OXT atoms to C-terminal residues of protein chains with
    ideal geometry.

    Parameters
    ----------
    atom_array: AtomArray
        Biotite AtomArray to process (not AtomArrayStack).
    chain_info: dict[str, Any]
        Atomworks chain information dictionary.

    Returns
    -------
    AtomArray
        AtomArray with OXT atoms added to C-termini of protein chains.
    """
    new_oxt_atoms = []
    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=True)

    for i in range(len(chain_starts) - 1):
        chain_start = chain_starts[i]
        chain_end = chain_starts[i + 1]
        chain_atoms = cast(AtomArray, atom_array[chain_start:chain_end])
        chain_id = chain_atoms[0].chain_id

        if chain_id not in chain_info:
            continue

        chain_type = chain_info[chain_id]["chain_type"]
        if chain_type not in (ChainType.POLYPEPTIDE_L, ChainType.POLYPEPTIDE_D):
            continue

        residue_starts = get_residue_starts(chain_atoms, add_exclusive_stop=True)
        if len(residue_starts) < 2:
            continue

        last_res_start = residue_starts[-2]
        last_res_end = residue_starts[-1]
        terminal_residue = cast(AtomArray, chain_atoms[last_res_start:last_res_end])

        if "OXT" in cast(np.ndarray, terminal_residue.atom_name):
            # check that OXT has coordinates
            oxt_mask = terminal_residue.atom_name == "OXT"
            oxt_atom = cast(AtomArray, terminal_residue[oxt_mask])
            if not np.isnan(cast(np.ndarray, oxt_atom.coord)).any():
                continue

        c_mask = terminal_residue.atom_name == "C"
        o_mask = terminal_residue.atom_name == "O"
        ca_mask = terminal_residue.atom_name == "CA"

        if not (c_mask.any() and o_mask.any() and ca_mask.any()):
            continue

        c_atom = cast(AtomArray, terminal_residue[c_mask])[0]
        o_atom = cast(AtomArray, terminal_residue[o_mask])[0]
        ca_atom = cast(AtomArray, terminal_residue[ca_mask])[0]

        c_coord = cast(np.ndarray, c_atom.coord)
        o_coord = cast(np.ndarray, o_atom.coord)
        ca_coord = cast(np.ndarray, ca_atom.coord)

        c_to_o = o_coord - c_coord
        c_to_ca = ca_coord - c_coord

        c_o_distance = np.linalg.norm(c_to_o)

        plane_normal = np.cross(c_to_ca, c_to_o)
        plane_normal_length = np.linalg.norm(plane_normal)

        if plane_normal_length < 1e-6:
            plane_normal = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(c_to_o / c_o_distance, plane_normal)) > 0.9:
                plane_normal = np.array([1.0, 0.0, 0.0])
        else:
            plane_normal = plane_normal / plane_normal_length

        angle = np.deg2rad(126.0)

        c_to_o_normalized = c_to_o / c_o_distance

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rotated = (
            c_to_o_normalized * cos_angle
            + np.cross(plane_normal, c_to_o_normalized) * sin_angle
            + plane_normal * np.dot(plane_normal, c_to_o_normalized) * (1 - cos_angle)
        )

        oxt_coord = c_coord + rotated * c_o_distance

        oxt_annotations = {}
        for annotation in atom_array.get_annotation_categories():
            oxt_annotations[annotation] = getattr(o_atom, annotation)

        oxt_annotations["atom_name"] = "OXT"
        oxt_annotations["element"] = "O"
        oxt_annotations["coord"] = oxt_coord

        oxt_atom = Atom(**oxt_annotations)

        new_oxt_atoms.append(oxt_atom)

    if not new_oxt_atoms:
        return atom_array

    oxt_atom_array = array([oxt_atom for oxt_atom in new_oxt_atoms])

    combined_array = concatenate([atom_array, oxt_atom_array])

    return cast(AtomArray, combined_array)


def add_unique_chain_and_copy_ids(atom_array):
    """Add unique chain_id and copy_id annotations to AtomArray.

    Parameters
    ----------
    atom_array: AtomArray
        Biotite AtomArray to annotate (not AtomArrayStack).

    Returns
    -------
    AtomArray
        Annotated AtomArray with chain_id and copy_id fields.

    Notes
    -----
    Expects AtomArray, not AtomArrayStack. Use ensure_atom_array() first
    if working with atomworks.parse() output.
    """
    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=False)
    chain_starts_atom_array = atom_array[chain_starts]

    unique_label_entity_id = np.unique(atom_array.label_entity_id)
    chain_id_to_copy_id_dict = {}

    for label_entity_id in unique_label_entity_id:
        chain_ids_in_entity = chain_starts_atom_array.chain_id[
            chain_starts_atom_array.label_entity_id == label_entity_id
        ]
        for chain_count, chain_id in enumerate(chain_ids_in_entity):
            chain_id_to_copy_id_dict[chain_id] = chain_count + 1

    copy_id = np.vectorize(chain_id_to_copy_id_dict.get)(atom_array.chain_id)
    atom_array.set_annotation("copy_id", copy_id)

    return atom_array


def get_sequences(atom_array, chain_info, valid_positions=None):
    """Extract entity sequences from AtomArray.

    Parameters
    ----------
    atom_array: AtomArray
        Biotite AtomArray containing structure (not AtomArrayStack).
    chain_info: dict[str, Any]
        Atomworks chain information dictionary.
    valid_positions: dict[str, set[int]] | None
        Optional mapping from chain_id to set of res_id values with valid coords.
        If provided, sequences are filtered to only include residues at valid positions.

    Returns
    -------
    dict[str, str]
        Mapping from label_entity_id to sequence string.

    Notes
    -----
    Expects AtomArray, not AtomArrayStack. Use ensure_atom_array() first
    if working with atomworks.parse() output.
    """
    entity_seq = {}
    for label_entity_id in np.unique(atom_array.label_entity_id):
        for chain_id, info in chain_info.items():
            chain_atom = atom_array[atom_array.chain_id == chain_id]
            if len(chain_atom) > 0:
                if chain_atom[0].label_entity_id == label_entity_id:
                    chain_type = info["chain_type"]
                    if chain_type.is_polymer():
                        canonical_seq = info.get("processed_entity_canonical_sequence", "")
                        if valid_positions is not None and chain_id in valid_positions:
                            chain_valid = valid_positions[chain_id]
                            n_valid = len(chain_valid)
                            n_seq = len(canonical_seq)
                            if n_valid == n_seq:
                                entity_seq[label_entity_id] = canonical_seq
                            elif n_valid == 0:
                                entity_seq[label_entity_id] = ""
                            else:
                                min_pos = min(chain_valid)
                                max_pos = max(chain_valid)
                                n_missing_start = min_pos - 1
                                n_missing_end = n_seq - max_pos
                                start_idx = n_missing_start
                                end_idx = n_seq - n_missing_end
                                filtered = canonical_seq[start_idx:end_idx]
                                entity_seq[label_entity_id] = filtered
                        else:
                            entity_seq[label_entity_id] = canonical_seq
                    break
    return entity_seq


def get_poly_res_names(atom_array, chain_info, valid_positions=None):
    """Get residue names and sequence positions for polymer entities.

    Parameters
    ----------
    atom_array: AtomArray
        Biotite AtomArray containing structure (not AtomArrayStack).
    chain_info: dict[str, Any]
        Atomworks chain information dictionary.
    valid_positions: dict[str, set[int]] | None
        Optional mapping from chain_id to set of res_id values with valid coords.
        If provided, only residues at valid positions are included.

    Returns
    -------
    dict[str, list[tuple[int, str]]]
        Mapping from label_entity_id to list of (position, res_name) tuples.
        Position is 1-indexed based on label_seq_id.

    Notes
    -----
    Expects AtomArray, not AtomArrayStack. Use ensure_atom_array() first
    if working with atomworks.parse() output.

    Filters to single conformation (altloc '' or 'A') and single chain instance
    per entity to ensure correct residue counting.
    """
    poly_res_names = {}
    for label_entity_id in np.unique(atom_array.label_entity_id):
        for chain_id, info in chain_info.items():
            chain_atom = atom_array[atom_array.chain_id == chain_id]
            if len(chain_atom) > 0:
                if chain_atom[0].label_entity_id == label_entity_id:
                    chain_type = info["chain_type"]
                    if chain_type.is_polymer():
                        chain_array = filter_altloc(chain_atom)
                        if len(chain_array) == 0:
                            break
                        starts = get_residue_starts(chain_array, add_exclusive_stop=True)
                        res_names = cast(np.ndarray, chain_array.res_name)[starts[:-1]].tolist()
                        if hasattr(chain_array, "res_id"):
                            res_ids = cast(np.ndarray, chain_array.res_id)[starts[:-1]].tolist()
                            min_res_id = min(res_ids) if res_ids else 1
                            positions = [r - min_res_id + 1 for r in res_ids]
                        else:
                            positions = list(range(1, len(res_names) + 1))

                        pos_res_pairs = list(zip(positions, res_names, strict=False))
                        if valid_positions is not None and chain_id in valid_positions:
                            chain_valid = valid_positions[chain_id]
                            min_valid = min(chain_valid) if chain_valid else 1
                            valid_seq_positions = {r - min_valid + 1 for r in chain_valid}
                            pos_res_pairs = [
                                (p, r) for p, r in pos_res_pairs if p in valid_seq_positions
                            ]
                        poly_res_names[label_entity_id] = pos_res_pairs
                    break
    return poly_res_names


def detect_modifications(atom_array, chain_info, valid_positions=None):
    """Detect polymer modifications (non-standard residues).

    Parameters
    ----------
    atom_array: AtomArray
        Biotite AtomArray containing structure (not AtomArrayStack).
    chain_info: dict[str, Any]
        Atomworks chain information dictionary.
    valid_positions: dict[str, set[int]] | None
        Optional mapping from chain_id to set of res_id values with valid coords.

    Returns
    -------
    dict[str, list[tuple[int, str]]]
        Mapping from label_entity_id to list of (position, mod_ccd_code).
        Position is 1-indexed based on label_seq_id from the structure.

    Notes
    -----
    Expects AtomArray, not AtomArrayStack. Use ensure_atom_array() first
    if working with atomworks.parse() output.
    """
    entity_id_to_mod_list = {}
    poly_res_data = get_poly_res_names(atom_array, chain_info, valid_positions)

    for entity_id, pos_res_pairs in poly_res_data.items():
        modifications_list = []
        for position, res_name in pos_res_pairs:
            if res_name not in STD_RESIDUES:
                modifications_list.append((position, f"CCD_{res_name}"))
        if modifications_list:
            entity_id_to_mod_list[entity_id] = modifications_list

    return entity_id_to_mod_list


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
        bond_unique_string = []
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


def structure_to_protenix_json(structure: dict) -> dict[str, Any]:
    """Convert Atomworks structure to Protenix input JSON.

    Parameters
    ----------
    structure: dict
        Atomworks structure dictionary with 'asym_unit' key containing
        AtomArray or AtomArrayStack.

    Returns
    -------
    dict[str, Any]
        Protenix-compatible JSON dictionary.

    Notes
    -----
    Automatically handles both AtomArray and AtomArrayStack inputs.
    For AtomArrayStack, extracts first frame since annotations are
    identical across all frames.
    """
    if "asym_unit" not in structure:
        raise ValueError("structure must contain asym_unit key")

    atom_array = structure["asym_unit"]
    atom_array = ensure_atom_array(atom_array)
    chain_info = structure.get("chain_info", {})

    valid_positions = get_valid_residue_positions(atom_array)
    atom_array = filter_zero_occupancy(atom_array)

    atom_array = add_terminal_oxt_atoms(atom_array, chain_info)

    # label_entity_id groups chains into biological entities; ligands get a "_lig" suffix
    # to keep them distinct from polymer entities sharing the same base ID.
    if not hasattr(atom_array, "label_entity_id"):
        chain_to_entity = {}
        for chain_id, info in chain_info.items():
            base_entity_id = info.get("rcsb_entity", chain_id)
            chain_type = info["chain_type"]
            if not chain_type.is_polymer():
                entity_id = f"{base_entity_id}_lig"
            else:
                entity_id = str(base_entity_id)
            chain_to_entity[chain_id] = entity_id

        label_entity_ids = np.array(
            [chain_to_entity.get(cid, cid) for cid in cast(np.ndarray, atom_array.chain_id)]
        )
        atom_array.set_annotation("label_entity_id", label_entity_ids)

    if not hasattr(atom_array, "label_asym_id"):
        atom_array.set_annotation("label_asym_id", atom_array.chain_id)

    # entity_seq: {label_entity_id -> amino-acid / nucleotide sequence string}
    # copy_id annotation: distinguishes homo-multimer copies of the same entity
    entity_seq = get_sequences(atom_array, chain_info, valid_positions)

    atom_array = add_unique_chain_and_copy_ids(atom_array)

    # label_entity_id_to_sequences: ligand CCD residue-name lists (e.g. ["ATP"])
    # lig_chain_ids: chain IDs belonging to non-polymer (ligand/ion) entities
    label_entity_id_to_sequences = {}
    lig_chain_ids = []

    for label_entity_id in np.unique(cast(np.ndarray, atom_array.label_entity_id)):
        entity_chain_type = None
        for chain_id, info in chain_info.items():
            chain_atom = atom_array[atom_array.chain_id == chain_id]
            assert isinstance(chain_atom, AtomArray | AtomArrayStack)
            if len(chain_atom) > 0:
                if chain_atom[0].label_entity_id == label_entity_id:
                    entity_chain_type = info["chain_type"]
                    break

        if entity_chain_type and not entity_chain_type.is_polymer():
            current_lig_chain_ids = np.unique(
                cast(np.ndarray, atom_array.chain_id)[atom_array.label_entity_id == label_entity_id]
            ).tolist()
            lig_chain_ids += current_lig_chain_ids

            for chain_id in current_lig_chain_ids:
                lig_atom_array = atom_array[atom_array.chain_id == chain_id]
                starts = get_residue_starts(lig_atom_array, add_exclusive_stop=True)
                seq = cast(np.ndarray, lig_atom_array.res_name)[starts[:-1]].tolist()
                label_entity_id_to_sequences[label_entity_id] = seq
                break

    entity_id_to_mod_list = detect_modifications(atom_array, chain_info, valid_positions)

    # Build the "sequences" list in Protenix JSON format
    # Each entry is {entity_type: entity_dict} where entity_type is one of
    # "proteinChain", "dnaSequence", "rnaSequence", or "ligand".
    # Also builds two side-maps used later for covalent bond serialization:
    #   label_entity_id_to_entity_id_in_json: internal entity ID -> 1-based JSON entity index
    #   all_entity_counts: JSON entity index -> number of symmetric copies
    json_dict = {"sequences": []}

    unique_label_entity_id = np.unique(cast(np.ndarray, atom_array.label_entity_id))
    all_entity_counts = {}
    label_entity_id_to_entity_id_in_json = {}
    entity_idx = 0

    for label_entity_id in unique_label_entity_id:
        entity_dict = {}
        entity_mask = atom_array.label_entity_id == label_entity_id
        unique_chain_ids_for_entity = np.unique(cast(np.ndarray, atom_array.chain_id)[entity_mask])
        n_chains_for_entity = len(unique_chain_ids_for_entity)

        entity_chain_type: ChainType | None = None
        for chain_id, info in chain_info.items():
            chain_atom = atom_array[atom_array.chain_id == chain_id]
            if isinstance(chain_atom, AtomArray | AtomArrayStack) and len(chain_atom) > 0:
                if chain_atom[0].label_entity_id == label_entity_id:
                    entity_chain_type = info["chain_type"]
                    break

        if not entity_chain_type:
            continue

        if entity_chain_type.is_polymer():
            if entity_chain_type in (
                ChainType.POLYPEPTIDE_L,
                ChainType.POLYPEPTIDE_D,
            ):
                entity_type = "proteinChain"
            elif entity_chain_type == ChainType.DNA:
                entity_type = "dnaSequence"
            elif entity_chain_type == ChainType.RNA:
                entity_type = "rnaSequence"
            else:
                continue

            sequence = entity_seq.get(label_entity_id, "")
            entity_dict["sequence"] = sequence
            if entity_type == "proteinChain":
                entity_dict["msa"] = {}
        else:
            entity_type = "ligand"
            lig_ccd_components = label_entity_id_to_sequences.get(label_entity_id, ["UNK"])
            lig_ccd_components = [
                _expand_tilde_ccd_code(c) if c.startswith("~") else c for c in lig_ccd_components
            ]
            lig_ccd = "_".join(lig_ccd_components)
            entity_dict["ligand"] = f"CCD_{lig_ccd}"

        entity_dict["count"] = n_chains_for_entity
        entity_idx += 1
        entity_id_in_json = str(entity_idx)
        label_entity_id_to_entity_id_in_json[label_entity_id] = entity_id_in_json
        all_entity_counts[entity_id_in_json] = n_chains_for_entity

        if label_entity_id in entity_id_to_mod_list:
            modifications = entity_id_to_mod_list[label_entity_id]
            if entity_type == "proteinChain":
                entity_dict["modifications"] = [
                    {"ptmPosition": position, "ptmType": mod_ccd_code}
                    for position, mod_ccd_code in modifications
                ]
            elif entity_type in ("dnaSequence", "rnaSequence"):
                entity_dict["modifications"] = [
                    {
                        "basePosition": position,
                        "modificationType": mod_ccd_code,
                    }
                    for position, mod_ccd_code in modifications
                ]

        json_dict["sequences"].append({entity_type: entity_dict})

    # Filter atom array to only atoms belonging to recognized entities
    atom_array = cast(
        AtomArray,
        atom_array[
            np.isin(
                cast(np.ndarray, atom_array.label_entity_id),
                list(label_entity_id_to_entity_id_in_json.keys()),
            )
        ],
    )

    # Build entity_poly_type: {label_entity_id -> mmCIF polymer type string}
    # Used below to distinguish polymer vs ligand atoms for mol_type annotations
    # and to determine how bond positions are computed (1-based offset vs fixed 1).
    entity_poly_type = {}
    for chain_id, info in chain_info.items():
        chain_atom = cast(AtomArray, atom_array[atom_array.chain_id == chain_id])
        if len(chain_atom) > 0:
            label_entity_id = chain_atom[0].label_entity_id
            chain_type = info["chain_type"]
            if chain_type.is_polymer():
                if chain_type in (
                    ChainType.POLYPEPTIDE_L,
                    ChainType.POLYPEPTIDE_D,
                ):
                    entity_poly_type[label_entity_id] = "polypeptide(L)"
                elif chain_type == ChainType.DNA:
                    entity_poly_type[label_entity_id] = "polydeoxyribonucleotide"
                elif chain_type == ChainType.RNA:
                    entity_poly_type[label_entity_id] = "polyribonucleotide"

    # token_mol_type: uppercase "PROTEIN"/"LIGAND" (used by Protenix tokenizer)
    # mol_type: lowercase "protein"/"ligand" (used by Protenix data pipeline)
    if not hasattr(atom_array, "token_mol_type"):
        token_mol_types = []
        for i in range(len(atom_array)):
            label_ent_id = cast(np.ndarray, atom_array.label_entity_id)[i]
            if label_ent_id in entity_poly_type:
                token_mol_types.append("PROTEIN")
            else:
                token_mol_types.append("LIGAND")
        atom_array.set_annotation("token_mol_type", np.array(token_mol_types))

    if not hasattr(atom_array, "mol_type"):
        mol_types = []
        for i in range(len(atom_array)):
            label_ent_id = cast(np.ndarray, atom_array.label_entity_id)[i]
            if label_ent_id in entity_poly_type:
                mol_types.append("protein")
            else:
                mol_types.append("ligand")
        atom_array.set_annotation("mol_type", np.array(mol_types))

    # Protenix needs explicit covalent bonds for ligand attachments and modified
    # residues. Bonds are described in a JSON format that references atoms
    # by (entity index, 1-based residue position, atom name, copy number) rather
    # than raw atom indices.
    has_modifications = len(entity_id_to_mod_list) > 0

    lig_polymer_bonds = get_ligand_polymer_bond_mask(atom_array, lig_include_ions=False)
    lig_lig_bonds = get_lig_lig_bonds(atom_array, lig_include_ions=False)

    has_ligand_bonds = lig_polymer_bonds.size > 0 or lig_lig_bonds.size > 0

    if has_modifications or has_ligand_bonds:
        # Per-entity minimum res_id, used as offset to convert raw res_id values
        # into 1-based positions within each entity's sequence.
        entity_min_res_id: dict[str, int] = {}
        for label_entity_id in np.unique(cast(np.ndarray, atom_array.label_entity_id)):
            for chain_id in chain_info:
                chain_atom = atom_array[atom_array.chain_id == chain_id]
                if len(chain_atom) > 0 and chain_atom[0].label_entity_id == label_entity_id:
                    if chain_id in valid_positions and valid_positions[chain_id]:
                        entity_min_res_id[str(label_entity_id)] = min(valid_positions[chain_id])
                    else:
                        chain_res_ids = cast(np.ndarray, chain_atom.res_id)
                        entity_min_res_id[str(label_entity_id)] = int(np.min(chain_res_ids))
                    break

        # Collect all bond atom-index pairs from two sources:
        # 1) ligand bonds (lig-polymer + lig-lig), filtered to only those involving
        #    a ligand chain atom
        # 2) polymer-polymer bonds from modified residues (e.g. disulfides)
        token_bonds_list = []

        if has_ligand_bonds:
            ligand_bonds = np.vstack((lig_polymer_bonds, lig_lig_bonds))
            # Keep only bonds where at least one partner belongs to a ligand chain
            lig_indices = np.where(np.isin(cast(np.ndarray, atom_array.chain_id), lig_chain_ids))[0]
            lig_bond_mask = np.any(np.isin(ligand_bonds[:, :2], lig_indices), axis=1)
            ligand_bonds = ligand_bonds[lig_bond_mask]
            if ligand_bonds.size > 0:
                token_bonds_list.append(ligand_bonds)

        if has_modifications:
            polymer_polymer_bond = get_polymer_polymer_bond(atom_array, entity_poly_type)
            if polymer_polymer_bond.size > 0:
                token_bonds_list.append(polymer_polymer_bond)

        # Convert atom-index bond pairs into Protenix's JSON bond format.
        # Each bond partner is identified by:
        #   entity  – 1-based JSON entity index
        #   position – 1-based residue position within the entity (polymer) or 1 (ligand)
        #   atom    – PDB atom name (e.g. "SG", "C1")
        #   copy    – which symmetric copy of this entity the atom belongs to
        if token_bonds_list:
            token_bonds = np.vstack(token_bonds_list)
            covalent_bonds = []
            for atoms in token_bonds[:, :2]:
                bond_dict = {}
                for i in range(2):
                    raw_res_id = int(cast(np.ndarray, atom_array.res_id)[atoms[i]])
                    label_entity_id = atom_array.get_annotation("label_entity_id")[atoms[i]]
                    # For polymers, convert raw res_id to 1-based position relative
                    # to the entity's first residue. Non-polymer entities are single-residue
                    # entities so position is always 1.
                    if label_entity_id in entity_poly_type:
                        min_res_id = entity_min_res_id.get(str(label_entity_id), 1)
                        position = raw_res_id - min_res_id + 1
                    else:
                        position = 1
                    bond_dict[f"entity{i + 1}"] = int(
                        label_entity_id_to_entity_id_in_json[label_entity_id]
                    )
                    bond_dict[f"position{i + 1}"] = position
                    bond_dict[f"atom{i + 1}"] = cast(np.ndarray, atom_array.atom_name)[atoms[i]]
                    bond_dict[f"copy{i + 1}"] = int(atom_array.get_annotation("copy_id")[atoms[i]])

                covalent_bonds.append(bond_dict)

            # Deduplicate and normalize bond dicts across symmetric copies
            merged_covalent_bonds = merge_covalent_bonds(covalent_bonds, all_entity_counts)
            json_dict["covalent_bonds"] = merged_covalent_bonds

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
