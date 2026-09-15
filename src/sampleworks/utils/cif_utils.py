import itertools
import tempfile
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArrayStack
from biotite.structure.io.pdbx.cif import CIFBlock, CIFCategory, CIFFile
from loguru import logger

from sampleworks.utils.atom_array_utils import (
    find_all_altloc_ids,
    save_structure_to_cif,
    select_altloc,
)


def find_altloc_selections(
    cif_file: Path | str,
    altloc_label: str = "label_alt_id",
    min_span: int = 5,
    include_all_altlocs: bool = True,
) -> Iterable[str]:
    """Find alternative location selections in a CIF file.

    Individual spans at least ``min_span`` residues long are yielded as selection strings.
    Optionally, a final batch of selection strings is also yielded that contains all residues
    with altlocs, one selection per chain.

    Parameters
    ----------
    cif_file : Path | str
        Path to the CIF file.
    altloc_label : str
        Label for alternative location identifier. Default is ``'label_alt_id'``.
        If you don't know it, search for ``"_atom_site"`` in your CIF file to identify it.
    min_span : int
        Minimum number of consecutive residues to consider an altloc selection.
        Spans of altlocs shorter than this are not yielded as selection strings, but ARE
        included in the final selections which includes all residues with altlocs in each chain when
        ``include_all_altlocs=True``.
    include_all_altlocs : bool
        If True (default), yield a final per-chain selection string containing all residues
        with altlocs regardless of span length.

    Yields
    ------
    str
        Alternative location selections, keyed by altloc ID.

    Examples
    --------
    For RCSB PDB entry 5SOP, this should yield items like:
    ``['chain A and resi 125-137', "chain_id == 'A' and ((res_id >= 3 and res_id <= 6) or ...)"]``
    """
    cif_file = Path(cif_file)
    logger.info(f"Finding altloc selections for {cif_file}")
    structure = load_any(cif_file, altloc="all", extra_fields=["occupancy", altloc_label])

    # our other methods rely on the annotation "altloc_id" being present, so we'll add it here.
    structure.set_annotation("altloc_id", structure.get_annotation(altloc_label))

    altlocs = OrderedDict()
    for altloc_id in find_all_altloc_ids(structure):
        altk = select_altloc(structure, altloc_id=altloc_id)
        unique_altk = set((ch, res) for ch, res in zip(altk.chain_id, altk.res_id, strict=True))
        # probably unnecessary but making sure these are consistently ordered
        # FIXME? This is a little clunky. Perhaps should be hierarchical by chain then altloc?
        #   At some point though we'll do altloc selections using correlations/contacts
        #   so this is probably not a big deal.
        altlocs[altloc_id] = sorted(list(unique_altk))

    all_altloc_selections = {}
    for chain, start, end, _ in find_consecutive_residues(altlocs):
        if end - start >= min_span - 1:
            # FIXME use new style selection https://github.com/prism-science/sampleworks/issues/56
            yield f"chain {chain} and resi {start}-{end}"  # old style, more compact, selection

        if include_all_altlocs:
            if chain not in all_altloc_selections:
                all_altloc_selections[chain] = []
            if start == end:
                all_altloc_selections[chain].append(f"(res_id == {start})")
            else:
                all_altloc_selections[chain].append(f"(res_id >= {start} and res_id <= {end})")

    for chain, selections in all_altloc_selections.items():
        yield f"chain_id == '{chain}' and ({' or '.join(selections)})"


def find_consecutive_residues(
    altlocs: dict[str, list[tuple[str, int]]],  # Ex: {'A': [('X', 1), ('X', 2), ('X', 3)]}
) -> Iterable[tuple[str, int, int, set[str]]]:
    """Find and yield spans of consecutive residues with the same set of altloc identifiers.

    This function processes a dictionary mapping alternate location identifiers (altlocs)
    to (chain_id, residue_id) tuples having that altloc. For each chain_id in the structure,
    it yields spans of consecutive residues when membership in altlocs changes
    or where a break in residue numbering occurs. The yielded spans include information about
    the chain, start residue, end residue, and the corresponding membership.

    Parameters
    ----------
    altlocs : dict[str, list[tuple[str, int]]]
        A dictionary where keys are alternate location identifiers and values are
        lists of tuples representing chain identifiers (str) and residue IDs (int).

    Yields
    ------
    tuple[str, int, int, set[str]]
        A tuple containing the chain, start residue ID, end residue ID, and a set
        of alternate location identifiers representing the membership in the span.

    Examples
    --------
    For RCSB PDB entry 5SOP, this should yield::

        [('A', 3, 6, {'A', 'B'}),
         ('A', 10, 12, {'A', 'B'}),
         ('A', 20, 26, {'A', 'B'}),
         ('A', 28, 31, {'A', 'B'}),
         ('A', 38, 38, {'A', 'B'}),
         ('A', 42, 42, {'A', 'B'}),
         ('A', 44, 59, {'A', 'B'}),
         ('A', 87, 88, {'A', 'B'}),
         ('A', 97, 108, {'A', 'B'}),
         ('A', 113, 113, {'A', 'B'}),
         ('A', 125, 137, {'A', 'B', 'C'}),
         ('A', 138, 141, {'A', 'B'}),
         ('A', 155, 169, {'A', 'B'})]
    """
    # TODO create test cases from 5SOP and 7Z0E, low priority since this isn't a critical function
    #   and will likely change in the future anyway.
    #   https://github.com/prism-science/sampleworks/issues/111

    # First find the chains
    all_chains = {res[0] for altloc in altlocs.values() for res in altloc}

    # iterating over chains, check each residue's membership in altlocs.
    # Yield spans when membership changes or there is a break in the residue number
    for chain in all_chains:
        chain_altlocs = {
            altloc_id: {res[1] for res in altlocs[altloc_id] if res[0] == chain}
            for altloc_id in altlocs
        }
        all_res_ids = sorted(list(set.union(*chain_altlocs.values())))
        if not all_res_ids:
            continue

        start = all_res_ids[0]
        next_res_id = None
        current_membership = {k for k in chain_altlocs if start in chain_altlocs[k]}
        start = start if len(current_membership) > 1 else None
        for current_res_id, next_res_id in itertools.pairwise(all_res_ids):
            res_membership = {k for k in chain_altlocs if next_res_id in chain_altlocs[k]}
            if res_membership != current_membership or next_res_id - current_res_id > 1:
                if start is not None:
                    yield chain, start, current_res_id, current_membership

                start = next_res_id if len(res_membership) > 1 else None
                current_membership = res_membership if len(res_membership) > 1 else None
        if start is not None and next_res_id:
            yield chain, start, next_res_id, current_membership


def resolve_mixed_hetatm_atom_altlocs(cif_path: Path | str) -> Path:
    """Pre-process a CIF file where ATOM and HETATM records with different residue names
    share the same (chain, residue) position via different altloc IDs.

    This occurs when a residue has a modified form (e.g. CSO, cysteic acid) as some
    altlocs and the canonical form (e.g. CYS) as another altloc at the same sequence
    position. Atomworks treats these as two sequential residues rather than alternates,
    inserting a spurious extra residue into the sequence fed to Boltz2.

    Should Atomworks fix the underlying issue in the future, we should remove this method.

    The fix: for each affected position, remove the HETATM (modified) records and keep
    only the ATOM (canonical) records. Also cleans up the ``_struct_conn`` covale bonds
    referencing the removed residues, since ``save_structure_to_cif`` only writes
    ``_atom_site``.

    A warning is logged for every affected (chain, residue) position.

    Parameters
    ----------
    cif_path
        Path to the input CIF file.

    Returns
    -------
    Path
        Path to a fixed temporary CIF file if any positions were modified, or the
        original ``cif_path`` unchanged if no issues were found.
    """
    cif_path = Path(cif_path)
    atom_array = load_any(cif_path, altloc="all", extra_fields=["occupancy", "b_factor"])
    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]

    chain_id = atom_array.chain_id
    res_id = atom_array.res_id
    res_name = atom_array.res_name
    hetero = atom_array.hetero

    keep_mask = np.ones(len(atom_array), dtype=bool)
    found_any = False

    for chain in np.unique(chain_id):
        for rid in np.unique(res_id[chain_id == chain]):
            pos_mask = (chain_id == chain) & (res_id == rid)
            has_no_hetatm = np.any(~hetero[pos_mask])
            has_hetatm = np.any(hetero[pos_mask])

            if not (has_no_hetatm and has_hetatm):
                # there are either only HETATM or only ATOM records at this position, or none at all
                continue

            atom_res_names = np.unique(res_name[pos_mask & ~hetero])
            hetatm_res_names = np.unique(res_name[pos_mask & hetero])

            if set(atom_res_names) == set(hetatm_res_names):
                continue  # Same residue name on both — not the case we're fixing

            logger.warning(
                f"Chain {chain}, residue {rid}: found mixed ATOM {list(atom_res_names)} "
                f"and HETATM {list(hetatm_res_names)} records with different residue names "
                f"at the same sequence position. Removing HETATM records to prevent "
                f"atomworks from inserting a duplicate residue into the Boltz2 input sequence."
            )
            keep_mask[pos_mask & hetero] = False
            found_any = True

    if not found_any:
        return cif_path

    fixed_array = atom_array[keep_mask]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cif", prefix="sampleworks_fixed_cif_", delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        save_structure_to_cif(fixed_array, tmp_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError as error:
            logger.warning(
                f"Failed to remove temporary CIF after save failure: {tmp_path}: {error}"
            )
        raise

    logger.info(f"Wrote altloc-fixed CIF to temporary file: {tmp_path}")
    return tmp_path


def renumber_atom_site_ids(cif_file: CIFFile) -> None:
    """Renumber ``_atom_site.id`` values across all models in place.

    ``set_structure()`` copies and tiles an ``atom_id`` annotation for multi-model
    structures, which can produce duplicate category keys. Call this function after
    ``set_structure()`` on a single-block CIF file.

    Parameters
    ----------
    cif_file : CIFFile
        Single-block CIF file whose ``atom_site`` category is modified in place.
    """
    category = cif_file.block["atom_site"]
    category["id"] = np.arange(1, category.row_count + 1)


_POLYMER_ENTITY_CATEGORIES = ("entity", "entity_poly", "entity_poly_seq")


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


def _matching_entity(
    reference_block: CIFBlock,
    residue_numbers: list[int],
    residue_names: list[str],
) -> tuple[str, int]:
    """Find one reference polymer entity containing an output residue sequence.

    Parameters
    ----------
    reference_block : CIFBlock
        Deposited CIF block containing polymer entity categories.
    residue_numbers : list[int]
        Sorted output ``label_seq_id`` values.
    residue_names : list[str]
        Output residue names corresponding to ``residue_numbers``.

    Returns
    -------
    tuple[str, int]
        Reference entity ID and numbering offset to apply to its sequence.

    Raises
    ------
    ValueError
        If the output is empty, gapped, unmatched, or ambiguously matched.
    """
    if not residue_numbers:
        raise ValueError("Output entity has no polymer residues")
    if np.any(np.diff(residue_numbers) != 1):
        raise ValueError("Output entity residue numbering is not contiguous")

    entity_poly = reference_block["entity_poly"]
    sequence = reference_block["entity_poly_seq"]
    sequence_ids = np.asarray(sequence["entity_id"].as_array(str))
    sequence_numbers = np.asarray(sequence["num"].as_array(int))
    sequence_names = np.asarray(sequence["mon_id"].as_array(str))
    matches: list[tuple[str, int]] = []
    for entity_id in entity_poly["entity_id"].as_array(str):
        mask = sequence_ids == entity_id
        reference_numbers = sequence_numbers[mask]
        reference_names = sequence_names[mask]
        for start in range(len(reference_names) - len(residue_names) + 1):
            if list(reference_names[start : start + len(residue_names)]) == residue_names:
                matches.append((str(entity_id), residue_numbers[0] - int(reference_numbers[start])))

    if len(matches) != 1:
        raise ValueError(f"Expected one reference entity sequence match, found {len(matches)}")
    return matches[0]


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


def add_category_to_cif(
    ciffile: CIFFile,
    data: dict[str, Any],
    category_name: str,
    overwrite: bool = False,
    block_name: str | None = None,
) -> None:
    """Add a custom category in-place to a CIFFile.

    Parameters
    ----------
    ciffile : CIFFile
        The CIF file object to modify.
    data : dict[str, Any]
        Dictionary with column names as keys and column data as values.
    category_name : str
        Name of the category to add (e.g., "custom_data").
    overwrite : bool, optional
        If False and the category already exists, raise RuntimeError. Default is False.
    block_name : str | None, optional
        Name of the block to add the category to. If None, check that there is only
        one block and add to that block. Default is None.

    Raises
    ------
    RuntimeError
        If category already exists and overwrite is False.
    ValueError
        If block_name is None but the file has multiple blocks, or if the specified
        block_name does not exist.

    Examples
    --------
    >>> from biotite.structure.io.pdbx.cif import CIFFile
    >>> ciffile = CIFFile.read("example.cif")  # assuming it contains a single block
    >>> data = {"id": [1, 2, 3], "value": ["a", "b", "c"]}
    >>> add_category_to_cif(ciffile, data, "my_custom_data")
    >>> print(ciffile.block["my_custom_data"].serialize())
    loop_
    _my_custom_data.id
    _my_custom_data.value
    1 a
    2 b
    3 c
    >>> data = {"sampleworks_version": "0.4.0", "pdb_id": "1L63"}
    >>> add_category_to_cif(ciffile, data, "sampleworks_metadata")
    >>> print(ciffile.block["sampleworks_metadata"].serialize())
    _sampleworks_metadata.sampleworks_version 0.4.0
    _sampleworks_metadata.pdb_id              1L63
    """
    # Determine which block to use
    if block_name is None:
        # CIFFile is a Mapping, so inherits .keys(), which ultimately iterates over blocks
        blocks = list(ciffile.keys())
        if len(blocks) == 0:
            raise ValueError("CIFFile has no blocks. Cannot add category.")
        elif len(blocks) > 1:
            raise ValueError(
                f"CIFFile has multiple blocks: {blocks}. Please specify block_name parameter."
            )
        block = ciffile[blocks[0]]
    else:
        if block_name not in ciffile:
            raise ValueError(f"Block '{block_name}' not found in CIFFile.")
        block = ciffile[block_name]

    # Check if a category with name category_name already exists
    if category_name in block and not overwrite:
        raise RuntimeError(
            f"Category '{category_name}' already exists in block with value: {block[category_name]}"
        )

    # Create and add the category--remove any None values, CIF requires non-null values
    category = CIFCategory(
        columns={k: _normalize_nulls(v) for k, v in data.items()}, name=category_name
    )
    block[category_name] = category


def _normalize_nulls(value: Any) -> Any:
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return ["?" if item is None else item for item in value]
    return "?" if value is None else value


def _output_polymer_entities(
    output_block: CIFBlock,
) -> list[tuple[str, set[str], list[int], list[str]]]:
    """Extract each output entity's unique polymer residue sequence.

    Parameters
    ----------
    output_block : CIFBlock
        Output CIF block containing ``atom_site``.

    Returns
    -------
    list[tuple[str, set[str], list[int], list[str]]]
        Entity ID, label chain IDs, residue numbers, and residue names.

    Raises
    ------
    ValueError
        If an entity has no polymer residues or conflicting residue names.
    """
    atom_site = output_block["atom_site"]
    entity_ids = atom_site["label_entity_id"].as_array(str)
    chain_ids = atom_site["label_asym_id"].as_array(str)
    sequence_ids = atom_site["label_seq_id"].as_array(str)
    residue_names = atom_site["label_comp_id"].as_array(str)
    entities: list[tuple[str, set[str], list[int], list[str]]] = []
    for entity_id in dict.fromkeys(entity_ids):
        residues: dict[int, str] = {}
        chains: set[str] = set()
        for row_entity, chain_id, sequence_id, residue_name in zip(
            entity_ids, chain_ids, sequence_ids, residue_names, strict=True
        ):
            if row_entity != entity_id:
                continue
            chains.add(str(chain_id))
            if sequence_id in (".", "?", ""):
                continue
            number = int(sequence_id)
            if number in residues and residues[number] != residue_name:
                raise ValueError(
                    f"Output entity {entity_id} has conflicting names for residue {number}"
                )
            residues[number] = str(residue_name)
        numbers = sorted(residues)
        if not numbers:
            raise ValueError(f"Output entity {entity_id} has no polymer residues")
        entities.append((str(entity_id), chains, numbers, [residues[number] for number in numbers]))
    return entities


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
    """
    columns = {name: [] for name in rows[0]}
    for row in rows:
        for name, values in row.items():
            columns[name].extend(values)
    return columns


def carry_polymer_entity_categories(
    output: CIFFile,
    reference: str | Path | CIFFile,
) -> tuple[str, ...]:
    """Carry consistent deposited polymer entity categories into an output CIF.

    The output's residue sequence is matched to the deposited sequence for each
    entity. Deposited numbering is shifted only when needed to match the output's
    ``label_seq_id`` values. The output is modified only after every entity has
    matched and validated successfully.

    Parameters
    ----------
    output : CIFFile
        Single-block output CIF containing ``atom_site``.
    reference : str | Path | CIFFile
        Deposited reference CIF or its path.

    Returns
    -------
    tuple[str, ...]
        Names of the categories written to ``output``.

    Raises
    ------
    ValueError
        If required categories are absent or any output entity cannot be matched.
    """
    output_block = _single_block(output)
    reference_file = reference if isinstance(reference, CIFFile) else CIFFile.read(str(reference))
    reference_block = _single_block(reference_file)
    missing = [name for name in _POLYMER_ENTITY_CATEGORIES if name not in reference_block]
    if missing:
        raise ValueError(f"Reference CIF lacks required categories: {missing}")

    entity_rows: list[dict[str, list[str]]] = []
    entity_poly_rows: list[dict[str, list[str]]] = []
    sequence_rows: list[dict[str, list[str]]] = []
    struct_asym = {"id": [], "entity_id": []}
    for output_id, chains, numbers, names in _output_polymer_entities(output_block):
        reference_id, offset = _matching_entity(reference_block, numbers, names)
        entity_row = _select_category_rows(reference_block["entity"], "id", reference_id)
        entity_row["id"] = [output_id]
        entity_rows.append(entity_row)

        entity_poly_row = _select_category_rows(
            reference_block["entity_poly"], "entity_id", reference_id
        )
        entity_poly_row["entity_id"] = [output_id]
        if "pdbx_strand_id" in entity_poly_row:
            entity_poly_row["pdbx_strand_id"] = [",".join(sorted(chains))]
        entity_poly_rows.append(entity_poly_row)

        sequence_row = _select_category_rows(
            reference_block["entity_poly_seq"], "entity_id", reference_id
        )
        sequence_row["entity_id"] = [output_id] * len(sequence_row["entity_id"])
        sequence_row["num"] = [str(int(number) + offset) for number in sequence_row["num"]]
        carried_residues = dict(
            zip(map(int, sequence_row["num"]), sequence_row["mon_id"], strict=True)
        )
        output_residues = zip(numbers, names, strict=True)
        if any(carried_residues.get(number) != name for number, name in output_residues):
            raise ValueError(f"Carried sequence does not align with output entity {output_id}")
        sequence_rows.append(sequence_row)
        struct_asym["id"].extend(sorted(chains))
        struct_asym["entity_id"].extend([output_id] * len(chains))

    categories = {
        "entity": _concatenate_category_rows(entity_rows),
        "entity_poly": _concatenate_category_rows(entity_poly_rows),
        "entity_poly_seq": _concatenate_category_rows(sequence_rows),
        "struct_asym": struct_asym,
    }
    for name, data in categories.items():
        add_category_to_cif(output, data, name, overwrite=True)
    return tuple(categories)
