import itertools
import tempfile
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io.utils.io_utils import load_any
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArrayStack
from biotite.structure.io.pdbx.cif import CIFBlock, CIFCategory, CIFFile
from loguru import logger

from sampleworks.utils.atom_array_utils import (
    _closest_canonical_amino_acid,
    find_all_altloc_ids,
    save_structure_to_cif,
    select_altloc,
)


_POLYMER_ENTITY_CATEGORIES = ("entity", "entity_poly", "entity_poly_seq")
# The author-number lookup additionally needs these; without them the carry falls back to
# sequence alignment.
_SCHEME_CATEGORIES = ("struct_asym", "pdbx_poly_seq_scheme")


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
    return _closest_canonical_amino_acid(name) or name


def _unique_subsequence_indices(reference: list[str], modeled: list[str]) -> list[int] | None:
    """Find a unique ordered embedding with the fewest deletion runs.

    Cost counts runs of consecutive skipped reference residues, not residues, so a single
    gap of three costs 1 while two separate one-residue gaps cost 2.

    Parameters
    ----------
    reference : list[str]
        Deposited monomer names.
    modeled : list[str]
        Modeled monomer names.

    Returns
    -------
    list[int] | None
        Reference indices for the unique optimal embedding, or ``None`` when absent or ambiguous.
    """
    reference = [_canonical_monomer(name) for name in reference]
    modeled = [_canonical_monomer(name) for name in modeled]
    states: dict[tuple[int, bool], tuple[int, tuple[int, ...] | None, int]] = {
        (0, False): (0, (), 1)
    }

    def update(
        target: dict[tuple[int, bool], tuple[int, tuple[int, ...] | None, int]],
        key: tuple[int, bool],
        cost: int,
        path: tuple[int, ...] | None,
        count: int,
    ) -> None:
        """Keep minimum-cost paths and mark equally optimal alternatives.

        Parameters
        ----------
        target : dict
            Dynamic-programming states for the next reference residue.
        key : tuple[int, bool]
            Modeled position and whether the path is deleting reference residues.
        cost : int
            Number of deletion runs in the candidate path.
        path : tuple[int, ...] | None
            Matched reference indices, or ``None`` for an ambiguous path.
        count : int
            Number of equally optimal paths, capped at two.
        """
        current = target.get(key)
        if current is None or cost < current[0]:
            target[key] = (cost, path, count)
        elif cost == current[0]:
            target[key] = (cost, None, min(2, current[2] + count))

    for reference_index, reference_name in enumerate(reference):
        next_states: dict[tuple[int, bool], tuple[int, tuple[int, ...] | None, int]] = {}
        for (modeled_index, deleting), (cost, path, count) in states.items():
            update(next_states, (modeled_index, True), cost + (not deleting), path, count)
            if modeled_index < len(modeled) and reference_name == modeled[modeled_index]:
                matched_path = None if path is None else (*path, reference_index)
                update(next_states, (modeled_index + 1, False), cost, matched_path, count)
        states = next_states

    candidates = [value for (index, _), value in states.items() if index == len(modeled)]
    if not candidates:
        return None
    minimum_cost = min(value[0] for value in candidates)
    optimal = [value for value in candidates if value[0] == minimum_cost]
    if sum(value[2] for value in optimal) != 1 or optimal[0][1] is None:
        return None
    return list(optimal[0][1])


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
    sequence = reference_block["entity_poly_seq"]
    sequence_ids = np.asarray(sequence["entity_id"].as_array(str))
    sequence_nums = np.asarray(sequence["num"].as_array(str))
    sequence_names = np.asarray(sequence["mon_id"].as_array(str))
    matches: list[tuple[str, list[str]]] = []
    ambiguous: list[str] = []
    for entity_id in reference_block["entity_poly"]["entity_id"].as_array(str):
        rows = np.flatnonzero(sequence_ids == entity_id)
        # Microheterogeneity lists several monomers at one num (e.g. CYS/CSO); keep the first.
        rows = rows[np.unique(sequence_nums[rows], return_index=True)[1]]
        rows = rows[np.argsort(sequence_nums[rows].astype(int), kind="stable")]
        reference_names = list(sequence_names[rows])
        indices = _unique_subsequence_indices(reference_names, residue_names)
        if indices is not None:
            matches.append((str(entity_id), list(sequence_nums[rows][indices])))
            continue
        remaining = iter(_canonical_monomer(name) for name in reference_names)
        if all(_canonical_monomer(name) in remaining for name in residue_names):
            ambiguous.append(str(entity_id))
    if len(matches) == 1:
        return matches[0]
    if matches:
        entity_ids = [entity_id for entity_id, _ in matches]
        raise ValueError(f"Modeled sequence matches several deposited entities: {entity_ids}")
    if ambiguous:
        raise ValueError(
            f"Modeled sequence fits deposited entity {ambiguous} in several equally good ways "
            "(e.g. a gap inside a run of repeated residues)"
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
        Entity ID, label chain IDs, residue numbers, and residue names. Entities without
        polymer residues (ligands, waters) are omitted.

    Raises
    ------
    ValueError
        If no entity has polymer residues, or an entity has conflicting residue names.
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
            # Ligands and waters have no entity_poly* rows to carry; the polymer entities
            # alongside them still do.
            continue
        entities.append((str(entity_id), chains, numbers, [residues[number] for number in numbers]))
    if not entities:
        raise ValueError("Output has no polymer entities")
    return entities


# Unused since carry_polymer_entity_categories looks residues up in pdbx_poly_seq_scheme.
def _one_letter_sequence(residue_names: list[str]) -> str:
    """Convert three-letter residue names to a one-letter sequence.

    Parameters
    ----------
    residue_names : list[str]
        Three-letter residue names in sequence order.

    Returns
    -------
    str
        One-letter sequence.

    Raises
    ------
    ValueError
        If any name has no one-letter representation, for example a nucleotide or a
        modified residue. ``ProteinSequence`` raises ``KeyError`` for those, which would
        otherwise escape the callers that degrade on ``ValueError``.
    """
    letters: list[str] = []
    for name in residue_names:
        try:
            letters.append(ProteinSequence.convert_letter_3to1(name))
        except KeyError:
            raise ValueError(f"Residue {name} has no one-letter representation") from None
    return "".join(letters)


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


_Label = tuple[str, str, str]  # (label_asym_id, label_seq_id, label_entity_id)


def _labels_from_author_numbering(
    output_block: CIFBlock, reference_block: CIFBlock
) -> tuple[list[_Label | None], dict[str, dict[str, list[str]]]]:
    """Label output residues by looking their author ids up in the deposit's scheme.

    Parameters
    ----------
    output_block : CIFBlock
        Output block whose ``atom_site`` keeps the deposit's author numbering.
    reference_block : CIFBlock
        Deposited block with ``pdbx_poly_seq_scheme`` and ``struct_asym``.

    Returns
    -------
    tuple[list[_Label | None], dict[str, dict[str, list[str]]]]
        One label per ``atom_site`` row (``None`` for rows outside the scheme, e.g. ligands),
        and the deposited categories to carry for the chains present.

    Raises
    ------
    ValueError
        If scheme categories are absent, an ``ATOM`` residue has no scheme row, or a
        residue's name disagrees with its scheme row.
    """
    missing = [name for name in _SCHEME_CATEGORIES if name not in reference_block]
    if missing:
        raise ValueError(f"Reference CIF lacks {missing}")
    scheme = reference_block["pdbx_poly_seq_scheme"]
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

    atom_site = output_block["atom_site"]
    residue_keys = list(
        zip(
            atom_site["auth_asym_id"].as_array(str),
            atom_site["auth_seq_id"].as_array(str),
            _blank_to_empty(atom_site["pdbx_PDB_ins_code"].as_array(str)),
        )
    )
    residue_names = atom_site["label_comp_id"].as_array(str)
    is_atom = atom_site["group_PDB"].as_array(str) == "ATOM"
    unmatched = {key for key, atom in zip(residue_keys, is_atom) if atom and key not in lookup}
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} residue(s) have no deposited pdbx_poly_seq_scheme row, "
            f"e.g. {sorted(unmatched)[:3]}"
        )
    renamed = {
        (key, name)
        for key, name in zip(residue_keys, residue_names)
        if key in lookup and _canonical_monomer(name) not in monomers[key]
    }
    if renamed:
        raise ValueError(f"Residue names disagree with the deposit, e.g. {sorted(renamed)[:3]}")
    labels = [lookup.get(key) for key in residue_keys]
    polymer_labels = {label for label in labels if label is not None}
    if not polymer_labels:
        raise ValueError("Output has no polymer residues")

    asym_ids = {asym_id for asym_id, _, _ in polymer_labels}
    entity_ids = {entity_id for _, _, entity_id in polymer_labels}
    categories = {
        name: _concatenate_category_rows(
            [_select_category_rows(reference_block[name], column, value) for value in sorted(keep)]
        )
        for name, column, keep in (
            ("entity", "id", entity_ids),
            ("entity_poly", "entity_id", entity_ids),
            ("entity_poly_seq", "entity_id", entity_ids),
            ("struct_asym", "id", asym_ids),
            ("pdbx_poly_seq_scheme", "asym_id", asym_ids),
        )
    }
    carried_scheme = categories["pdbx_poly_seq_scheme"]
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
    return labels, categories


def _labels_from_sequence_alignment(
    output_block: CIFBlock, reference_block: CIFBlock
) -> tuple[list[_Label | None], dict[str, dict[str, list[str]]]]:
    """Label output residues by aligning each output entity's sequence to the deposit's.

    Used when the output does not keep the deposit's author numbering (a model that renumbers
    residues, or residues the input never had). Each residue gets the deposited
    ``entity_poly_seq.num`` it aligns to as ``label_seq_id`` and the deposited entity id;
    ``label_asym_id`` keeps the output chain, so ``pdbx_poly_seq_scheme`` is not carried.

    Parameters
    ----------
    output_block : CIFBlock
        Output block containing ``atom_site``.
    reference_block : CIFBlock
        Deposited block with ``entity``, ``entity_poly`` and ``entity_poly_seq``.

    Returns
    -------
    tuple[list[_Label | None], dict[str, dict[str, list[str]]]]
        One label per ``atom_site`` row (``None`` for non-polymer rows), and the deposited
        categories to carry.

    Raises
    ------
    ValueError
        If an output entity has no unique match among the deposited entities.
    """
    remap: dict[tuple[str, str], tuple[str, str]] = {}  # (entity, seq) -> (deposit entity, num)
    chain_entities: dict[str, str] = {}
    for output_id, chains, numbers, names in _output_polymer_entities(output_block):
        reference_id, reference_nums = _matching_entity(reference_block, names)
        for number, num in zip(numbers, reference_nums, strict=True):
            remap[(output_id, str(number))] = (reference_id, num)
        chain_entities.update(dict.fromkeys(chains, reference_id))

    atom_site = output_block["atom_site"]
    labels: list[_Label | None] = []
    for chain, entity, seq in zip(
        atom_site["label_asym_id"].as_array(str),
        atom_site["label_entity_id"].as_array(str),
        atom_site["label_seq_id"].as_array(str),
        strict=True,
    ):
        match = remap.get((entity, seq))
        labels.append(None if match is None else (chain, match[1], match[0]))

    entity_ids = sorted(set(chain_entities.values()))
    categories = {
        name: _concatenate_category_rows(
            [_select_category_rows(reference_block[name], column, value) for value in entity_ids]
        )
        for name, column in (
            ("entity", "id"),
            ("entity_poly", "entity_id"),
            ("entity_poly_seq", "entity_id"),
        )
    }
    categories["struct_asym"] = {
        "id": sorted(chain_entities),
        "entity_id": [chain_entities[chain] for chain in sorted(chain_entities)],
    }
    return labels, categories


def carry_polymer_entity_categories(
    output: CIFFile,
    reference: str | Path | CIFFile,
) -> tuple[str, ...]:
    """Carry deposited polymer categories into an output CIF and match its label ids to them.

    ``set_structure`` writes author values into ``label_asym_id`` and ``label_seq_id``. Each
    output residue is first looked up by its author ``(chain, number, insertion code)`` in the
    deposit's ``pdbx_poly_seq_scheme``; its label chain, sequence position and entity are taken
    from that row, and the deposit's polymer categories are carried unchanged for the chains
    present. When the output does not keep the deposit's author numbering, each output entity's
    sequence is aligned to the deposited ones instead and mapped onto the deposited
    ``entity_poly_seq`` numbering. The output is modified only after every residue has been
    matched and validated.

    Parameters
    ----------
    output : CIFFile
        Single-block output CIF containing ``atom_site``. Side effect: its polymer rows'
        label ids are rewritten and the carried categories are added or replaced.
    reference : str | Path | CIFFile
        Deposited reference CIF or its path.

    Returns
    -------
    tuple[str, ...]
        Names of the categories written to ``output``.

    Raises
    ------
    ValueError
        If required categories are absent, or neither the author-number lookup nor the
        sequence alignment matches every polymer residue.
    """
    output_block = _single_block(output)
    reference_file = reference if isinstance(reference, CIFFile) else CIFFile.read(str(reference))
    reference_block = _single_block(reference_file)
    missing = [name for name in _POLYMER_ENTITY_CATEGORIES if name not in reference_block]
    if missing:
        raise ValueError(f"Reference CIF lacks required categories: {missing}")

    try:
        labels, categories = _labels_from_author_numbering(output_block, reference_block)
    except ValueError as lookup_error:
        try:
            labels, categories = _labels_from_sequence_alignment(output_block, reference_block)
        except ValueError as alignment_error:
            raise ValueError(
                f"author lookup: {lookup_error}; sequence alignment: {alignment_error}"
            ) from alignment_error

    atom_site = output_block["atom_site"]
    if "chem_comp" in reference_block:
        modeled_components = set(atom_site["label_comp_id"].as_array(str))
        chem_comp = reference_block["chem_comp"]
        reference_components = np.asarray(chem_comp["id"].as_array(str))
        component_mask = np.isin(reference_components, list(modeled_components))
        missing_components = modeled_components - set(reference_components[component_mask])
        if missing_components:
            raise ValueError(f"Reference CIF lacks chem_comp rows for {sorted(missing_components)}")
        categories["chem_comp"] = {
            column: list(np.asarray(chem_comp[column].as_array(str))[component_mask])
            for column in chem_comp
        }

    # HETATM rows outside the scheme (ligands, waters) keep what set_structure wrote.
    for column, index in (("label_asym_id", 0), ("label_seq_id", 1), ("label_entity_id", 2)):
        current = atom_site[column].as_array(str)
        atom_site[column] = np.array(
            [value if label is None else label[index] for value, label in zip(current, labels)]
        )
    for name, data in categories.items():
        add_category_to_cif(output, data, name, overwrite=True)
    return tuple(categories)
