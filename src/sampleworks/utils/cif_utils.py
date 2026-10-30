import itertools
import re
import tempfile
from collections import Counter, OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from atomworks.io.transforms.categories import category_to_dict
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArrayStack
from biotite.structure.info import get_from_ccd
from biotite.structure.io.pdbx.cif import CIFBlock, CIFCategory, CIFFile
from loguru import logger

from sampleworks.utils.atom_array_utils import (
    _closest_canonical_amino_acid,
    BLANK_ALTLOC_IDS,
    find_all_altloc_ids,
    save_structure_to_cif,
    select_altloc,
)
from sampleworks.utils.sequence import align_to_sequence


# Extended PDB IDs use `pdb_` plus eight alphanumerics. Legacy IDs have four
# alphanumerics and start with a digit, which excludes folder names such as `TEST`.
RCSB_ID_PATTERN = re.compile(r"pdb_[A-Za-z0-9]{8}|[0-9][A-Za-z0-9]{3}")

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

    A multi-model ensemble that keeps altloc annotation is treated as one conformer per model.
    This means that atoms with non-blank altloc ID in the first model conceptually corresponds
    to altloc A atoms. Its spans are the contiguous runs of residues with non-blank altlocs.

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
    if isinstance(structure, AtomArrayStack) and structure.stack_depth() > 1:
        # Multi-model ensemble: one conformer per model with altloc annotation. Atoms share the
        # altloc annotation across models (e.g. an atom cannot have blank altloc in model 1 but
        # altloc "X" in model 2).
        altk_mask = ~np.isin(structure.altloc_id, list(BLANK_ALTLOC_IDS))
        residues = sorted(
            set(zip(structure.chain_id[altk_mask], structure.res_id[altk_mask].tolist()))
        )
        for model_index in range(structure.stack_depth()):
            altlocs[f"model_{model_index}"] = residues
    else:
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
        current_membership = {k for k in chain_altlocs if start in chain_altlocs[k]}
        start = start if len(current_membership) > 1 else None
        for current_res_id, next_res_id in itertools.pairwise(all_res_ids):
            res_membership = {k for k in chain_altlocs if next_res_id in chain_altlocs[k]}
            if res_membership != current_membership or next_res_id - current_res_id > 1:
                if start is not None:
                    yield chain, start, current_res_id, current_membership

                start = next_res_id if len(res_membership) > 1 else None
                current_membership = res_membership if len(res_membership) > 1 else None
        # Close the open span at the chain's last altloc residue. When the chain has only
        # one altloc residue the pairwise loop never runs, so the end cannot come from it.
        if start is not None:
            yield chain, start, all_res_ids[-1], current_membership


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
    category = cif_file.block[_ATOM_SITE]
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
        # Microheterogeneity lists several monomers at one num (e.g. CYS/CSO, or a point
        # mutation ASN/ASP): one alignment position, any of its monomers may match.
        nums = sorted({str(num) for num in sequence_nums[rows]}, key=lambda num: int(num))
        reference_names = [
            [str(name) for name in sequence_names[rows][sequence_nums[rows] == num]] for num in nums
        ]
        # The output's own numbering is not the deposit's here, so it places no gaps.
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


def _is_polymer_row(atom_site: CIFCategory, reference_block: CIFBlock) -> np.ndarray:
    """Mark the ``atom_site`` rows that belong to a polymer chain.

    ``ATOM`` rows always do. A ``HETATM`` row does only when the deposit's ``entity_poly_seq``
    lists its monomer (a modified residue such as MSE or LYR), compared after canonicalizing
    both sides. Ligands and waters are outside the polymer whatever their author number, which
    can coincide with a deposited sequence position (6b8x numbers its glycerols 301-303,
    inside chain A's 1-321).

    Parameters
    ----------
    atom_site : CIFCategory
        Output ``atom_site``.
    reference_block : CIFBlock
        Deposited block with ``entity_poly_seq``.

    Returns
    -------
    np.ndarray
        Boolean mask, one entry per row.
    """
    monomers = {
        _canonical_monomer(str(name))
        for name in reference_block[_ENTITY_POLY_SEQ]["mon_id"].as_array(str)
    }
    is_atom = atom_site["group_PDB"].as_array(str) == "ATOM"
    names = atom_site["label_comp_id"].as_array(str)
    return is_atom | np.array([_canonical_monomer(str(name)) in monomers for name in names])


def _output_polymer_chains(
    output_block: CIFBlock, reference_block: CIFBlock
) -> list[tuple[str, list[tuple[int, str]], list[str]]]:
    """Extract each output chain's unique polymer residue sequence.

    Chains are taken one by one, not merged per entity: a model numbers each chain from 1, so
    two copies of one entity resolved over different ranges share numbers but not residues.

    Parameters
    ----------
    output_block : CIFBlock
        Output CIF block containing ``atom_site``.
    reference_block : CIFBlock
        Deposited block whose ``entity_poly_seq`` says which ``HETATM`` monomers are polymer.

    Returns
    -------
    list[tuple[str, list[tuple[int, str]], list[str]]]
        Label chain ID, residue ``(number, insertion code)`` keys, and residue names. Keys are
        ordered by number, and residues sharing a number keep their order of appearance
        (``14, 14A, 14B``, but also thrombin's ``1H ... 1A, 1``). Chains without polymer
        residues (ligands, waters) are omitted.

    Raises
    ------
    ValueError
        If no chain has polymer residues, or a chain has conflicting residue names.
    """
    atom_site = output_block[_ATOM_SITE]
    chain_ids = atom_site["label_asym_id"].as_array(str)
    sequence_ids = atom_site["label_seq_id"].as_array(str)
    insertion_codes = _blank_to_empty(atom_site["pdbx_PDB_ins_code"].as_array(str))
    residue_names = atom_site["label_comp_id"].as_array(str)
    polymer_rows = _is_polymer_row(atom_site, reference_block)
    chains: list[tuple[str, list[tuple[int, str]], list[str]]] = []
    for chain in dict.fromkeys(chain_ids):
        residues: dict[tuple[int, str], str] = {}
        for chain_id, sequence_id, insertion_code, residue_name, polymer in zip(
            chain_ids, sequence_ids, insertion_codes, residue_names, polymer_rows, strict=True
        ):
            if chain_id != chain or not polymer or sequence_id in (".", "?", ""):
                continue
            key = (int(sequence_id), str(insertion_code))
            if key in residues and residues[key] != residue_name:
                raise ValueError(f"Output chain {chain} has conflicting names for residue {key}")
            residues[key] = str(residue_name)
        keys = sorted(residues, key=lambda key: key[0])  # stable: same-number residues keep order
        if not keys:
            # Ligands and waters have no entity_poly* rows to carry; the polymer chains
            # alongside them still do.
            continue
        chains.append((str(chain), keys, [residues[key] for key in keys]))
    if not chains:
        raise ValueError("Output has no polymer chains")
    return chains


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


class SchemePosition(NamedTuple):
    """One deposited polymer position, a ``pdbx_poly_seq_scheme`` row.

    Attributes
    ----------
    asym_id, seq_id, entity_id : str
        Label chain, sequence position and entity.
    strand_id, number, ins_code : str
        Author chain, number and insertion code (``""`` when absent).
    monomers : frozenset[str]
        Canonical monomer names: several for a microheterogeneity site (e.g. CYS/CSO), whose
        scheme rows share the position.
    """

    asym_id: str
    seq_id: str
    entity_id: str
    strand_id: str
    number: str
    ins_code: str
    monomers: frozenset[str]

    @property
    def label(self) -> _Label:
        """``(label_asym_id, label_seq_id, label_entity_id)`` of the position."""
        return (self.asym_id, self.seq_id, self.entity_id)


def scheme_positions(reference: CIFFile | CIFBlock) -> list[SchemePosition]:
    """List every deposited polymer position, resolved or not, in scheme order.

    Parameters
    ----------
    reference : CIFFile | CIFBlock
        Deposited CIF, or its block, with ``pdbx_poly_seq_scheme``.

    Returns
    -------
    list[SchemePosition]
        One entry per ``(asym_id, seq_id)``; microheterogeneity rows merge into ``monomers``.
    """
    block = _single_block(reference) if isinstance(reference, CIFFile) else reference
    scheme = category_to_dict(block, _PDBX_POLY_SEQ_SCHEME)
    positions: dict[tuple[str, str], SchemePosition] = {}
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
        position = positions.get((asym, seq)) or SchemePosition(
            asym, seq, entity, strand, number, ins_code, frozenset()
        )
        monomers = position.monomers | {_canonical_monomer(monomer)}
        positions[(asym, seq)] = position._replace(monomers=monomers)
    return list(positions.values())


def _labels_from_scheme(
    output_block: CIFBlock, reference_block: CIFBlock, by_label: bool = False
) -> list[_Label | None]:
    """Label output residues by looking their ids up in the deposit's scheme.

    Parameters
    ----------
    output_block : CIFBlock
        Output block whose ``atom_site`` keeps the deposit's author numbering, or its label
        numbering when ``by_label``.
    reference_block : CIFBlock
        Deposited block with ``pdbx_poly_seq_scheme``.
    by_label : bool
        Look up ``(label_asym_id, label_seq_id)`` as the scheme's ``(asym_id, seq_id)``
        instead of the author ``(chain, number, insertion code)`` as its
        ``(pdb_strand_id, pdb_seq_num, pdb_ins_code)``.

    Returns
    -------
    list[_Label | None]
        One label per ``atom_site`` row, ``None`` for rows outside the scheme (e.g. ligands).

    Raises
    ------
    ValueError
        If an ``ATOM`` residue has no scheme row, or a residue's name disagrees with its
        scheme row.
    """
    output_columns = (
        ("label_asym_id", "label_seq_id")
        if by_label
        else ("auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code")
    )
    lookup: dict[tuple[str, ...], SchemePosition] = {
        ((p.asym_id, p.seq_id) if by_label else (p.strand_id, p.number, p.ins_code)): p
        for p in scheme_positions(reference_block)
    }

    atom_site = output_block[_ATOM_SITE]
    residue_keys = list(
        zip(*(_blank_to_empty(atom_site[column].as_array(str)) for column in output_columns))
    )
    residue_names = atom_site["label_comp_id"].as_array(str)
    is_atom = atom_site["group_PDB"].as_array(str) == "ATOM"
    polymer_rows = _is_polymer_row(atom_site, reference_block)
    unmatched = {key for key, atom in zip(residue_keys, is_atom) if atom and key not in lookup}
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} residue(s) have no deposited pdbx_poly_seq_scheme row, "
            f"e.g. {sorted(unmatched)[:3]}"
        )
    renamed = {
        (key, name)
        for key, name, polymer in zip(residue_keys, residue_names, polymer_rows)
        if polymer and key in lookup and _canonical_monomer(name) not in lookup[key].monomers
    }
    if renamed:
        raise ValueError(f"Residue names disagree with the deposit, e.g. {sorted(renamed)[:3]}")
    return [
        lookup[key].label if polymer and key in lookup else None
        for key, polymer in zip(residue_keys, polymer_rows)
    ]


def _require_one_residue_per_label(labels: list[_Label | None], atom_site: CIFCategory) -> None:
    """Check that no two output residues take the same deposited position and monomer.

    Two residues the lookup could not tell apart (e.g. one the model built, numbered like a
    resolved one) give one label, monomer and model to two atoms of the same name. Different
    monomers at one position are the deposit's microheterogeneity (a CYS/CSO site) and are
    allowed, as are altlocs.

    Parameters
    ----------
    labels : list[_Label | None]
        Deposit label per ``atom_site`` row, ``None`` for non-polymer rows.
    atom_site : CIFCategory
        Output ``atom_site``.

    Raises
    ------
    ValueError
        If two atoms share a label, monomer, model, atom name and altloc.
    """
    atoms = Counter(
        atom
        for atom in zip(
            labels,
            *(
                atom_site[column].as_array(str)
                for column in (
                    "label_comp_id",
                    "pdbx_PDB_model_num",
                    "label_atom_id",
                    "label_alt_id",
                )
            ),
            strict=True,
        )
        if atom[0] is not None
    )
    shared = sorted({atom[:2] for atom, count in atoms.items() if count > 1})
    if shared:
        raise ValueError(
            f"{len(shared)} deposited position(s) name several residues, e.g. {shared[:3]}"
        )


def _labels_from_sequence_alignment(
    output_block: CIFBlock, reference_block: CIFBlock
) -> list[_Label | None]:
    """Label output residues by aligning each output chain's sequence to the deposit's.

    Used when the output keeps neither the deposit's author nor its label numbering (a model
    that renumbers residues, or residues the model built numbered past the deposit's
    numbering). Each residue gets the deposited ``entity_poly_seq.num`` it aligns to as
    ``label_seq_id`` and the deposited entity id. Each output chain becomes the deposited chain
    of that entity whose ``asym_id`` or author ``pdb_strand_id`` it carries: an input read from
    an RCSB CIF brings label chain ids, a density input author ones.

    Parameters
    ----------
    output_block : CIFBlock
        Output block containing ``atom_site``.
    reference_block : CIFBlock
        Deposited block with ``entity_poly``, ``entity_poly_seq`` and ``pdbx_poly_seq_scheme``.

    Returns
    -------
    list[_Label | None]
        One label per ``atom_site`` row, ``None`` for non-polymer rows.

    Raises
    ------
    ValueError
        If an output chain has no unique match among the deposited entities, or does not name
        exactly one deposited chain of its entity, or two output chains name the same one.
    """
    deposited_chains = {
        (p.asym_id, p.strand_id, p.entity_id) for p in scheme_positions(reference_block)
    }
    remap: dict[tuple[str, str, str], _Label] = {}  # (chain, seq, insertion code) -> label
    claimed: dict[str, str] = {}  # deposited asym_id -> output chain
    for chain, keys, names in _output_polymer_chains(output_block, reference_block):
        entity_id, nums = _matching_entity(reference_block, names)
        asyms = {
            asym
            for asym, strand, entity in deposited_chains
            if entity == entity_id and chain in (asym, strand)
        }
        if len(asyms) != 1:
            raise ValueError(
                f"Output chain {chain} names deposited chains {sorted(asyms)} of entity "
                f"{entity_id}, not exactly one"
            )
        asym = asyms.pop()
        if asym in claimed:
            raise ValueError(f"Output chains {claimed[asym]} and {chain} both name chain {asym}")
        claimed[asym] = chain
        for (number, insertion_code), num in zip(keys, nums, strict=True):
            remap[(chain, str(number), insertion_code)] = (asym, num, entity_id)

    atom_site = output_block[_ATOM_SITE]
    # A ligand numbered like a sequence position is not that residue.
    return [
        remap.get(key) if polymer else None
        for key, polymer in zip(
            zip(
                atom_site["label_asym_id"].as_array(str),
                atom_site["label_seq_id"].as_array(str),
                _blank_to_empty(atom_site["pdbx_PDB_ins_code"].as_array(str)),
                strict=True,
            ),
            _is_polymer_row(atom_site, reference_block),
            strict=True,
        )
    ]


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


def carry_polymer_entity_categories(
    output: CIFFile,
    reference: str | Path | CIFFile,
) -> tuple[str, ...]:
    """Carry deposited polymer categories into an output CIF and match its residue ids to them.

    ``set_structure`` writes the same values into the label and author ids, in whichever
    numbering the input used. Each output residue is looked up in the deposit's
    ``pdbx_poly_seq_scheme`` by those values as its author ``(chain, number, insertion code)``,
    or failing that as its label ``(chain, sequence position)``. When neither matches every
    residue (a model that renumbers, or residues the model built numbered past the deposit's
    numbering), each output chain's sequence is aligned to the deposited ones and mapped onto
    the deposited ``entity_poly_seq`` numbering, the chain becoming the deposited chain whose
    label or author id it carries. Every polymer residue then takes its label chain, sequence
    position and entity, and its author chain, number and insertion code, from its scheme row,
    so residues the model built that the experiment did not resolve get the deposit's numbers
    too. The deposit's polymer categories and scheme are carried for the chains present, the
    scheme's ``auth_seq_num`` filled for every modeled position. A ligand or water the deposit's
    ``pdbx_nonpoly_scheme`` lists under the same author number, insertion code and monomer,
    and under the label or author chain the row carries, takes that row's chain, entity and
    author ids, and its ``pdbx_nonpoly_scheme``, ``struct_asym`` and ``entity`` rows are
    carried; one the deposit lacks keeps its own. Where the model writes the
    canonical parent of a deposited modified residue (MET for MSE), the modeled monomer is
    added to ``entity_poly_seq`` as a
    heterogeneity row beside the deposit's, and ``chem_comp`` rows are carried for every
    monomer either names. A monomer the deposit does not list gets its row from biotite's
    bundled Chemical Component Dictionary, in the deposit's columns (a column the dictionary
    lacks is ``?``); one the dictionary lacks too is warned about, not fatal. The output is
    modified only after every residue has been matched and validated.

    Parameters
    ----------
    output : CIFFile
        Single-block output CIF containing ``atom_site``. Side effect: its polymer and
        deposited ligand rows' label and author ids are rewritten, its non-polymer rows'
        ``label_seq_id`` becomes ``.``,
        ``struct_conn`` partners follow both, and the carried categories are added or replaced.
    reference : str | Path | CIFFile
        Deposited reference CIF or its path.

    Returns
    -------
    tuple[str, ...]
        Names of the categories written to ``output``.

    Raises
    ------
    ValueError
        If required categories are absent, or neither lookup nor the sequence alignment
        matches every polymer residue.
    """
    output_block = _single_block(output)
    reference_file = reference if isinstance(reference, CIFFile) else CIFFile.read(str(reference))
    reference_block = _single_block(reference_file)
    missing = [name for name in _POLYMER_ENTITY_CATEGORIES if name not in reference_block]
    if missing:
        raise ValueError(f"Reference CIF lacks required categories: {missing}")

    labels: list[_Label | None] | None = None
    errors: list[str] = []
    for method, find_labels in (
        ("author lookup", lambda: _labels_from_scheme(output_block, reference_block)),
        ("label lookup", lambda: _labels_from_scheme(output_block, reference_block, True)),
        (
            "sequence alignment",
            lambda: _labels_from_sequence_alignment(output_block, reference_block),
        ),
    ):
        try:
            found = find_labels()
            _require_one_residue_per_label(found, output_block[_ATOM_SITE])
        except ValueError as error:
            errors.append(f"{method}: {error}")
            continue
        labels = found
        break
    if labels is None:
        raise ValueError("; ".join(errors))
    polymer_labels = {label for label in labels if label is not None}
    if not polymer_labels:
        raise ValueError("Output has no polymer residues")

    # Each other row (ligand, water) is the deposit's pdbx_nonpoly_scheme row with its author
    # number, insertion code and monomer, under the label or author chain the row carries.
    atom_site = output_block[_ATOM_SITE]
    nonpoly = category_to_dict(reference_block, _PDBX_NONPOLY_SCHEME)  # empty without one
    candidates: dict[tuple[str, ...], set[int]] = {}  # (chain, number, code, monomer) -> rows
    if nonpoly:
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
    ligands: dict[int, int] = {}  # atom_site row -> pdbx_nonpoly_scheme row
    for row, (label, *key) in enumerate(
        zip(
            labels,
            atom_site["auth_asym_id"].as_array(str),
            atom_site["auth_seq_id"].as_array(str),
            _blank_to_empty(atom_site["pdbx_PDB_ins_code"].as_array(str)),
            atom_site["label_comp_id"].as_array(str),
            strict=True,
        )
    ):
        matches = candidates.get(tuple(key), set()) if label is None else set()
        if len(matches) > 1:
            raise ValueError(f"{key[3]} {key[0]} {key[1]}{key[2]} names several deposited ligands")
        if matches:
            (ligands[row],) = matches
    ligand_rows = sorted(set(ligands.values()))

    asym_ids = {asym_id for asym_id, _, _ in polymer_labels}
    entity_ids = {entity_id for _, _, entity_id in polymer_labels}
    categories = {
        name: _concatenate_category_rows(
            [_select_category_rows(reference_block[name], column, value) for value in sorted(keep)]
        )
        for name, column, keep in (
            (_ENTITY, "id", entity_ids | {nonpoly["entity_id"][row] for row in ligand_rows}),
            (_ENTITY_POLY, "entity_id", entity_ids),
            (_ENTITY_POLY_SEQ, "entity_id", entity_ids),
            (_STRUCT_ASYM, "id", asym_ids | {nonpoly["asym_id"][row] for row in ligand_rows}),
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
    # (asym_id, seq_id) -> author (chain, number, insertion code token)
    author_ids = {
        (p.asym_id, p.seq_id): (p.strand_id, p.number, p.ins_code or _INAPPLICABLE)
        for p in scheme_positions(reference_block)
    }

    _add_modeled_monomers(categories[_ENTITY_POLY_SEQ], atom_site, labels)
    if _CHEM_COMP in reference_block:
        # Every monomer named by atom_site or by the carried sequence needs a chem_comp parent.
        components = set(atom_site["label_comp_id"].as_array(str)) | set(
            categories[_ENTITY_POLY_SEQ]["mon_id"]
        )
        chem_comp = reference_block[_CHEM_COMP]
        reference_components = np.asarray(chem_comp["id"].as_array(str))
        component_mask = np.isin(reference_components, list(components))
        chem_comp_rows = {
            column: list(np.asarray(chem_comp[column].as_array(str))[component_mask])
            for column in chem_comp
        }
        # A monomer the deposit does not list (MET beside a deposited MSE) takes its row from
        # biotite's bundled CCD, in the deposit's columns; a column the CCD lacks is unknown.
        unlisted_components = sorted(
            map(str, components - set(reference_components[component_mask]))
        )
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
        categories[_CHEM_COMP] = chem_comp_rows

    # One (label asym, seq, entity, author chain, number, insertion code) per row: a polymer
    # residue's from its scheme position, a deposited ligand's from its nonpoly row, any other
    # row (ligand, water) its own without a sequence position. Insertion codes are CIF tokens
    # ("." when absent): biotite matches struct_conn partners to atom_site on the raw token.
    id_columns = (
        "label_asym_id",
        "label_seq_id",
        "label_entity_id",
        "auth_asym_id",
        "auth_seq_id",
        "pdbx_PDB_ins_code",
    )
    old_ids = list(zip(*(atom_site[column].as_array(str) for column in id_columns), strict=True))
    new_ids: list[tuple[str, ...]] = []
    for row, (label, (asym, _, entity, chain, number, code)) in enumerate(
        zip(labels, old_ids, strict=True)
    ):
        if label is not None:
            new_ids.append((*label, *author_ids[label[:2]]))
        elif row in ligands:
            columns = ("asym_id", "entity_id", "pdb_strand_id", "pdb_seq_num", "pdb_ins_code")
            asym, entity, chain, number, code = (nonpoly[c][ligands[row]] for c in columns)
            new_ids.append((asym, _INAPPLICABLE, entity, chain, number, code))
        else:
            code = _INAPPLICABLE if code in ("", "?") else code
            new_ids.append((asym, _INAPPLICABLE, entity, chain, number, code))
    # set_structure names struct_conn partners by the atom_site labels it wrote; map them over.
    relabel = {
        (asym, seq, "" if code in (".", "?") else code, name): (ids[0], ids[1], ids[5])
        for (asym, seq, _, _, _, code), name, ids in zip(
            old_ids, atom_site["label_comp_id"].as_array(str), new_ids, strict=True
        )
    }
    partner_labels: dict[str, np.ndarray] = {}
    if _STRUCT_CONN in output_block:
        struct_conn = output_block[_STRUCT_CONN]
        for partner in (1, 2):
            partners = zip(
                struct_conn[f"ptnr{partner}_label_asym_id"].as_array(str),
                struct_conn[f"ptnr{partner}_label_seq_id"].as_array(str),
                _blank_to_empty(struct_conn[f"pdbx_ptnr{partner}_PDB_ins_code"].as_array(str)),
                struct_conn[f"ptnr{partner}_label_comp_id"].as_array(str),
                strict=True,
            )
            asym_ids, seq_ids, partner_codes = zip(*(relabel[key] for key in partners), strict=True)
            partner_labels[f"ptnr{partner}_label_asym_id"] = np.array(asym_ids)
            partner_labels[f"ptnr{partner}_label_seq_id"] = np.array(seq_ids)
            partner_labels[f"pdbx_ptnr{partner}_PDB_ins_code"] = np.array(partner_codes)

    for index, column in enumerate(id_columns):
        atom_site[column] = np.array([ids[index] for ids in new_ids])
    for column, values in partner_labels.items():
        output_block[_STRUCT_CONN][column] = values
    for name, data in categories.items():
        add_category_to_cif(output, data, name, overwrite=True)
    return tuple(categories)
