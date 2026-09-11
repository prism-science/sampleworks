"""Sanitize residue records, altlocs, occupancies, and incomplete conformers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from biotite.structure import AtomArray, superimpose
from biotite.structure.info import residue as get_residue_template
from biotite.structure.io.pdbx.cif import CIFBlock

from sampleworks.utils.atom_array_utils import BLANK_ALTLOC_IDS, closest_canonical_residue_name
from sampleworks.utils.cif_utils import AtomRecord, map_category_values, ResidueRecords


ATOM_RENAMES = {("MSE", "MET", "SE"): "SD"}


@dataclass(frozen=True)
class ResidueSanitizationResult:
    """Result of sanitizing one residue position.

    Attributes
    ----------
    atom_records : list[AtomRecord]
        Canonicalized atom records, including any added atoms.
    replacements : dict[str, str]
        Source-to-canonical component names.
    canonicalized : int
        Number of residue positions canonicalized.
    added : int
        Number of atoms added.
    removed : int
        Number of noncanonical atoms removed.
    skipped : int
        Number of conformers that could not be oriented.
    demoted : int
        Number of lone-altloc atoms moved to the blank altloc.
    occupancy_normalized : int
        Number of blank-altloc atom occupancies changed to 1.0.
    """

    atom_records: list[AtomRecord]
    replacements: dict[str, str]
    canonicalized: int = 0
    added: int = 0
    removed: int = 0
    skipped: int = 0
    demoted: int = 0
    occupancy_normalized: int = 0


def sanitize_residues(
    residue_records: ResidueRecords,
    *,
    canonicalize_residues: bool,
    add_missing_atoms: bool,
    demote_lone_altlocs: bool,
    set_blank_altlocs_full_occupancy: bool,
) -> tuple[list[AtomRecord], dict[str, str], dict[str, int]]:
    """Sanitize all residue positions and aggregate their summaries.

    Parameters
    ----------
    residue_records : ResidueRecords
        Atom records keyed by residue position and model.
    canonicalize_residues : bool
        Whether to convert residues to canonical atom schemas.
    add_missing_atoms : bool
        Whether to add missing heavy atoms after canonicalization.
    demote_lone_altlocs : bool
        Whether to move atoms represented under only one altloc to the blank altloc.
    set_blank_altlocs_full_occupancy : bool
        Whether to set blank-altloc atom occupancies to 1.0.

    Returns
    -------
    tuple[list[AtomRecord], dict[str, str], dict[str, int]]
        Output atom records, component replacements, and sanitization counts.
    """
    next_id = next_atom_id(residue_records) if add_missing_atoms else 1
    output_atom_records: list[AtomRecord] = []
    replacements: dict[str, str] = {}
    counts = {
        "canonicalized": 0,
        "added": 0,
        "removed": 0,
        "skipped": 0,
        "demoted": 0,
        "occupancy_normalized": 0,
    }

    for atom_records in residue_records.values():
        result = sanitize_residue(
            atom_records,
            first_id=next_id,
            canonicalize_residues=canonicalize_residues,
            add_missing_atoms=add_missing_atoms,
            demote_lone_altlocs=demote_lone_altlocs,
            set_blank_altlocs_full_occupancy=set_blank_altlocs_full_occupancy,
        )
        output_atom_records.extend(result.atom_records)
        replacements.update(result.replacements)
        counts["canonicalized"] += result.canonicalized
        counts["added"] += result.added
        counts["removed"] += result.removed
        counts["skipped"] += result.skipped
        counts["demoted"] += result.demoted
        counts["occupancy_normalized"] += result.occupancy_normalized
        next_id += result.added

    return output_atom_records, replacements, counts


def sanitize_residue(
    atom_records: list[AtomRecord],
    *,
    first_id: int,
    canonicalize_residues: bool,
    add_missing_atoms: bool,
    demote_lone_altlocs: bool,
    set_blank_altlocs_full_occupancy: bool,
) -> ResidueSanitizationResult:
    """Canonicalize one residue and optionally add its missing heavy atoms.

    Parameters
    ----------
    atom_records : list[AtomRecord]
        Atom-site records for one residue position and model.
    first_id : int
        First identifier available for a new atom.
    canonicalize_residues : bool
        Whether to convert this residue to its canonical atom schema.
    add_missing_atoms : bool
        Whether to add missing heavy atoms after canonicalization.
    demote_lone_altlocs : bool
        Whether to move atoms represented under only one altloc to the blank altloc.
    set_blank_altlocs_full_occupancy : bool
        Whether to set blank-altloc atom occupancies to 1.0.

    Returns
    -------
    ResidueSanitizationResult
        Sanitized atom records, component replacements, and counts.

    Raises
    ------
    ValueError
        If one residue position contains incompatible residue identities.
    """
    if not any(
        (
            canonicalize_residues,
            add_missing_atoms,
            demote_lone_altlocs,
            set_blank_altlocs_full_occupancy,
        )
    ):
        return ResidueSanitizationResult(atom_records, {})

    source_names = tuple(
        dict.fromkeys(atom_record["label_comp_id"] for atom_record in atom_records)
    )
    if (
        not canonicalize_residues
        and len(source_names) != 1
        and (add_missing_atoms or demote_lone_altlocs)
    ):
        raise ValueError(
            "Cannot add missing atoms or demote lone altlocs without canonicalizing "
            f"a residue position with multiple component identities: {source_names}"
        )

    output_atom_records = atom_records
    replacements: dict[str, str] = {}
    removed = 0
    canonicalized = 0
    output_residue_name: str | None = None
    template: AtomArray | None = None
    if canonicalize_residues or add_missing_atoms:
        targets = {closest_canonical_residue_name(name) for name in source_names}
        eligible_for_template = (
            atom_records[0]["label_seq_id"] not in BLANK_ALTLOC_IDS and None not in targets
        )
        if eligible_for_template:
            if len(targets) != 1:
                raise ValueError(f"Conflicting residue identities at one position: {source_names}")
            canonical_target = targets.pop()
            assert canonical_target is not None
            if canonicalize_residues:
                output_residue_name = canonical_target
                template = residue_template(canonical_target)
                output_atom_records, replacements, removed = canonicalize_residue(
                    atom_records,
                    canonical_target,
                    template,
                )
                canonicalized = int(any(name != canonical_target for name in source_names))
            else:
                output_residue_name = source_names[0]
                template = residue_template(output_residue_name)

    demoted = demote_lone_altloc_atoms(output_atom_records) if demote_lone_altlocs else 0
    occupancy_normalized = (
        set_blank_altloc_occupancy(output_atom_records) if set_blank_altlocs_full_occupancy else 0
    )

    added_atom_records: list[AtomRecord] = []
    skipped = 0
    if add_missing_atoms and output_residue_name is not None and template is not None:
        added_atom_records, skipped = add_missing_atoms_to_residue(
            output_atom_records,
            output_residue_name,
            template,
            first_id,
        )
    return ResidueSanitizationResult(
        atom_records=output_atom_records + added_atom_records,
        replacements=replacements,
        canonicalized=canonicalized,
        added=len(added_atom_records),
        removed=removed,
        skipped=skipped,
        demoted=demoted,
        occupancy_normalized=occupancy_normalized,
    )


def demote_lone_altloc_atoms(atom_records: list[AtomRecord]) -> int:
    """Move atoms represented under only one altloc to the blank altloc.

    Existing blank records take precedence: an alternate record with the same
    component and atom name is not demoted because doing so would create a duplicate.
    This function mutates ``atom_records``.

    Parameters
    ----------
    atom_records : list[AtomRecord]
        Atom-site records for one residue position and model.

    Returns
    -------
    int
        Number of records moved to the blank altloc.
    """
    blank_value = next(
        (
            atom_record["label_alt_id"]
            for atom_record in atom_records
            if atom_record["label_alt_id"] in BLANK_ALTLOC_IDS
        ),
        ".",
    )
    blank_atom_keys = {
        (atom_record["label_comp_id"], atom_record["label_atom_id"])
        for atom_record in atom_records
        if atom_record["label_alt_id"] in BLANK_ALTLOC_IDS
    }
    altlocs_by_atom: dict[tuple[str, str], set[str]] = {}
    for atom_record in atom_records:
        altloc = atom_record["label_alt_id"]
        if altloc in BLANK_ALTLOC_IDS:
            continue
        atom_key = (atom_record["label_comp_id"], atom_record["label_atom_id"])
        altlocs_by_atom.setdefault(atom_key, set()).add(altloc)
    lone_atom_keys = {
        atom_key
        for atom_key, altlocs in altlocs_by_atom.items()
        if len(altlocs) == 1 and atom_key not in blank_atom_keys
    }

    demoted = 0
    for atom_record in atom_records:
        atom_key = (atom_record["label_comp_id"], atom_record["label_atom_id"])
        if atom_key in lone_atom_keys:
            atom_record["label_alt_id"] = blank_value
            demoted += 1
    return demoted


def set_blank_altloc_occupancy(atom_records: list[AtomRecord]) -> int:
    """Set blank-altloc atom occupancies to 1.0.

    This function mutates ``atom_records``.

    Parameters
    ----------
    atom_records : list[AtomRecord]
        Atom-site records for one residue position and model.

    Returns
    -------
    int
        Number of occupancies changed.
    """
    normalized = 0
    for atom_record in atom_records:
        if atom_record["label_alt_id"] not in BLANK_ALTLOC_IDS:
            continue
        try:
            occupancy = float(atom_record["occupancy"])
        except ValueError:
            occupancy = None
        if occupancy != 1.0:
            atom_record["occupancy"] = "1.0"
            normalized += 1
    return normalized


def canonicalize_residue(
    atom_records: list[AtomRecord],
    target: str,
    template: AtomArray,
) -> tuple[list[AtomRecord], dict[str, str], int]:
    """Convert one residue to a canonical atom schema without moving atoms.

    This function updates retained records in place.

    Parameters
    ----------
    atom_records : list[AtomRecord]
        Atom-site records for one residue position.
    target : str
        Canonical residue name.
    template : biotite.structure.AtomArray
        Ideal canonical residue template.

    Returns
    -------
    tuple[list[AtomRecord], dict[str, str], int]
        Canonicalized atom records, component replacements, and removed-atom count.
    """
    elements_by_name = dict(
        zip(template.atom_name.tolist(), template.element.tolist(), strict=True)
    )
    canonical_atom_records: list[AtomRecord] = []
    replacements: dict[str, str] = {}
    removed = 0
    for atom_record in atom_records:
        source = atom_record["label_comp_id"]
        atom = atom_record["label_atom_id"]
        atom = ATOM_RENAMES.get((source, target, atom), atom)
        if atom not in elements_by_name:
            # Atoms outside the canonical definition are dropped only when they
            # belong to a modification being renamed away. An already-canonical
            # residue keeps them: N-terminal H1/H3 and the deuterium of a neutron
            # structure are absent from the template but are not modifications.
            if source != target:
                removed += 1
                continue
            canonical_atom_records.append(atom_record)
            continue
        replacements[source] = target
        atom_record["group_PDB"] = "ATOM"
        atom_record["label_comp_id"] = target
        atom_record["label_atom_id"] = atom
        atom_record["type_symbol"] = elements_by_name[atom]
        if "auth_comp_id" in atom_record:
            atom_record["auth_comp_id"] = target
        if "auth_atom_id" in atom_record:
            atom_record["auth_atom_id"] = atom
        canonical_atom_records.append(atom_record)
    return canonical_atom_records, replacements, removed


def next_atom_id(residue_records: ResidueRecords) -> int:
    """Find the first available numeric atom-site identifier.

    Parameters
    ----------
    residue_records : ResidueRecords
        Atom records keyed by residue position and model.

    Returns
    -------
    int
        One greater than the largest existing identifier, or one when absent.
    """
    return (
        max(
            (
                int(atom_record["id"])
                for atom_records in residue_records.values()
                for atom_record in atom_records
                if "id" in atom_record
            ),
            default=0,
        )
        + 1
    )


def add_missing_atoms_to_residue(
    atom_records: list[AtomRecord],
    residue_name: str,
    template: AtomArray,
    first_id: int,
) -> tuple[list[AtomRecord], int]:
    """Add missing heavy atoms to each conformer of one residue.

    The template-superposition and coordinate-transfer algorithm is adapted
    from PDBFixer's ``PDBFixer._addAtomsToTopology()`` implementation.

    Parameters
    ----------
    atom_records : list[AtomRecord]
        Canonicalized atom records for one residue.
    residue_name : str
        Canonical residue name.
    template : biotite.structure.AtomArray
        Ideal canonical residue template.
    first_id : int
        First identifier available for a new atom.

    Returns
    -------
    tuple[list[AtomRecord], int]
        New atom records and the number of conformers that could not be oriented.
    """
    shared_atom_records = [
        atom_record
        for atom_record in atom_records
        if atom_record["label_alt_id"] in BLANK_ALTLOC_IDS
    ]
    altlocs = tuple(
        dict.fromkeys(
            atom_record["label_alt_id"]
            for atom_record in atom_records
            if atom_record["label_alt_id"] not in BLANK_ALTLOC_IDS
        )
    ) or (None,)
    heavy_template = template[(template.element != "H") & (template.atom_name != "OXT")]
    added_atom_records: list[AtomRecord] = []
    skipped = 0

    for altloc in altlocs:
        conformer_atom_records = shared_atom_records + [
            atom_record for atom_record in atom_records if atom_record["label_alt_id"] == altloc
        ]
        atom_name_to_record = {
            atom_record["label_atom_id"]: atom_record for atom_record in conformer_atom_records
        }
        present_mask = np.isin(heavy_template.atom_name, tuple(atom_name_to_record))
        if np.all(present_mask):
            continue
        common_template = heavy_template[present_mask]
        fixed_template = common_template.copy()
        fixed_template.coord = np.asarray(
            [
                [
                    float(atom_name_to_record[name][axis])
                    for axis in ("Cartn_x", "Cartn_y", "Cartn_z")
                ]
                for name in common_template.atom_name
            ],
            dtype=float,
        )  # shape: (common_atoms, 3)
        fixed_coordinate_rank = np.linalg.matrix_rank(
            fixed_template.coord - fixed_template.coord.mean(axis=0)
        )
        if len(common_template) < 3 or fixed_coordinate_rank < 2:
            skipped += 1
            continue

        _, transform = superimpose(fixed_template, common_template)
        fitted_template = heavy_template.copy()
        fitted_template.coord = transform.apply(heavy_template.coord)
        missing_template = fitted_template[~present_mask]
        seed_atom_record = next(
            (atom_record for atom_record in atom_records if atom_record["label_alt_id"] == altloc),
            atom_records[0],
        )
        altloc_value = (
            seed_atom_record["label_alt_id"]
            if altloc is not None
            else shared_atom_records[0]["label_alt_id"]
        )
        for name, element, coordinate in zip(
            missing_template.atom_name,
            missing_template.element,
            missing_template.coord,
            strict=True,
        ):
            new_atom_record = seed_atom_record.copy()
            new_atom_record["group_PDB"] = "ATOM"
            new_atom_record["type_symbol"] = element
            new_atom_record["label_atom_id"] = name
            new_atom_record["label_alt_id"] = altloc_value
            new_atom_record["label_comp_id"] = residue_name
            if "auth_atom_id" in new_atom_record:
                new_atom_record["auth_atom_id"] = name
            if "auth_comp_id" in new_atom_record:
                new_atom_record["auth_comp_id"] = residue_name
            for axis, value in zip(("Cartn_x", "Cartn_y", "Cartn_z"), coordinate, strict=True):
                new_atom_record[axis] = f"{value:.3f}"
            if "id" in new_atom_record:
                new_atom_record["id"] = str(first_id + len(added_atom_records))
            added_atom_records.append(new_atom_record)
    return added_atom_records, skipped


@cache
def residue_template(residue_name: str) -> AtomArray:
    """Load a cached canonical residue template from Biotite.

    Parameters
    ----------
    residue_name : str
        Canonical residue name.

    Returns
    -------
    biotite.structure.AtomArray
        Ideal residue template. Callers must treat the cached array as read-only.
    """
    return get_residue_template(residue_name)


def update_connections(cif_block: CIFBlock, replacements: dict[str, str]) -> None:
    """Update component and renamed atom identifiers in ``_struct_conn``.

    This function mutates ``cif_block``.

    Parameters
    ----------
    cif_block : biotite.structure.io.pdbx.cif.CIFBlock
        CIF block modified in place.
    replacements : dict[str, str]
        Source-to-canonical component names.
    """
    if "struct_conn" not in cif_block or not replacements:
        return
    connections = cif_block["struct_conn"]
    for partner in ("ptnr1", "ptnr2", "pdbx_ptnr3"):
        component_column = f"{partner}_label_comp_id"
        if component_column not in connections:
            continue
        sources = connections[component_column].as_array(str)
        map_category_values(
            connections,
            (component_column, f"{partner}_auth_comp_id"),
            replacements,
        )
        atom_column = f"{partner}_label_atom_id"
        if atom_column not in connections:
            continue
        targets = connections[component_column].as_array(str)
        atoms = connections[atom_column].as_array(str)
        connections[atom_column] = np.array(
            [
                ATOM_RENAMES.get((source, target, atom), atom)
                for source, target, atom in zip(sources, targets, atoms, strict=True)
            ]
        )


def update_polymer_sequence(cif_block: CIFBlock, replacements: dict[str, str]) -> None:
    """Canonicalize component names in polymer sequence metadata.

    This function mutates ``cif_block``. It does not add or remove sequence rows,
    including rows for residues that are wholly absent from ``_atom_site``.

    Parameters
    ----------
    cif_block : biotite.structure.io.pdbx.cif.CIFBlock
        CIF block modified in place.
    replacements : dict[str, str]
        Source-to-canonical component names found in existing atom records.
    """
    category_to_columns = {
        "entity_poly_seq": ("mon_id",),
        "pdbx_poly_seq_scheme": ("mon_id", "pdb_mon_id", "auth_mon_id"),
    }
    for category_name, component_columns in category_to_columns.items():
        if category_name not in cif_block:
            continue
        map_category_values(cif_block[category_name], component_columns, replacements)
