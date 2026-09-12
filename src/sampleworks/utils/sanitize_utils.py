"""Sanitize residue records, altlocs, occupancies, and incomplete conformers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from string import ascii_uppercase

import numpy as np
from atomworks.io.utils.io_utils import get_structure
from biotite.structure import AtomArray, filter_polymer, get_residue_starts, superimpose
from biotite.structure.info import residue as get_residue_template
from biotite.structure.io.pdbx.cif import CIFBlock, CIFFile

from sampleworks.utils.atom_array_utils import BLANK_ALTLOC_IDS, closest_canonical_residue_name
from sampleworks.utils.cif_utils import AtomRecord, map_category_values, ResidueRecords


# A modification's substituted atom, and the canonical atom that stands in its place.
# The atom keeps its measured position, so it also keeps the heavier element's bond
# length: MSE's Se-C is ~1.95 A against methionine's 1.79 A S-C ideal. Regularizing
# that is left to geometry minimization, which is told to treat a substituted atom as
# movable. Placing it from the ideal template instead is worse -- a rigid whole-residue
# fit cannot reproduce the observed side-chain torsions, and SD is bonded on both sides.
ATOM_RENAMES = {("MSE", "MET", "SE"): "SD"}
# One residue position, as (label_asym_id, label_seq_id, insertion code).
PolymerResidueKey = tuple[str, str, str]


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
    dropped : int
        Number of atoms deleted with a zero-occupancy conformer.
    demoted : int
        Number of lone-altloc atoms moved to the blank altloc.
    relabelled : int
        Number of residue positions whose altloc identifiers were relabelled.
    occupancy_normalized : int
        Number of blank-altloc atom occupancies changed to 1.0.
    """

    atom_records: list[AtomRecord]
    replacements: dict[str, str]
    canonicalized: int = 0
    added: int = 0
    removed: int = 0
    skipped: int = 0
    dropped: int = 0
    demoted: int = 0
    relabelled: int = 0
    occupancy_normalized: int = 0


def polymer_residue_keys(cif_file: CIFFile) -> set[PolymerResidueKey]:
    """Find the residue positions that are bonded into a peptide polymer.

    Component identity cannot answer this. The CCD classifies ligands such as
    ``UMA`` as peptide-linking, so any name-, link-type-, or template-based test
    admits them and then canonicalization strips them to a bare amino acid.
    Connectivity separates a modified residue in a chain from a free ligand, and
    it still admits an observed residue that is missing a backbone atom.

    Parameters
    ----------
    cif_file : biotite.structure.io.pdbx.cif.CIFFile
        Structure to inspect. Passing the caller's already-read file avoids
        reparsing it.

    Returns
    -------
    set[PolymerResidueKey]
        One key per polymer residue, with blank insertion codes normalized to the
        empty string. Atomworks reads label rather than author fields, so these
        are ``_atom_site`` label values and compare directly against the keys of
        :func:`sampleworks.utils.cif_utils.group_atom_site_records_by_residue`.
    """
    atom_array = get_structure(cif_file, altloc="all", model=1)
    polymer_mask = filter_polymer(atom_array, pol_type="peptide")
    has_ins_code = "ins_code" in atom_array.get_annotation_categories()
    residue_starts = [*get_residue_starts(atom_array), len(atom_array)]
    keys: set[PolymerResidueKey] = set()
    for start, stop in zip(residue_starts[:-1], residue_starts[1:], strict=True):
        if not polymer_mask[start:stop].any():
            continue
        insertion_code = str(atom_array.ins_code[start]).strip() if has_ins_code else ""
        keys.add(
            (str(atom_array.chain_id[start]), str(int(atom_array.res_id[start])), insertion_code)
        )
    return keys


def sanitize_residues(
    residue_records: ResidueRecords,
    *,
    polymer_residues: set[PolymerResidueKey],
    canonicalize_residues: bool,
    add_missing_atoms: bool,
    drop_zero_occupancy_conformers: bool,
    demote_lone_altlocs: bool,
    relabel_altloc_ids: bool,
    set_blank_altlocs_full_occupancy: bool,
) -> tuple[list[AtomRecord], dict[str, str], dict[str, int]]:
    """Sanitize all residue positions and aggregate their summaries.

    Parameters
    ----------
    residue_records : ResidueRecords
        Atom records keyed by residue position and model.
    polymer_residues : set[PolymerResidueKey]
        Residue positions eligible for template work, from
        :func:`polymer_residue_keys`.
    canonicalize_residues : bool
        Whether to convert residues to canonical atom schemas.
    add_missing_atoms : bool
        Whether to add missing heavy atoms after canonicalization.
    drop_zero_occupancy_conformers : bool
        Whether to delete alternate conformers modelled entirely at occupancy 0,
        when another conformer at the position still carries weight.
    demote_lone_altlocs : bool
        Whether to move atoms represented under only one altloc to the blank altloc.
    relabel_altloc_ids : bool
        Whether to relabel each residue's conformers consecutively from ``A``.
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
        "dropped": 0,
        "demoted": 0,
        "relabelled": 0,
        "occupancy_normalized": 0,
    }

    for residue_key, atom_records in residue_records.items():
        _, label_asym_id, label_seq_id, insertion_code, _ = residue_key
        is_polymer = (
            label_asym_id,
            label_seq_id,
            "" if insertion_code in BLANK_ALTLOC_IDS else insertion_code,
        ) in polymer_residues
        result = sanitize_residue(
            atom_records,
            first_id=next_id,
            is_polymer=is_polymer,
            canonicalize_residues=canonicalize_residues,
            add_missing_atoms=add_missing_atoms,
            drop_zero_occupancy_conformers=drop_zero_occupancy_conformers,
            demote_lone_altlocs=demote_lone_altlocs,
            relabel_altloc_ids=relabel_altloc_ids,
            set_blank_altlocs_full_occupancy=set_blank_altlocs_full_occupancy,
        )
        output_atom_records.extend(result.atom_records)
        replacements.update(result.replacements)
        counts["canonicalized"] += result.canonicalized
        counts["added"] += result.added
        counts["removed"] += result.removed
        counts["skipped"] += result.skipped
        counts["dropped"] += result.dropped
        counts["demoted"] += result.demoted
        counts["relabelled"] += result.relabelled
        counts["occupancy_normalized"] += result.occupancy_normalized
        next_id += result.added

    return output_atom_records, replacements, counts


def sanitize_residue(
    atom_records: list[AtomRecord],
    *,
    first_id: int,
    is_polymer: bool,
    canonicalize_residues: bool,
    add_missing_atoms: bool,
    drop_zero_occupancy_conformers: bool,
    demote_lone_altlocs: bool,
    relabel_altloc_ids: bool,
    set_blank_altlocs_full_occupancy: bool,
) -> ResidueSanitizationResult:
    """Canonicalize one residue and optionally add its missing heavy atoms.

    Parameters
    ----------
    atom_records : list[AtomRecord]
        Atom-site records for one residue position and model.
    first_id : int
        First identifier available for a new atom.
    is_polymer : bool
        Whether this residue is bonded into a peptide polymer. Only polymer
        residues are canonicalized or completed from a template.
    canonicalize_residues : bool
        Whether to convert this residue to its canonical atom schema.
    add_missing_atoms : bool
        Whether to add missing heavy atoms after canonicalization.
    drop_zero_occupancy_conformers : bool
        Whether to delete alternate conformers modelled entirely at occupancy 0,
        when another conformer at the position still carries weight.
    demote_lone_altlocs : bool
        Whether to move atoms represented under only one altloc to the blank altloc.
    relabel_altloc_ids : bool
        Whether to relabel this residue's conformers consecutively from ``A``.
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
            drop_zero_occupancy_conformers,
            demote_lone_altlocs,
            relabel_altloc_ids,
            set_blank_altlocs_full_occupancy,
        )
    ):
        return ResidueSanitizationResult(atom_records, {})

    # Before everything else: a conformer that is about to be deleted should not be
    # renamed, completed from a template, or counted as a second conformer by the
    # lone-altloc test. Dropping it can also reduce the position's component count, which
    # only widens what the guard below accepts.
    dropped = 0
    if drop_zero_occupancy_conformers:
        atom_records, dropped = drop_zero_occupancy_conformer_records(atom_records)

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
        eligible_for_template = is_polymer and None not in targets
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
    relabelled = relabel_residue_altlocs(output_atom_records) if relabel_altloc_ids else 0
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
        dropped=dropped,
        demoted=demoted,
        relabelled=relabelled,
        occupancy_normalized=occupancy_normalized,
    )


def drop_zero_occupancy_conformer_records(
    atom_records: list[AtomRecord],
) -> tuple[list[AtomRecord], int]:
    """Delete alternate conformers modelled entirely at occupancy 0.

    A depositor writes a conformer at zero occupancy to say it is not there. It still
    carries coordinates, so every downstream consumer that ignores occupancy still sees
    it: SFcalculator derives its solvent fraction from atom positions with no occupancy
    term, and a synthetic-target generator that rewrites occupancies resurrects the
    conformer at full weight. Measured on 2A26, 16 such atoms in a 394-atom target moved
    R(|Fprotein|) to 0.162 against the same target without them.

    Deletion is safe exactly when another conformer at the position still carries weight,
    which is the condition required here. Across the three carriers in the initial-40 set
    (3HVV, 5RBR, 5SOP: 30 positions, 350 atoms) the surviving conformers summed to 1.0 in
    every case, so nothing has to be redistributed.

    Two shapes are deliberately left alone:

    - A *lone* conformer at zero occupancy, and a position where *every* conformer is at
      zero. Nothing else models those positions, so deleting them would lose the only
      coordinates available and :func:`add_missing_atoms_to_residue` would refit them from
      an ideal template, which is worse. :func:`demote_lone_altloc_atoms` moves the lone
      case to the blank altloc and :func:`set_blank_altloc_occupancy` gives it 1.0 --
      the same treatment a blank-altloc zero gets, for the same reason.
    - A *partially* zeroed conformer, where some of its atoms are at zero and the rest
      carry the conformer's real occupancy. Those are unmodelled side-chain tips inside a
      conformer that does exist, not a phantom, and deleting the conformer would discard
      real geometry. None occurred in the initial-40 set (0 of 30 positions).

    Running before :func:`canonicalize_residue` and :func:`add_missing_atoms_to_residue`
    keeps them from renaming and completing a conformer that is about to be deleted, and
    can lower the position's component count, which only widens what those two accept.

    Unlike its neighbours this returns a new list rather than mutating in place, since it
    removes records rather than editing them.

    Parameters
    ----------
    atom_records : list[AtomRecord]
        Atom-site records for one residue position and model.

    Returns
    -------
    tuple[list[AtomRecord], int]
        The surviving records, and the number deleted.
    """
    conformers: dict[str, list[AtomRecord]] = {}
    for atom_record in atom_records:
        altloc = atom_record["label_alt_id"]
        if altloc not in BLANK_ALTLOC_IDS:
            conformers.setdefault(altloc, []).append(atom_record)
    # A lone conformer has no sibling to inherit the position's weight, so it is kept.
    if len(conformers) < 2:
        return atom_records, 0

    def all_zero(records: list[AtomRecord]) -> bool:
        """Whether every record in one conformer parses to occupancy 0."""
        for record in records:
            try:
                if float(record["occupancy"]) != 0.0:
                    return False
            except ValueError:  # unparseable occupancy: assume the atom is real
                return False
        return True

    zeroed = {altloc for altloc, records in conformers.items() if all_zero(records)}
    # Every conformer at zero leaves no sibling carrying weight; keep them all.
    if not zeroed or len(zeroed) == len(conformers):
        return atom_records, 0

    surviving = [
        atom_record for atom_record in atom_records if atom_record["label_alt_id"] not in zeroed
    ]
    return surviving, len(atom_records) - len(surviving)


def demote_lone_altloc_atoms(atom_records: list[AtomRecord]) -> int:
    """Move a residue modelled under a single altloc into the blank altloc.

    A residue position occupied by one conformer offers no alternative to choose
    between, so its altloc label carries no information and every atom is shared.
    The residue may be partial; completeness is not what makes it unambiguous.

    The test is per residue and never per atom. An atom name appearing under only
    one of several conformers is not shared: it is a side chain modelled to its tip
    in one conformer and truncated in another. Demoting it would attach it to every
    conformer, and a spec-strict reader would then bond it into the others across
    whatever distance separates them. Those residues keep their altlocs, and
    ``add_missing_atoms`` completes each conformer separately instead.

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
    residue_altlocs = {
        atom_record["label_alt_id"]
        for atom_record in atom_records
        if atom_record["label_alt_id"] not in BLANK_ALTLOC_IDS
    }
    if len(residue_altlocs) != 1:
        return 0

    demoted = 0
    for atom_record in atom_records:
        if atom_record["label_alt_id"] in BLANK_ALTLOC_IDS:
            continue
        atom_key = (atom_record["label_comp_id"], atom_record["label_atom_id"])
        if atom_key in blank_atom_keys:
            continue
        atom_record["label_alt_id"] = blank_value
        demoted += 1
    return demoted


def relabel_residue_altlocs(atom_records: list[AtomRecord]) -> int:
    """Relabel one residue's conformers consecutively from ``A``.

    Existing non-blank identifiers are sorted before they are mapped to uppercase
    letters. This merges conformer networks that the deposition kept apart, which
    is appropriate for constructing a synthetic target but not a faithful ensemble.
    More than 26 conformers are left unchanged rather than truncating identifiers
    and colliding multiple conformers onto one label. This function mutates
    ``atom_records``.

    Parameters
    ----------
    atom_records : list[AtomRecord]
        Atom-site records for one residue position and model.

    Returns
    -------
    int
        One if the residue's altloc identifiers changed, otherwise zero.
    """
    altlocs = sorted(
        {
            atom_record["label_alt_id"]
            for atom_record in atom_records
            if atom_record["label_alt_id"] not in BLANK_ALTLOC_IDS
        }
    )
    if len(altlocs) > len(ascii_uppercase):
        return 0

    relabels = dict(zip(altlocs, ascii_uppercase[: len(altlocs)], strict=True))
    if all(source == target for source, target in relabels.items()):
        return 0

    for atom_record in atom_records:
        altloc = atom_record["label_alt_id"]
        if altloc in relabels:
            atom_record["label_alt_id"] = relabels[altloc]
    return 1


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
        # Checked before the coordinates are gathered: three shared atoms are the
        # minimum that can orient a template, and gathering none of them would
        # build a one-dimensional array that AtomArray rejects.
        if len(common_template) < 3:
            skipped += 1
            continue
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
        if fixed_coordinate_rank < 2:
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
