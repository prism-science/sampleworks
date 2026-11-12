"""Build FoldDisco queries that distinguish two altloc conformers of a loop.

Given a PDB file, a chain and a set of residues modeled with two alternate
conformers (altlocs), this script

    1. writes one PDB file per altloc in which only that conformer of the chosen
       residues is kept (altloc label cleared, occupancy set to 1.00); every other
       line of the file is copied unchanged,
    2. picks up to ``--max-residues`` of the chosen residues whose geometry changes
       most between the two conformers, and the same number of anchor residues
       (residues with no altlocs, outside the chosen set) whose distances to them
       change most,
    3. writes a FoldDisco batch query file with one line per conformer, both using
       the same motif, ready for ``folddisco-altloc search``.

Residues are scored with the geometry FoldDisco hashes: Ca-Ca and Cb-Cb distances
between motif residues, with a virtual Cb for glycine. FoldDisco ignores altloc
labels and keeps the last conformer it reads for each residue, so anchors are
restricted to residues without altlocs.

Command line: ``folddisco-altloc motif 6B8X_single_001.pdb --chain A --residues 178-193``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import gemmi
import numpy as np
from jaxtyping import Float


ATOM_RECORDS = ("ATOM  ", "HETATM", "ANISOU")


class MotifError(ValueError):
    """The residue set cannot be split into two conformers or yields no motif."""


class ResidueId(NamedTuple):
    """A residue identified by chain and sequence number (no insertion code)."""

    chain: str
    number: int

    @property
    def label(self) -> str:
        """FoldDisco motif syntax for this residue, e.g. ``A181``."""
        return f"{self.chain}{self.number}"


@dataclass(frozen=True)
class AltlocSegment:
    """A run of consecutive residues in one chain that all carry altlocs.

    Attributes
    ----------
    chain : str
        Chain ID.
    start, end : int
        First and last residue number (inclusive).
    altlocs : tuple of str
        Altloc labels used anywhere in the segment, sorted.
    incomplete : tuple of ResidueId
        Residues lacking at least one of ``altlocs``.
    """

    chain: str
    start: int
    end: int
    altlocs: tuple[str, ...]
    incomplete: tuple[ResidueId, ...]

    @property
    def residues(self) -> str:
        """Residue range in ``--residues`` syntax, e.g. ``178-193``."""
        return f"{self.start}-{self.end}" if self.end > self.start else str(self.start)


@dataclass(frozen=True)
class MotifChoice:
    """Motif residues chosen to separate two conformers.

    Attributes
    ----------
    loop : list of ResidueId
        Chosen residues from the user's set, in sequence order.
    anchors : list of ResidueId
        Chosen residues without altlocs, in chain and sequence order.
    loop_scores : dict of ResidueId to float
        Summed distance change (Å) of each chosen loop residue to nearby anchors.
    anchor_scores : dict of ResidueId to float
        Summed distance change (Å) of each chosen anchor to the chosen loop residues.
    """

    loop: list[ResidueId]
    anchors: list[ResidueId]
    loop_scores: dict[ResidueId, float]
    anchor_scores: dict[ResidueId, float]

    @property
    def motif(self) -> str:
        """Comma-separated FoldDisco motif, loop residues then anchors."""
        return ",".join(residue.label for residue in [*self.loop, *self.anchors])


def parse_residue_spec(spec: str) -> list[int]:
    """Expand a residue specification such as ``178-193,200`` into residue numbers.

    Parameters
    ----------
    spec : str
        Comma-separated residue numbers and inclusive ranges.

    Returns
    -------
    list of int
        Sorted, unique residue numbers.
    """
    numbers = set()
    for part in spec.split(","):
        start, _, end = part.strip().partition("-")
        first, last = int(start), int(end or start)
        if last < first:
            raise ValueError(f"range {part!r} ends before it starts")
        numbers.update(range(first, last + 1))
    return sorted(numbers)


def _is_selected_atom(line: str, residues: set[ResidueId]) -> bool:
    """Whether a PDB line is an atom record of one of ``residues``."""
    return (
        line.startswith(ATOM_RECORDS)
        and line[26] == " "
        and ResidueId(line[21], int(line[22:26])) in residues
    )


def find_altlocs(lines: list[str], residues: set[ResidueId]) -> dict[ResidueId, set[str]]:
    """Find the altloc labels used by each of ``residues``.

    Parameters
    ----------
    lines : list of str
        Lines of a PDB file.
    residues : set of ResidueId
        Residues to inspect.

    Returns
    -------
    dict of ResidueId to set of str
        Non-blank altloc labels per residue; residues without altlocs are absent.
    """
    altlocs: dict[ResidueId, set[str]] = {}
    for line in lines:
        if _is_selected_atom(line, residues) and line[16] != " ":
            altlocs.setdefault(ResidueId(line[21], int(line[22:26])), set()).add(line[16])
    return altlocs


def residues_missing_altloc(
    altlocs: dict[ResidueId, set[str]], chosen: list[str]
) -> list[ResidueId]:
    """Residues that have altlocs but lack at least one of ``chosen``.

    Parameters
    ----------
    altlocs : dict of ResidueId to set of str
        Output of :func:`find_altlocs`.
    chosen : list of str
        The two altlocs being compared.

    Returns
    -------
    list of ResidueId
        Residues that :func:`select_altloc` would delete from one conformer file.
    """
    return sorted(residue for residue, labels in altlocs.items() if not set(chosen) <= labels)


def select_altloc(lines: list[str], residues: set[ResidueId], altloc: str) -> list[str]:
    """Keep a single altloc for ``residues``, leaving every other line unchanged.

    Parameters
    ----------
    lines : list of str
        Lines of a PDB file, with line endings.
    residues : set of ResidueId
        Residues whose other conformers are removed.
    altloc : str
        Altloc label to keep.

    Returns
    -------
    list of str
        New list of lines. Kept atoms of ``residues`` have a blank altloc and
        occupancy 1.00.
    """
    selected = []
    for line in lines:
        if not _is_selected_atom(line, residues) or line[16] == " ":
            selected.append(line)
        elif line[16] == altloc:
            occupancy = "  1.00" if line.startswith(("ATOM  ", "HETATM")) else line[54:60]
            selected.append(f"{line[:16]} {line[17:54]}{occupancy}{line[60:]}")
    return selected


def _virtual_cb(
    n: Float[np.ndarray, "3"], ca: Float[np.ndarray, "3"], c: Float[np.ndarray, "3"]
) -> Float[np.ndarray, "3"]:
    """Ideal Cb position from backbone N, Ca and C (trRosetta coefficients)."""
    b = ca - n
    bc = c - ca
    return -0.58273431 * np.cross(b, bc) + 0.56802827 * b - 0.54067466 * bc + ca


def backbone_points(
    structure: gemmi.Structure,
) -> dict[ResidueId, Float[np.ndarray, "2 3"]]:
    """Ca and Cb coordinates of every altloc-free amino acid in the first model.

    Parameters
    ----------
    structure : gemmi.Structure
        Parsed structure.

    Returns
    -------
    dict of ResidueId to ndarray
        ``[Ca, Cb]`` per residue; glycine and residues missing Cb get a virtual Cb.
        Residues with altlocs, insertion codes, or missing N/Ca/C are left out.
    """
    points = {}
    for chain in structure[0]:
        for residue in chain:
            if (
                not gemmi.find_tabulated_residue(residue.name)
                or not gemmi.find_tabulated_residue(residue.name).is_amino_acid()
                or residue.seqid.icode != " "
                or any(atom.altloc != "\0" for atom in residue)
            ):
                continue
            atoms = {atom.name: np.array(atom.pos.tolist()) for atom in residue}
            if not {"N", "CA", "C"} <= atoms.keys():
                continue
            cb = atoms.get("CB")
            if cb is None:
                cb = _virtual_cb(atoms["N"], atoms["CA"], atoms["C"])
            points[ResidueId(chain.name, residue.seqid.num)] = np.stack([atoms["CA"], cb])
    return points


def anchor_candidates(
    points_a: dict[ResidueId, Float[np.ndarray, "2 3"]],
    points_b: dict[ResidueId, Float[np.ndarray, "2 3"]],
    loop: list[ResidueId],
    min_separation: int,
) -> list[ResidueId]:
    """Residues eligible as anchors: altloc-free, outside the set, away in sequence.

    Parameters
    ----------
    points_a, points_b : dict of ResidueId to ndarray
        :func:`backbone_points` of the two conformer files, which already exclude
        residues with altlocs.
    loop : list of ResidueId
        The user's full residue set.
    min_separation : int
        Minimum sequence distance from any residue of ``loop`` in the same chain.

    Returns
    -------
    list of ResidueId
        Candidate anchors in chain and sequence order.
    """
    loop_set = set(loop)
    return sorted(
        residue
        for residue in points_a.keys() & points_b.keys()
        if residue not in loop_set
        and all(
            residue.chain != loop_residue.chain
            or abs(residue.number - loop_residue.number) >= min_separation
            for loop_residue in loop
        )
    )


def distance_changes(
    loop_a: Float[np.ndarray, "loop 2 3"],
    loop_b: Float[np.ndarray, "loop 2 3"],
    anchors: Float[np.ndarray, "anchors 2 3"],
    cutoff: float,
) -> Float[np.ndarray, "loop anchors"]:
    """Change in Ca-Ca plus Cb-Cb distance between conformers, per loop-anchor pair.

    Parameters
    ----------
    loop_a, loop_b : ndarray
        Ca and Cb of each loop residue in conformer A and conformer B.
    anchors : ndarray
        Ca and Cb of each anchor candidate (identical in both conformers).
    cutoff : float
        Pairs whose Ca-Ca and Cb-Cb distances both exceed this (Å) in both
        conformers score zero, so motifs stay compact.

    Returns
    -------
    ndarray
        ``|dCa_A - dCa_B| + |dCb_A - dCb_B|`` in Å, zero for pairs beyond ``cutoff``.
    """
    # [loop, anchors, 2]: Ca-Ca and Cb-Cb distances
    dist_a = np.linalg.norm(loop_a[:, None] - anchors[None], axis=-1)
    dist_b = np.linalg.norm(loop_b[:, None] - anchors[None], axis=-1)
    in_range = (np.minimum(dist_a, dist_b) <= cutoff).any(axis=-1)
    return np.abs(dist_a - dist_b).sum(axis=-1) * in_range


def choose_motif(
    points_a: dict[ResidueId, Float[np.ndarray, "2 3"]],
    points_b: dict[ResidueId, Float[np.ndarray, "2 3"]],
    loop: list[ResidueId],
    max_residues: int,
    cutoff: float,
    min_separation: int,
) -> MotifChoice:
    """Pick loop residues and anchors whose relative geometry differs most.

    Loop residues are ranked by their summed distance change to all anchor
    candidates within ``cutoff``; anchors are then ranked by their summed
    distance change to the chosen loop residues.

    Parameters
    ----------
    points_a, points_b : dict of ResidueId to ndarray
        :func:`backbone_points` of the two single-conformer structures.
    loop : list of ResidueId
        The user's residue set.
    max_residues : int
        Maximum number of loop residues, and of anchors, in the motif.
    cutoff : float
        Distance cutoff (Å) passed to :func:`distance_changes`.
    min_separation : int
        Passed to :func:`anchor_candidates`.

    Returns
    -------
    MotifChoice
        The chosen residues and their scores.
    """
    candidates = anchor_candidates(points_a, points_b, loop, min_separation)
    loop = [residue for residue in loop if residue in points_a and residue in points_b]
    changes = distance_changes(
        np.stack([points_a[r] for r in loop]),
        np.stack([points_b[r] for r in loop]),
        np.stack([points_a[r] for r in candidates]),
        cutoff,
    )

    loop_totals = changes.sum(axis=1)
    loop_rows = [i for i in np.argsort(-loop_totals)[:max_residues] if loop_totals[i] > 0]
    anchor_totals = changes[loop_rows].sum(axis=0)
    anchor_cols = [j for j in np.argsort(-anchor_totals)[: len(loop_rows)] if anchor_totals[j] > 0]

    return MotifChoice(
        loop=sorted(loop[i] for i in loop_rows),
        anchors=sorted(candidates[j] for j in anchor_cols),
        loop_scores={loop[i]: float(loop_totals[i]) for i in loop_rows},
        anchor_scores={candidates[j]: float(anchor_totals[j]) for j in anchor_cols},
    )


def output_stem(pdb: Path, chain: str, numbers: list[int]) -> str:
    """File-name stem shared by all outputs, e.g. ``6B8X_single_001_A178-193``."""
    return f"{pdb.stem}_{chain}{numbers[0]}-{numbers[-1]}"


def choose_altlocs(altlocs: dict[ResidueId, set[str]], requested: list[str] | None) -> list[str]:
    """Pick the two altlocs to compare.

    Parameters
    ----------
    altlocs : dict of ResidueId to set of str
        Output of :func:`find_altlocs` for the user's residue set.
    requested : list of str or None
        Two altlocs chosen by the user; None to use the only two present.

    Returns
    -------
    list of str
        The two altloc labels, in the order given (or sorted).

    Raises
    ------
    MotifError
        If the residues do not have exactly two usable altlocs, or some residue
        lacks one of them and would be deleted from a conformer file.
    """
    present = sorted(set().union(*altlocs.values()))
    if requested is None and len(present) != 2:
        raise MotifError(
            f"need exactly two altlocs, found {present or 'none'} on these residues"
            + ("; choose two of them" if len(present) > 2 else "")
        )
    chosen = list(requested or present)
    if len(set(chosen)) != 2 or not set(chosen) <= set(present):
        raise MotifError(f"altlocs must be two different labels from {present}")

    missing = residues_missing_altloc(altlocs, chosen)
    if missing:
        raise MotifError(
            f"these residues lack altloc {' or '.join(chosen)} and would be deleted from "
            f"one conformer file: {', '.join(r.label for r in missing)}. "
            "Narrow the residue range to exclude them."
        )
    return chosen


def find_altloc_segments(lines: list[str]) -> list[AltlocSegment]:
    """List runs of consecutive amino acids that carry altlocs.

    Parameters
    ----------
    lines : list of str
        Lines of a PDB file.

    Returns
    -------
    list of AltlocSegment
        Segments in file order; a residue without altlocs or a numbering gap ends
        a segment. Non-amino-acid residues (waters, ions, ligands) and residues with
        insertion codes are ignored.
    """
    labels: dict[ResidueId, set[str]] = {}
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")) or line[26] != " ":
            continue
        info = gemmi.find_tabulated_residue(line[17:20].strip())
        if info is None or not info.is_amino_acid():
            continue
        residue = ResidueId(line[21], int(line[22:26]))
        labels.setdefault(residue, set())
        if line[16] != " ":
            labels[residue].add(line[16])

    runs: list[list[ResidueId]] = []
    previous: ResidueId | None = None
    for residue, residue_labels in labels.items():
        if not residue_labels:
            previous = None
            continue
        contiguous = (
            previous is not None
            and residue.chain == previous.chain
            and residue.number == previous.number + 1
        )
        if contiguous:
            runs[-1].append(residue)
        else:
            runs.append([residue])
        previous = residue

    segments = []
    for run in runs:
        union = set().union(*(labels[r] for r in run))
        incomplete = tuple(r for r in run if labels[r] != union)
        segments.append(
            AltlocSegment(
                run[0].chain, run[0].number, run[-1].number, tuple(sorted(union)), incomplete
            )
        )
    return segments
