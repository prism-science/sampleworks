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
       the same motif, ready for ``scripts/folddisco.py``.

Residues are scored with the geometry FoldDisco hashes: Ca-Ca and Cb-Cb distances
between motif residues, with a virtual Cb for glycine. FoldDisco ignores altloc
labels and keeps the last conformer it reads for each residue, so anchors are
restricted to residues without altlocs.

Usage:
    python scripts/folddisco_loop_motif.py 6B8X_single_001.pdb --chain A --residues 178-193
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import gemmi
import numpy as np
from folddisco import parse_query_file
from jaxtyping import Float


ATOM_RECORDS = ("ATOM  ", "HETATM", "ANISOU")


class ResidueId(NamedTuple):
    """A residue identified by chain and sequence number (no insertion code)."""

    chain: str
    number: int

    @property
    def label(self) -> str:
        """FoldDisco motif syntax for this residue, e.g. ``A181``."""
        return f"{self.chain}{self.number}"


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
    """Pick the two altlocs to compare, exiting with a message if that is impossible.

    Parameters
    ----------
    altlocs : dict of ResidueId to set of str
        Output of :func:`find_altlocs` for the user's residue set.
    requested : list of str or None
        Altlocs passed with ``--altlocs``; None to use the only two present.

    Returns
    -------
    list of str
        The two altloc labels, in the order given (or sorted).
    """
    present = sorted(set().union(*altlocs.values()))
    if requested is None and len(present) != 2:
        sys.exit(
            f"need exactly two altlocs, found {present or 'none'} on these residues"
            + ("; choose two with --altlocs" if len(present) > 2 else "")
        )
    chosen = requested or present
    if len(set(chosen)) != 2 or not set(chosen) <= set(present):
        sys.exit(f"--altlocs must be two different labels from {present}")

    missing = residues_missing_altloc(altlocs, chosen)
    if missing:
        sys.exit(
            f"these residues lack altloc {' or '.join(chosen)} and would be deleted from "
            f"one conformer file: {', '.join(r.label for r in missing)}. "
            "Narrow --residues to exclude them."
        )
    return chosen


def describe(
    choice: MotifChoice,
    points_a: dict[ResidueId, Float[np.ndarray, "2 3"]],
    points_b: dict[ResidueId, Float[np.ndarray, "2 3"]],
    altlocs: list[str],
) -> str:
    """Human-readable table of the chosen residues' Cb-Cb distances in each conformer.

    Parameters
    ----------
    choice : MotifChoice
        Chosen motif.
    points_a, points_b : dict of ResidueId to ndarray
        :func:`backbone_points` of the two single-conformer structures.
    altlocs : list of str
        The two altloc labels, for column headers.

    Returns
    -------
    str
        Multi-line report.
    """
    lines = [f"motif: {choice.motif}", "", "loop residue scores (summed Å change to anchors):"]
    lines += [f"  {r.label:>8s}  {choice.loop_scores[r]:6.1f}" for r in choice.loop]
    lines += ["", "anchor scores (summed Å change to chosen loop residues):"]
    lines += [f"  {r.label:>8s}  {choice.anchor_scores[r]:6.1f}" for r in choice.anchors]
    lines += ["", f"Cb-Cb distances (Å), conformer {altlocs[0]} / {altlocs[1]}:"]
    lines.append("          " + "".join(f"{a.label:>14s}" for a in choice.anchors))
    for loop_residue in choice.loop:
        cells = "".join(
            f"{np.linalg.norm(points_a[loop_residue][1] - points_a[a][1]):6.1f} /"
            f"{np.linalg.norm(points_b[loop_residue][1] - points_b[a][1]):5.1f} "
            for a in choice.anchors
        )
        lines.append(f"  {loop_residue.label:>8s}" + cells)
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("pdb", type=Path, help="PDB file with two altlocs on the residues")
    parser.add_argument("--chain", required=True, help="chain of the residues")
    parser.add_argument(
        "--residues", required=True, help="residue numbers and ranges, e.g. 178-193 or 50,52-55"
    )
    parser.add_argument(
        "--altlocs", nargs=2, metavar=("ALT1", "ALT2"), help="altlocs to compare (default: auto)"
    )
    parser.add_argument(
        "--max-residues", type=int, default=4, help="max loop residues, and anchors (default 4)"
    )
    parser.add_argument(
        "--cutoff", type=float, default=12.0, help="loop-anchor distance cutoff in Å (default 12)"
    )
    parser.add_argument(
        "--min-separation",
        type=int,
        default=3,
        help="min sequence distance from anchors to the residue set (default 3)",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="where to write files (default: the PDB's directory)"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    args = parser.parse_args()

    try:
        numbers = parse_residue_spec(args.residues)
    except ValueError as error:
        parser.error(f"--residues: {error}")
    pdb = args.pdb.expanduser().resolve()
    loop = [ResidueId(args.chain, number) for number in numbers]
    lines = pdb.read_text().splitlines(keepends=True)
    altlocs = choose_altlocs(find_altlocs(lines, set(loop)), args.altlocs)

    output_dir = (args.output_dir or pdb.parent).expanduser().resolve()
    stem = output_stem(pdb, args.chain, numbers)
    conformer_paths = [output_dir / f"{stem}_alt{altloc}.pdb" for altloc in altlocs]
    query_path = output_dir / f"{stem}.folddisco_query.txt"
    existing = [p for p in [*conformer_paths, query_path] if p.exists()]
    if existing and not args.overwrite:
        sys.exit(f"already exists (pass --overwrite to replace): {', '.join(map(str, existing))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for altloc, path in zip(altlocs, conformer_paths, strict=True):
        path.write_text("".join(select_altloc(lines, set(loop), altloc)))
        print(f"wrote {path}")

    points_a, points_b = (backbone_points(gemmi.read_structure(str(p))) for p in conformer_paths)
    choice = choose_motif(
        points_a, points_b, loop, args.max_residues, args.cutoff, args.min_separation
    )
    if not choice.loop or not choice.anchors:
        sys.exit("no loop residue changes position relative to any altloc-free residue")

    query_lines = [
        f"{path}\t{choice.motif}\t{output_dir / f'{stem}_alt{altloc}.folddisco.json'}\n"
        for altloc, path in zip(altlocs, conformer_paths, strict=True)
    ]
    query_path.write_text("".join(query_lines))
    parse_query_file(query_path)
    print(f"wrote {query_path}\n")
    print(describe(choice, points_a, points_b, altlocs))
    print(f"\nrun: python {Path(__file__).with_name('folddisco.py')} {query_path}")


if __name__ == "__main__":
    main()
