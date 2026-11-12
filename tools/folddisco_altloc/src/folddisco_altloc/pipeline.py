"""End-to-end run: split a loop's two altlocs, choose a motif, search FoldDisco, summarize.

Used by both the web interface (:mod:`folddisco_altloc.server`) and
``folddisco-altloc run``. Every file a run produces goes into one run directory:
the two single-conformer PDBs, the FoldDisco query file, the raw result JSON per
conformer, and the summary table as text and CSV.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np

from .loop_motif import (
    anchor_candidates,
    backbone_points,
    choose_altlocs,
    choose_motif,
    find_altlocs,
    MotifChoice,
    MotifError,
    output_stem,
    parse_residue_spec,
    ResidueId,
    select_altloc,
)
from .query import (
    DEFAULT_DATABASES,
    expand_motif,
    fetch_results,
    parse_query_file,
    prepare_queries,
    submit_query,
    wait_for_ticket,
)
from .summary import DatabaseRow, format_table, HitFilter, summarize, to_csv


@dataclass(frozen=True)
class MotifOptions:
    """How the motif is chosen.

    Attributes
    ----------
    motif_size : int
        Total motif residues, split evenly between moving residues and anchors.
    cutoff : float
        Loop-anchor pairs farther apart than this (Å) in both conformers are ignored.
    min_separation : int
        Minimum sequence distance between anchors and the residue set.
    altlocs : tuple of str or None
        The two altlocs to compare; None picks them automatically.
    """

    motif_size: int = 4
    cutoff: float = 12.0
    min_separation: int = 3
    altlocs: tuple[str, str] | None = None


@dataclass(frozen=True)
class SearchOptions:
    """How FoldDisco is searched and which hits are counted.

    Attributes
    ----------
    aa_subst : str or None
        FoldDisco ``--aa-subst`` mode (``group``, ``size``, ``blosum62``) or None.
    confident : bool
        Count only hits passing FoldDisco's ``--confident`` rule.
    rmsd_max : float
        Count hits with RMSD below this (Å).
    min_matched : int or None
        Minimum matched motif residues for a hit to count; None means all of them.
    databases : tuple of str
        FoldDisco databases to search.
    """

    aa_subst: str | None = None
    confident: bool = False
    rmsd_max: float = 0.5
    min_matched: int | None = None
    databases: tuple[str, ...] = DEFAULT_DATABASES


@dataclass(frozen=True)
class PreparedMotif:
    """Files and motif for one residue set, ready to search.

    Attributes
    ----------
    run_dir : Path
        Directory holding every file of the run.
    stem : str
        File-name stem, e.g. ``5MHX_single_001_A158-167``.
    conformer_paths : dict of str to Path
        Single-conformer PDB per altloc label, in comparison order.
    query_path : Path
        FoldDisco batch query file (one line per conformer).
    choice : MotifChoice
        Chosen moving residues and anchors.
    cb_distances : dict of (ResidueId, ResidueId) to tuple of float
        Cb-Cb distance (Å) per (moving residue, anchor), one value per conformer.
    """

    run_dir: Path
    stem: str
    conformer_paths: dict[str, Path]
    query_path: Path
    choice: MotifChoice
    cb_distances: dict[tuple[ResidueId, ResidueId], tuple[float, ...]] = field(repr=False)

    @property
    def altlocs(self) -> list[str]:
        """The two altloc labels, in comparison order."""
        return list(self.conformer_paths)


@dataclass(frozen=True)
class SearchOutcome:
    """Result of searching both conformers.

    Attributes
    ----------
    submitted_motif : str
        Motif as sent to FoldDisco (after any ``--aa-subst`` rewriting).
    tickets : dict of str to str
        FoldDisco ticket ID per altloc label.
    result_paths : dict of str to Path
        Raw result JSON per altloc label.
    rows : list of DatabaseRow
        Summary table.
    hit_filter : HitFilter
        Which hits were counted.
    """

    submitted_motif: str
    tickets: dict[str, str]
    result_paths: dict[str, Path]
    rows: list[DatabaseRow]
    hit_filter: HitFilter


def _cb_distances(
    choice: MotifChoice, points: list[dict[ResidueId, np.ndarray]]
) -> dict[tuple[ResidueId, ResidueId], tuple[float, ...]]:
    """Cb-Cb distance per (moving residue, anchor) in each conformer."""
    return {
        (moving, anchor): tuple(
            float(np.linalg.norm(conformer[moving][1] - conformer[anchor][1]))
            for conformer in points
        )
        for moving in choice.loop
        for anchor in choice.anchors
    }


def _write_conformers(
    lines: list[str], residues: list[ResidueId], altlocs: list[str], run_dir: Path, stem: str
) -> dict[str, Path]:
    """Write one PDB per altloc with only that conformer of ``residues``."""
    paths = {}
    for altloc in altlocs:
        path = run_dir / f"{stem}_alt{altloc}.pdb"
        path.write_text("".join(select_altloc(lines, set(residues), altloc)))
        paths[altloc] = path
    return paths


def prepare_motif(
    pdb_path: Path, chain: str, residue_spec: str, options: MotifOptions, run_dir: Path
) -> PreparedMotif:
    """Write the two conformer PDBs, choose a motif and write the query file.

    Parameters
    ----------
    pdb_path : Path
        PDB-format structure with two altlocs on the residue set.
    chain : str
        Chain of the residue set.
    residue_spec : str
        Residue numbers and ranges, e.g. ``158-167``.
    options : MotifOptions
        Motif choice settings.
    run_dir : Path
        Output directory; created if needed.

    Returns
    -------
    PreparedMotif
        Paths, motif and per-pair distances.

    Raises
    ------
    MotifError
        If the residue set, altlocs or motif size cannot produce a motif.
    """
    if options.motif_size < 2 or options.motif_size % 2:
        raise MotifError("motif size must be an even number of at least 2")
    try:
        numbers = parse_residue_spec(residue_spec)
    except ValueError as error:
        raise MotifError(f"residues: {error}") from error
    residues = [ResidueId(chain, number) for number in numbers]
    lines = pdb_path.read_text().splitlines(keepends=True)
    altlocs = choose_altlocs(
        find_altlocs(lines, set(residues)), list(options.altlocs) if options.altlocs else None
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(pdb_path, chain, numbers)
    conformer_paths = _write_conformers(lines, residues, altlocs, run_dir, stem)
    points = [backbone_points(gemmi.read_structure(str(p))) for p in conformer_paths.values()]
    if not any(r in points[0] and r in points[1] for r in residues):
        raise MotifError(f"no amino acid of chain {chain} {residue_spec} has N, CA and C atoms")
    if not anchor_candidates(points[0], points[1], residues, options.min_separation):
        raise MotifError("no residue without altlocs is available as an anchor")

    choice = choose_motif(
        points[0],
        points[1],
        residues,
        options.motif_size // 2,
        options.cutoff,
        options.min_separation,
    )
    if not choice.loop or not choice.anchors:
        raise MotifError("no residue in the set moves relative to any residue without altlocs")

    query_path = run_dir / f"{stem}.folddisco_query.txt"
    query_path.write_text(
        "".join(
            f"{path}\t{choice.motif}\t{run_dir / f'{stem}_alt{altloc}.folddisco.json'}\n"
            for altloc, path in conformer_paths.items()
        )
    )
    parse_query_file(query_path)
    return PreparedMotif(
        run_dir, stem, conformer_paths, query_path, choice, _cb_distances(choice, points)
    )


def run_search(
    prepared: PreparedMotif,
    options: SearchOptions,
    on_message: Callable[[str], None] = lambda message: None,
) -> SearchOutcome:
    """Search both conformers on FoldDisco, save raw results and summarize them.

    Parameters
    ----------
    prepared : PreparedMotif
        Output of :func:`prepare_motif`.
    options : SearchOptions
        Search and counting settings.
    on_message : callable, optional
        Receives progress messages.

    Returns
    -------
    SearchOutcome
        Tickets, result paths and the summary table. Also writes
        ``<stem>.summary.txt`` and ``<stem>.summary.csv`` to the run directory.
    """
    queries = prepare_queries(prepared.query_path, options.aa_subst)
    submitted_motif = queries[0].motif
    on_message(f"submitting motif {submitted_motif}")
    tickets = {
        altloc: submit_query(query, options.databases)
        for altloc, query in zip(prepared.altlocs, queries, strict=True)
    }

    results, result_paths = {}, {}
    for altloc, query in zip(prepared.altlocs, queries, strict=True):
        on_message(f"conformer {altloc}: waiting for ticket {tickets[altloc]}")
        wait_for_ticket(
            tickets[altloc],
            on_status=lambda status, a=altloc: on_message(f"conformer {a}: {status}"),
        )
        on_message(f"conformer {altloc}: downloading results")
        results[altloc] = fetch_results(tickets[altloc])
        assert query.output is not None, "prepare_motif always writes an output column"
        query.output.write_text(json.dumps(results[altloc]))
        result_paths[altloc] = query.output

    motif_size = len(expand_motif(submitted_motif))
    hit_filter = HitFilter(
        motif_size, options.min_matched or motif_size, options.rmsd_max, options.confident
    )
    rows = summarize(results, hit_filter)
    (prepared.run_dir / f"{prepared.stem}.summary.txt").write_text(format_table(rows) + "\n")
    (prepared.run_dir / f"{prepared.stem}.summary.csv").write_text(to_csv(rows))
    on_message("done")
    return SearchOutcome(submitted_motif, tickets, result_paths, rows, hit_filter)
