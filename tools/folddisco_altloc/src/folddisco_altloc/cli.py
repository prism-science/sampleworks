"""Find structural contexts like each altloc conformer of a loop with the FoldDisco webserver.

serve    open the web interface in a browser
run      the whole pipeline in the terminal: split altlocs, choose a motif,
         search FoldDisco, print the summary table
motif    only write the conformer PDBs and the FoldDisco query file
search   run an existing FoldDisco query file
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .loop_motif import MotifError, output_stem, parse_residue_spec
from .pipeline import MotifOptions, prepare_motif, PreparedMotif, run_search, SearchOptions
from .query import (
    AA_SUBST_MODES,
    DEFAULT_DATABASES,
    prepare_queries,
    QueryFileError,
    RESULT_PAGE_URL,
    run_folddisco,
)
from .summary import format_table


DEFAULT_WORKDIR = Path("~/folddisco-altloc-runs")


def describe_motif(prepared: PreparedMotif) -> str:
    """Motif, residue scores and Cb-Cb distances per conformer, as plain text.

    Parameters
    ----------
    prepared : PreparedMotif
        Output of :func:`folddisco_altloc.pipeline.prepare_motif`.

    Returns
    -------
    str
        Multi-line report.
    """
    choice = prepared.choice
    lines = [f"motif: {choice.motif}", "", "moving residues (summed Å distance change to anchors):"]
    lines += [f"  {r.label:>8s}  {choice.loop_scores[r]:6.1f}" for r in choice.loop]
    lines += ["", "anchors, no altlocs (summed Å distance change to moving residues):"]
    lines += [f"  {r.label:>8s}  {choice.anchor_scores[r]:6.1f}" for r in choice.anchors]
    lines += ["", f"Cb-Cb distances (Å), conformer {' / '.join(prepared.altlocs)}:"]
    lines.append("          " + "".join(f"{a.label:>14s}" for a in choice.anchors))
    for moving in choice.loop:
        cells = "".join(
            f"{' / '.join(f'{d:.1f}' for d in prepared.cb_distances[(moving, a)]):>14s}"
            for a in choice.anchors
        )
        lines.append(f"  {moving.label:>8s}{cells}")
    return "\n".join(lines)


def default_run_dir(workdir: Path, pdb: Path, chain: str, residues: str) -> Path:
    """A new, timestamped run directory under ``workdir``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return workdir.expanduser() / f"{stamp}_{pdb.stem}_{chain}{residues.replace(',', '_')}"


def _add_motif_arguments(parser: argparse.ArgumentParser) -> None:
    """Structure, residue set and motif-choice options."""
    parser.add_argument("pdb", type=Path, help="PDB file with two altlocs on the residues")
    parser.add_argument("--chain", required=True, help="chain of the residues")
    parser.add_argument(
        "--residues", required=True, help="residue numbers and ranges, e.g. 158-167"
    )
    parser.add_argument("--altlocs", nargs=2, metavar=("ALT1", "ALT2"), help="altlocs to compare")
    parser.add_argument(
        "--motif-size",
        type=int,
        default=4,
        help="total motif residues, half moving and half anchors (even, default 4)",
    )
    parser.add_argument(
        "--cutoff", type=float, default=12.0, help="moving-anchor distance cutoff in Å (default 12)"
    )
    parser.add_argument(
        "--min-separation",
        type=int,
        default=3,
        help="min sequence distance from anchors to the residue set (default 3)",
    )


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    """FoldDisco search options shared by ``run`` and ``search``."""
    parser.add_argument(
        "--database",
        action="append",
        dest="databases",
        help=f"database to search; repeatable (default: {', '.join(DEFAULT_DATABASES)})",
    )
    parser.add_argument(
        "--aa-subst",
        choices=AA_SUBST_MODES,
        help="let motif residues match their substitution class, as in folddisco --aa-subst",
    )
    parser.add_argument(
        "--confident",
        action="store_true",
        help="keep hits matching >=80%% of the motif within 1 Å, as in folddisco --confident",
    )


def _motif_options(args: argparse.Namespace) -> MotifOptions:
    return MotifOptions(
        args.motif_size,
        args.cutoff,
        args.min_separation,
        tuple(args.altlocs) if args.altlocs else None,
    )


def _serve(args: argparse.Namespace) -> None:
    from .server import serve

    serve(args.host, args.port, args.workdir.expanduser(), open_browser=not args.no_browser)


def _run(args: argparse.Namespace) -> None:
    pdb = args.pdb.expanduser().resolve()
    run_dir = (
        args.output_dir or default_run_dir(args.workdir, pdb, args.chain, args.residues)
    ).expanduser()
    prepared = prepare_motif(
        pdb, args.chain, args.residues, _motif_options(args), run_dir.resolve()
    )
    print(f"run directory: {prepared.run_dir}\n\n{describe_motif(prepared)}\n")
    search_options = SearchOptions(
        args.aa_subst,
        args.confident,
        args.rmsd_max,
        args.min_matched,
        tuple(args.databases or DEFAULT_DATABASES),
    )
    outcome = run_search(prepared, search_options, on_message=lambda m: print(f"  {m}"))
    f = outcome.hit_filter
    print(
        f"\nhits with >= {f.min_matched} of {f.motif_size} motif residues, RMSD < {f.rmsd_max} Å"
        f"{', confident' if f.confident else ''}: targets (distinct residue numberings)\n"
    )
    print(format_table(outcome.rows))
    for altloc, ticket in outcome.tickets.items():
        print(f"\nconformer {altloc}: {RESULT_PAGE_URL}/{ticket}")


def _motif(args: argparse.Namespace) -> None:
    pdb = args.pdb.expanduser().resolve()
    run_dir = (args.output_dir or pdb.parent).expanduser().resolve()
    try:
        stem = output_stem(pdb, args.chain, parse_residue_spec(args.residues))
    except ValueError as error:
        raise MotifError(f"residues: {error}") from error
    existing = sorted(run_dir.glob(f"{stem}_alt*.pdb")) + sorted(
        run_dir.glob(f"{stem}.folddisco_query.txt")
    )
    if existing and not args.overwrite:
        sys.exit(f"already exists (pass --overwrite to replace): {', '.join(map(str, existing))}")
    prepared = prepare_motif(pdb, args.chain, args.residues, _motif_options(args), run_dir)
    for path in [*prepared.conformer_paths.values(), prepared.query_path]:
        print(f"wrote {path}")
    print(f"\n{describe_motif(prepared)}\n\nrun: folddisco-altloc search {prepared.query_path}")


def _search(args: argparse.Namespace) -> None:
    if args.validate_only:
        for query in prepare_queries(args.query_file, args.aa_subst):
            print(f"line {query.line_number}: {query.structure}\t{query.motif}")
        print("query file is valid")
        return
    runs = run_folddisco(
        args.query_file,
        tuple(args.databases or DEFAULT_DATABASES),
        args.poll_interval,
        args.max_wait,
        args.aa_subst,
        args.confident,
    )
    for run in runs:
        print(f"line {run.query.line_number} ({run.query.structure.name} {run.query.motif})")
        print(f"  results: {run.output}")
        print(f"  page:    {RESULT_PAGE_URL}/{run.ticket_id}")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with the four subcommands."""
    parser = argparse.ArgumentParser(
        prog="folddisco-altloc", description=__doc__.split("\n\n")[0] if __doc__ else None
    )
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="open the web interface")
    serve.add_argument(
        "--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1)"
    )
    serve.add_argument(
        "--port", type=int, default=8417, help="port; the next free one if taken (default 8417)"
    )
    serve.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="where runs are saved")
    serve.add_argument("--no-browser", action="store_true", help="do not open a browser")
    serve.set_defaults(handler=_serve)

    run = commands.add_parser("run", help="whole pipeline in the terminal")
    _add_motif_arguments(run)
    _add_search_arguments(run)
    run.add_argument(
        "--rmsd-max", type=float, default=0.5, help="count hits below this RMSD in Å (default 0.5)"
    )
    run.add_argument(
        "--min-matched", type=int, help="min matched motif residues to count (default: all)"
    )
    run.add_argument(
        "--workdir", type=Path, default=DEFAULT_WORKDIR, help="parent of the run directory"
    )
    run.add_argument(
        "--output-dir", type=Path, help="run directory (default: new one under --workdir)"
    )
    run.set_defaults(handler=_run)

    motif = commands.add_parser("motif", help="write conformer PDBs and a FoldDisco query file")
    _add_motif_arguments(motif)
    motif.add_argument("--output-dir", type=Path, help="default: the PDB's directory")
    motif.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    motif.set_defaults(handler=_motif)

    search = commands.add_parser("search", help="run a FoldDisco batch query file")
    search.add_argument(
        "query_file", type=Path, help="tab-separated: structure, motif, optional output"
    )
    _add_search_arguments(search)
    search.add_argument(
        "--validate-only", action="store_true", help="check the query file and exit"
    )
    search.add_argument(
        "--poll-interval", type=float, default=5.0, help="seconds between status checks"
    )
    search.add_argument(
        "--max-wait", type=float, default=1800.0, help="seconds to wait for each search"
    )
    search.set_defaults(handler=_search)
    return parser


def main() -> None:
    """Entry point for the ``folddisco-altloc`` command."""
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except (MotifError, QueryFileError) as error:
        sys.exit(f"error: {error}")
