"""Run motif searches on the FoldDisco webserver (search.foldseek.com/folddisco).

The input is a FoldDisco batch query file, the same format the ``folddisco query -q``
CLI accepts (see https://github.com/steineggerlab/folddisco/tree/master/query): one
query per line, tab separated, with

    1. path to the query structure (PDB/mmCIF)
    2. comma-separated motif residues, e.g. ``B57,B102,C195`` or ``164:H,195,247:ND``
    3. (optional) path to an output file

The file is validated in full before anything is submitted, and each line is then
sent as its own ticket, since the web API takes one structure and one motif per
request. Each ticket is polled until the search finishes and its results are
written to the line's output file: raw JSON for a ``.json`` path, otherwise a TSV
with one row per hit. Lines without an output file get
``<query file stem>_line<N>.folddisco.json`` next to the query file.

The webserver runs a fixed ``folddisco query`` command and accepts no extra flags,
so two CLI options are reproduced client side:

    --aa-subst MODE   each motif residue without an explicit ``:ALT`` gets its
                      substitution class written into the motif (``A181`` -> ``A181:DE``)
    --confident       hits are filtered after download to those matching at least
                      80% of the motif residues within 1 Å RMSD

Usage:
    python scripts/folddisco.py query/zinc_finger.txt
    python scripts/folddisco.py query/zinc_finger.txt --aa-subst group --confident
"""

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

import gemmi
import requests


API_URL = "https://search.foldseek.com/api"
RESULT_PAGE_URL = "https://search.foldseek.com/result/folddisco"

TSV_HIT_FIELDS = (
    "target",
    "idfscore",
    "rmsd",
    "nodecount",
    "queryresidues",
    "targetresidues",
)

DEFAULT_DATABASES = (
    "esm30_folddisco",
    "afdb-proteome_folddisco",
    "afdb50_folddisco",
    "pdb_folddisco",
    "BFVD_folddisco",
)

STRUCTURE_SUFFIXES = {".pdb", ".ent", ".cif", ".mmcif"}

# Residue classes of FoldDisco's --aa-subst modes (README, "Amino acid substitution").
# "blosum62" needs no table: FoldDisco expands ":*" itself.
AA_SUBST_CLASSES = {
    "group": ("RHK", "DE", "NQST", "FWY", "AVLIMC", "GP"),
    "size": ("GAS", "CDPNT", "QEHV", "MILKR", "FWY"),
}
AA_SUBST_MODES = (*AA_SUBST_CLASSES, "blosum62")

# FoldDisco --confident: at least this fraction of motif residues within this RMSD (Å).
CONFIDENT_MIN_FRACTION = 0.8
CONFIDENT_MAX_RMSD = 1.0

# One motif residue, following the FoldDisco README "Residue & motif syntax":
#   chain + residue number, optionally separated by "_"  B57, B_57, AA_250, 10_100
#   a bare residue number                                  250
#   an inclusive range                                     1-10, A1-10
#   an optional substitution after ":"                     164:H, 247:ND, 30:X, 12:*
# A chain ID containing digits needs the "_" separator, otherwise "A181" could be
# chain A18, residue 1.
# Substitutions are one-letter amino acid codes (uppercase) or the categories
# X (any), p (positive), n (negative), h (polar), b (hydrophobic), a (aromatic),
# and * (the --aa-subst scheme).
_RESIDUE_RE = re.compile(
    r"""
    ^
    (?:(?P<chain>[A-Za-z]+)_?|(?P<sepchain>[A-Za-z0-9]+)_)?
    (?P<start>[0-9]+)
    (?:-(?P<end>[0-9]+))?
    (?::(?P<alt>[ACDEFGHIKLMNPQRSTVWYXpnhba*]+))?
    $
    """,
    re.VERBOSE,
)


class QueryFileError(ValueError):
    """Raised when a query file does not follow the FoldDisco batch query format."""


@dataclass(frozen=True)
class MotifQuery:
    """One validated line of a FoldDisco batch query file.

    Attributes
    ----------
    structure : Path
        Resolved path to the query structure.
    motif : str
        Comma-separated motif residues, in FoldDisco syntax.
    output : Path or None
        Output path from the optional third column.
    line_number : int
        1-based line number in the query file.
    """

    structure: Path
    motif: str
    output: Path | None
    line_number: int


def _validate_motif(motif: str, where: str) -> str:
    """Check that every residue in a motif follows FoldDisco's residue syntax.

    Parameters
    ----------
    motif : str
        Comma-separated motif residues.
    where : str
        ``file:line`` location used in error messages.

    Returns
    -------
    str
        The motif, unchanged.

    Raises
    ------
    QueryFileError
        If the motif is empty or any residue is malformed.
    """
    if not motif:
        raise QueryFileError(f"{where}: motif column is empty")
    if any(c.isspace() for c in motif):
        raise QueryFileError(f"{where}: motif {motif!r} contains whitespace")

    for residue in motif.split(","):
        match = _RESIDUE_RE.match(residue)
        if match is None:
            raise QueryFileError(
                f"{where}: {residue!r} is not a valid residue "
                "(expected e.g. B57, B_57, 250, 1-10, 164:H, 247:ND)"
            )
        end = match["end"]
        if end is not None and int(end) < int(match["start"]):
            raise QueryFileError(f"{where}: range {residue!r} ends before it starts")
    return motif


def _resolve_structure(raw: str, query_file: Path, where: str) -> Path:
    """Locate a query structure named in a query file.

    Parameters
    ----------
    raw : str
        Structure path as written in the query file.
    query_file : Path
        The query file, whose directory is the fallback for relative paths.
    where : str
        ``file:line`` location used in error messages.

    Returns
    -------
    Path
        Absolute path to the existing structure file.

    Raises
    ------
    QueryFileError
        If the path lacks a structure-file suffix or does not exist.
    """
    path = Path(raw).expanduser()
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    if not suffixes or suffixes[-1] not in STRUCTURE_SUFFIXES:
        raise QueryFileError(
            f"{where}: {raw!r} is not a structure file "
            f"(expected one of {sorted(STRUCTURE_SUFFIXES)}, optionally .gz)"
        )

    # FoldDisco's own examples use paths relative to the working directory
    # (query/4CHA.pdb from the repo root), so try that first, then fall back to
    # the query file's own directory.
    candidates = [path] if path.is_absolute() else [path, query_file.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise QueryFileError(
        f"{where}: structure file {raw!r} not found "
        f"(looked in {', '.join(str(c) for c in candidates)})"
    )


def parse_query_file(query_file: str | Path) -> list[MotifQuery]:
    """Parse and validate a FoldDisco batch query file.

    Blank lines and lines starting with "#" are skipped. A trailing empty third
    column (FoldDisco's own examples end each line with a tab) means no output file.

    Parameters
    ----------
    query_file : str or Path
        Tab-separated query file: structure, motif, optional output path.

    Returns
    -------
    list of MotifQuery
        One entry per query line, in file order.

    Raises
    ------
    QueryFileError
        On the first malformed line, or if the file has no queries.
    """
    query_file = Path(query_file).expanduser()
    if not query_file.is_file():
        raise QueryFileError(f"query file {query_file} does not exist")

    queries = []
    for line_number, line in enumerate(query_file.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        where = f"{query_file}:{line_number}"

        columns = line.split("\t")
        if len(columns) == 3 and not columns[2].strip():
            columns = columns[:2]
        if len(columns) not in (2, 3):
            hint = " (columns must be tab separated)" if len(columns) == 1 else ""
            raise QueryFileError(
                f"{where}: expected 2 or 3 tab-separated columns, got {len(columns)}{hint}"
            )

        structure = _resolve_structure(columns[0].strip(), query_file, where)
        motif = _validate_motif(columns[1].strip(), where)
        output = Path(columns[2].strip()).expanduser() if len(columns) == 3 else None
        queries.append(MotifQuery(structure, motif, output, line_number))

    if not queries:
        raise QueryFileError(f"{query_file}: no queries found")
    return queries


def residue_codes(structure: Path) -> dict[tuple[str, int], str]:
    """One-letter amino acid code of every residue in the first model.

    Parameters
    ----------
    structure : Path
        PDB or mmCIF file.

    Returns
    -------
    dict of (str, int) to str
        ``(chain, residue number) -> one-letter code``; non-amino-acid residues are
        left out.
    """
    codes: dict[tuple[str, int], str] = {}
    for chain in gemmi.read_structure(str(structure))[0]:
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info is None or not info.is_amino_acid() or not info.one_letter_code:
                continue
            codes.setdefault((chain.name, residue.seqid.num), info.one_letter_code.upper())
    return codes


def _substitution(code: str, mode: str, label: str) -> str:
    """The ``:ALT`` string FoldDisco's ``--aa-subst mode`` would use for ``code``."""
    if mode == "blosum62":
        return "*"
    for residue_class in AA_SUBST_CLASSES[mode]:
        if code in residue_class:
            return residue_class
    raise QueryFileError(f"{label}: residue type {code!r} has no --aa-subst {mode} class")


def expand_motif(motif: str) -> list[tuple[str, str | None, int, str | None]]:
    """Split a validated motif into single residues, expanding ranges.

    Parameters
    ----------
    motif : str
        Motif that passed :func:`_validate_motif`.

    Returns
    -------
    list of tuple
        ``(written chain prefix, chain or None, residue number, ALT or None)`` per
        residue, in motif order.
    """
    residues = []
    for token in motif.split(","):
        match = _RESIDUE_RE.match(token)
        assert match is not None, f"motif {motif!r} was not validated"
        chain = match["chain"] or match["sepchain"]
        prefix = f"{match['sepchain']}_" if match["sepchain"] else (match["chain"] or "")
        start = int(match["start"])
        for number in range(start, int(match["end"] or start) + 1):
            residues.append((prefix, chain, number, match["alt"]))
    return residues


def substituted_motif(query: MotifQuery, mode: str) -> str:
    """Rewrite a query's motif as FoldDisco's ``--aa-subst mode`` would interpret it.

    Residues that already carry ``:ALT`` are kept as written; the others get the
    substitution class of their amino acid in the query structure. Ranges are
    expanded to single residues.

    Parameters
    ----------
    query : MotifQuery
        Validated query.
    mode : str
        One of ``AA_SUBST_MODES``.

    Returns
    -------
    str
        Motif with an explicit ``:ALT`` on every residue.

    Raises
    ------
    QueryFileError
        If a motif residue is not an amino acid in the query structure.
    """
    codes = residue_codes(query.structure)
    tokens = []
    for prefix, chain, number, alt in expand_motif(query.motif):
        label = f"{query.structure.name} {prefix}{number}"
        if alt is None:
            code = next(
                (c for (ch, n), c in codes.items() if n == number and chain in (None, ch)),
                None,
            )
            if code is None:
                raise QueryFileError(f"{label}: no amino acid with that number")
            alt = _substitution(code, mode, label)
        tokens.append(f"{prefix}{number}:{alt}")
    return ",".join(tokens)


def confident_hits(results: dict, motif_size: int) -> dict:
    """Keep only the hits FoldDisco's ``--confident`` would report.

    Parameters
    ----------
    results : dict
        Result JSON from :func:`fetch_results`.
    motif_size : int
        Number of residues in the query motif.

    Returns
    -------
    dict
        Copy of ``results`` whose alignments match at least
        ``CONFIDENT_MIN_FRACTION`` of the motif within ``CONFIDENT_MAX_RMSD``.
    """
    min_nodes = CONFIDENT_MIN_FRACTION * motif_size
    return {
        **results,
        "results": [
            {
                **database,
                "alignments": [
                    hit
                    for hit in database["alignments"]
                    if hit["nodecount"] >= min_nodes and hit["rmsd"] <= CONFIDENT_MAX_RMSD
                ],
            }
            for database in results["results"]
        ],
    }


def submit_query(
    query: MotifQuery,
    databases: tuple[str, ...] | list[str] = DEFAULT_DATABASES,
    timeout: float = 60.0,
) -> str:
    """Submit one motif query to the FoldDisco API.

    Parameters
    ----------
    query : MotifQuery
        Validated query to submit; its motif is sent as written.
    databases : sequence of str, optional
        FoldDisco databases to search.
    timeout : float, optional
        Request timeout in seconds.

    Returns
    -------
    str
        The ticket ID the server assigned to the search.

    Raises
    ------
    requests.HTTPError
        If the server rejects the submission.
    """
    data = [("database[]", db) for db in databases]
    data.append(("motif", query.motif))
    with open(query.structure, "rb") as structure_file:
        response = requests.post(
            f"{API_URL}/ticket/folddisco",
            files={"q": structure_file},
            data=data,
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()["id"]


def wait_for_ticket(
    ticket_id: str,
    poll_interval: float = 5.0,
    max_wait: float = 1800.0,
    timeout: float = 60.0,
) -> None:
    """Poll a FoldDisco ticket until its search is complete.

    Parameters
    ----------
    ticket_id : str
        Ticket ID returned by :func:`submit_query`.
    poll_interval : float, optional
        Seconds between status checks.
    max_wait : float, optional
        Seconds to wait in total before giving up.
    timeout : float, optional
        Per-request timeout in seconds.

    Raises
    ------
    RuntimeError
        If the search fails, the server no longer knows the ticket, or it is still
        running after ``max_wait`` seconds.
    """
    deadline = time.monotonic() + max_wait
    while True:
        response = requests.get(f"{API_URL}/ticket/{ticket_id}", timeout=timeout)
        response.raise_for_status()
        status = response.json()["status"]
        if status == "COMPLETE":
            return
        if status not in ("PENDING", "RUNNING"):
            raise RuntimeError(f"FoldDisco ticket {ticket_id} ended with status {status}")
        if time.monotonic() + poll_interval > deadline:
            raise RuntimeError(
                f"FoldDisco ticket {ticket_id} still {status} after {max_wait:.0f} s"
            )
        time.sleep(poll_interval)


def fetch_results(ticket_id: str, timeout: float = 300.0) -> dict:
    """Download the results of a completed FoldDisco search.

    Parameters
    ----------
    ticket_id : str
        Ticket ID of a search whose status is ``COMPLETE``.
    timeout : float, optional
        Request timeout in seconds; responses are several MB.

    Returns
    -------
    dict
        The server's result JSON: ``{"motif": str, "results": [{"db": str,
        "alignments": [hit, ...], ...}, ...]}``, one entry per searched database.
    """
    response = requests.get(f"{API_URL}/result/folddisco/{ticket_id}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def write_results(results: dict, output: Path) -> None:
    """Write FoldDisco results to disk, as JSON or as a per-hit TSV.

    Parameters
    ----------
    results : dict
        Result JSON from :func:`fetch_results`.
    output : Path
        Destination. A ``.json`` suffix writes the raw JSON; any other suffix
        writes a TSV with one row per hit and the database as the first column.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(results))
        return

    with open(output, "w", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        writer.writerow(("db", *TSV_HIT_FIELDS))
        for database in results["results"]:
            for hit in database["alignments"]:
                writer.writerow((database["db"], *(hit[field] for field in TSV_HIT_FIELDS)))


def default_output(query_file: Path, query: MotifQuery) -> Path:
    """Output path for a query line that has no output column.

    Parameters
    ----------
    query_file : Path
        The batch query file the line came from.
    query : MotifQuery
        The query line.

    Returns
    -------
    Path
        ``<query file stem>_line<N>.folddisco.json`` in the query file's directory.
    """
    return query_file.parent / f"{query_file.stem}_line{query.line_number}.folddisco.json"


@dataclass(frozen=True)
class FolddiscoRun:
    """A finished FoldDisco search for one query line.

    Attributes
    ----------
    query : MotifQuery
        The query line that was searched.
    ticket_id : str
        Server ticket ID; the results page is ``RESULT_PAGE_URL/<ticket_id>``.
    output : Path
        Where the results were written.
    """

    query: MotifQuery
    ticket_id: str
    output: Path


def prepare_queries(query_file: Path, aa_subst: str | None) -> list[MotifQuery]:
    """Parse a query file and apply ``--aa-subst`` to every motif.

    Parameters
    ----------
    query_file : Path
        FoldDisco batch query file.
    aa_subst : str or None
        One of ``AA_SUBST_MODES``, or None to send motifs as written.

    Returns
    -------
    list of MotifQuery
        Queries ready to submit, in file order.

    Raises
    ------
    QueryFileError
        If the file is malformed or a motif residue cannot be substituted.
    """
    queries = parse_query_file(query_file)
    if aa_subst is None:
        return queries
    return [replace(query, motif=substituted_motif(query, aa_subst)) for query in queries]


def run_folddisco(
    query_file: str | Path,
    databases: tuple[str, ...] | list[str] = DEFAULT_DATABASES,
    poll_interval: float = 5.0,
    max_wait: float = 1800.0,
    aa_subst: str | None = None,
    confident: bool = False,
) -> list[FolddiscoRun]:
    """Validate a query file, run every query on FoldDisco and write the results.

    All queries are prepared before the first submission and submitted before any
    is awaited, so the searches run on the server concurrently. Writes one results
    file per query line.

    Parameters
    ----------
    query_file : str or Path
        FoldDisco batch query file.
    databases : sequence of str, optional
        FoldDisco databases to search.
    poll_interval : float, optional
        Seconds between ticket status checks.
    max_wait : float, optional
        Seconds to wait for each search before giving up.
    aa_subst : str or None, optional
        FoldDisco ``--aa-subst`` mode to apply to every motif (see
        :func:`substituted_motif`).
    confident : bool, optional
        Keep only hits FoldDisco's ``--confident`` would report (see
        :func:`confident_hits`).

    Returns
    -------
    list of FolddiscoRun
        One entry per query line, in file order, with the motif as submitted.

    Raises
    ------
    QueryFileError
        If the query file is malformed; nothing is submitted in that case.
    RuntimeError
        If a search fails or does not finish within ``max_wait``.
    requests.HTTPError
        If the server rejects a request.
    """
    query_file = Path(query_file).expanduser()
    queries = prepare_queries(query_file, aa_subst)
    ticket_ids = [submit_query(query, databases) for query in queries]

    runs = []
    for query, ticket_id in zip(queries, ticket_ids, strict=True):
        wait_for_ticket(ticket_id, poll_interval, max_wait)
        results = fetch_results(ticket_id)
        if confident:
            results = confident_hits(results, len(expand_motif(query.motif)))
        output = query.output or default_output(query_file, query)
        write_results(results, output)
        runs.append(FolddiscoRun(query, ticket_id, output))
    return runs


def main() -> None:
    """Command-line entry point: validate a query file, run it and save the results."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("query_file", type=Path, help="FoldDisco batch query file (.txt)")
    parser.add_argument(
        "--database",
        action="append",
        dest="databases",
        help=f"database to search; repeatable (default: {', '.join(DEFAULT_DATABASES)})",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="check the query file and exit"
    )
    parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="seconds between status checks"
    )
    parser.add_argument(
        "--max-wait", type=float, default=1800.0, help="seconds to wait for each search"
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
    args = parser.parse_args()

    if args.validate_only:
        for query in prepare_queries(args.query_file, args.aa_subst):
            print(f"line {query.line_number}: {query.structure}\t{query.motif}")
        print("query file is valid")
        return

    databases = args.databases or DEFAULT_DATABASES
    runs = run_folddisco(
        args.query_file,
        databases,
        args.poll_interval,
        args.max_wait,
        args.aa_subst,
        args.confident,
    )
    for run in runs:
        print(f"line {run.query.line_number} ({run.query.structure.name} {run.query.motif})")
        print(f"  results: {run.output}")
        print(f"  page:    {RESULT_PAGE_URL}/{run.ticket_id}")


if __name__ == "__main__":
    main()
