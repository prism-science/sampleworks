"""Count how well each conformer's motif context is represented in FoldDisco results.

A hit counts when it matches at least ``min_matched`` motif residues with RMSD below
``rmsd_max`` (and, if requested, passes FoldDisco's ``--confident`` rule). Hits are
counted as unique target structures, and as distinct matched residue numberings, a
rough proxy for distinct proteins: repeated depositions of one protein usually match
the same residue numbers.
"""

import csv
import io
import os
import re
from dataclasses import dataclass

from .query import CONFIDENT_MAX_RMSD, CONFIDENT_MIN_FRACTION


# Hits per database the webserver returns (folddisco query --top 1000).
SERVER_HIT_LIMIT = 1000
# A counted hit this far down a full-length list suggests more were cut off.
TRUNCATION_FRACTION = 0.8


@dataclass(frozen=True)
class HitFilter:
    """Which hits count as close matches.

    Attributes
    ----------
    motif_size : int
        Number of residues in the query motif.
    min_matched : int
        Minimum matched motif residues.
    rmsd_max : float
        Hits must have RMSD strictly below this (Å).
    confident : bool
        Also require FoldDisco's ``--confident`` rule.
    """

    motif_size: int
    min_matched: int
    rmsd_max: float
    confident: bool = False

    def accepts(self, hit: dict) -> bool:
        """Whether ``hit`` (one alignment from the result JSON) counts."""
        if hit["nodecount"] < self.min_matched or hit["rmsd"] >= self.rmsd_max:
            return False
        if self.confident:
            return (
                hit["nodecount"] >= CONFIDENT_MIN_FRACTION * self.motif_size
                and hit["rmsd"] <= CONFIDENT_MAX_RMSD
            )
        return True


@dataclass(frozen=True)
class ConformerCounts:
    """Close matches of one conformer in one database.

    Attributes
    ----------
    targets : frozenset of str
        Unique target structures with at least one counted hit.
    numberings : int
        Distinct matched residue numberings among counted hits.
    returned : int
        Hits the server returned for this database.
    last_rank : int or None
        0-based position of the last counted hit in the returned list.
    """

    targets: frozenset[str]
    numberings: int
    returned: int
    last_rank: int | None

    @property
    def possibly_truncated(self) -> bool:
        """Whether counted hits reach so far down a full list that more may be cut off."""
        return (
            self.returned >= SERVER_HIT_LIMIT
            and self.last_rank is not None
            and self.last_rank >= TRUNCATION_FRACTION * self.returned
        )


@dataclass(frozen=True)
class DatabaseRow:
    """One database's row of the summary table.

    Attributes
    ----------
    database : str
        FoldDisco database name.
    counts : dict of str to ConformerCounts
        Per altloc label.
    """

    database: str
    counts: dict[str, ConformerCounts]

    @property
    def shared(self) -> int:
        """Targets counted for every conformer."""
        return len(frozenset.intersection(*(c.targets for c in self.counts.values())))


def _target_name(target: str) -> str:
    """Structure ID from a FoldDisco target path, e.g. ``/db/pdb/4cha.ent`` -> ``4cha``."""
    return os.path.basename(target).split(".")[0]


def _numbering(target_residues: str) -> tuple[int, ...]:
    """Residue numbers of a hit's matched residues (``A39,_,B44`` -> ``(39, 44)``)."""
    return tuple(
        int(match.group())
        for residue in target_residues.split(",")
        if (match := re.search(r"\d+$", residue))
    )


def count_conformer(alignments: list[dict], hit_filter: HitFilter) -> ConformerCounts:
    """Count close matches in one database's alignment list.

    Parameters
    ----------
    alignments : list of dict
        ``alignments`` of one database from the result JSON, in server order.
    hit_filter : HitFilter
        Which hits count.

    Returns
    -------
    ConformerCounts
        Counts for this conformer and database.
    """
    counted = [(rank, hit) for rank, hit in enumerate(alignments) if hit_filter.accepts(hit)]
    return ConformerCounts(
        targets=frozenset(_target_name(hit["target"]) for _, hit in counted),
        numberings=len({_numbering(hit["targetresidues"]) for _, hit in counted}),
        returned=len(alignments),
        last_rank=counted[-1][0] if counted else None,
    )


def summarize(results: dict[str, dict], hit_filter: HitFilter) -> list[DatabaseRow]:
    """Build the per-database summary table for several conformers.

    Parameters
    ----------
    results : dict of str to dict
        Raw result JSON per altloc label.
    hit_filter : HitFilter
        Which hits count.

    Returns
    -------
    list of DatabaseRow
        One row per database, in the order of the first conformer's results.
    """
    by_database = {
        altloc: {entry["db"]: entry["alignments"] for entry in result["results"]}
        for altloc, result in results.items()
    }
    databases = next(iter(by_database.values())).keys()
    return [
        DatabaseRow(
            database,
            {
                altloc: count_conformer(per_db.get(database, []), hit_filter)
                for altloc, per_db in by_database.items()
            },
        )
        for database in databases
    ]


def totals(rows: list[DatabaseRow]) -> dict[str, int]:
    """Targets per altloc summed over databases, plus ``"shared"``.

    Parameters
    ----------
    rows : list of DatabaseRow
        Output of :func:`summarize`.

    Returns
    -------
    dict of str to int
        Altloc label -> total targets, and ``"shared"`` -> total shared targets.
    """
    altlocs = rows[0].counts.keys() if rows else []
    result = {altloc: sum(len(row.counts[altloc].targets) for row in rows) for altloc in altlocs}
    result["shared"] = sum(row.shared for row in rows)
    return result


def format_table(rows: list[DatabaseRow]) -> str:
    """Plain-text table: targets (distinct numberings) per conformer, and shared.

    Parameters
    ----------
    rows : list of DatabaseRow
        Output of :func:`summarize`.

    Returns
    -------
    str
        Multi-line table; ``*`` marks counts that may be cut off by the hit limit.
    """
    altlocs = list(rows[0].counts) if rows else []
    header = (
        f"{'database':26s}"
        + "".join(f"{'conformer ' + a:>16s}" for a in altlocs)
        + f"{'shared':>8s}"
    )
    lines = [header]
    for row in rows:
        cells = "".join(
            f"{f'{len(c.targets)} ({c.numberings})' + ('*' if c.possibly_truncated else ' '):>16s}"
            for c in row.counts.values()
        )
        lines.append(f"{row.database:26s}{cells}{row.shared:8d}")
    total = totals(rows)
    lines.append(
        f"{'total':26s}" + "".join(f"{total[a]:>15d} " for a in altlocs) + f"{total['shared']:8d}"
    )
    if any(c.possibly_truncated for row in rows for c in row.counts.values()):
        lines.append(
            f"* counted hits reach the last {100 - TRUNCATION_FRACTION * 100:.0f}% of a "
            f"{SERVER_HIT_LIMIT}-hit list; more may have been cut off"
        )
    return "\n".join(lines)


def to_csv(rows: list[DatabaseRow]) -> str:
    """CSV with one line per database and conformer.

    Parameters
    ----------
    rows : list of DatabaseRow
        Output of :func:`summarize`.

    Returns
    -------
    str
        CSV text with a header line.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "database",
            "conformer",
            "targets",
            "distinct_numberings",
            "shared_targets",
            "hits_returned",
            "last_counted_rank",
            "possibly_truncated",
        ]
    )
    for row in rows:
        for altloc, c in row.counts.items():
            writer.writerow(
                [
                    row.database,
                    altloc,
                    len(c.targets),
                    c.numberings,
                    row.shared,
                    c.returned,
                    "" if c.last_rank is None else c.last_rank,
                    c.possibly_truncated,
                ]
            )
    return buffer.getvalue()
