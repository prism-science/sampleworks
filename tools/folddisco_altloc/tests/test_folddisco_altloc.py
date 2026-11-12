"""Offline tests: nothing here contacts the FoldDisco server."""

from pathlib import Path

import pytest
from folddisco_altloc.loop_motif import (
    choose_altlocs,
    find_altloc_segments,
    find_altlocs,
    MotifError,
    ResidueId,
    select_altloc,
)
from folddisco_altloc.query import expand_motif, MotifQuery, substituted_motif
from folddisco_altloc.server import AppState, handle_upload
from folddisco_altloc.summary import count_conformer, HitFilter, summarize, totals


def atom(
    serial: int,
    name: str,
    altloc: str,
    resname: str,
    chain: str,
    number: int,
    occupancy: float = 1.0,
    x: float = 0.0,
    record: str = "ATOM  ",
) -> str:
    """One fixed-column PDB atom record."""
    return (
        f"{record}{serial:5d} {name:<4s}{altloc}{resname:>3s} {chain}{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{20.0:6.2f}          {name[0]:>2s}\n"
    )


@pytest.fixture
def pdb_lines() -> list[str]:
    """Chain A 1-5: residues 2-3 have altlocs A/B, residue 4 only A, plus a water."""
    lines = [atom(1, "CA", " ", "GLY", "A", 1)]
    lines += [
        atom(2, "CA", "A", "ALA", "A", 2, 0.4, 1.0),
        atom(3, "CA", "B", "ALA", "A", 2, 0.6, 2.0),
    ]
    lines += [
        atom(4, "CA", "A", "SER", "A", 3, 0.4, 3.0),
        atom(5, "CA", "B", "SER", "A", 3, 0.6, 4.0),
    ]
    lines += [atom(6, "CA", "A", "LEU", "A", 4, 0.5, 5.0)]
    lines += [atom(7, "CA", " ", "ASP", "A", 5)]
    lines += [atom(8, "O", "A", "HOH", "S", 100, 0.5, record="HETATM")]
    return lines


def test_segments_join_consecutive_altloc_residues_and_skip_water(pdb_lines):
    segments = find_altloc_segments(pdb_lines)
    assert [(s.chain, s.start, s.end, s.altlocs) for s in segments] == [("A", 2, 4, ("A", "B"))]
    assert segments[0].incomplete == (ResidueId("A", 4),)
    assert segments[0].residues == "2-4"


def test_select_altloc_keeps_one_conformer_and_leaves_other_lines(pdb_lines):
    residues = {ResidueId("A", 2), ResidueId("A", 3)}
    selected = select_altloc(pdb_lines, residues, "B")
    assert len(selected) == len(pdb_lines) - 2
    loop = [line for line in selected if int(line[22:26]) in (2, 3)]
    assert all(line[16] == " " and line[54:60] == "  1.00" for line in loop)
    assert [float(line[30:38]) for line in loop] == [2.0, 4.0]
    assert [line for line in selected if int(line[22:26]) not in (2, 3)] == [
        line for line in pdb_lines if int(line[22:26]) not in (2, 3)
    ]


def test_choose_altlocs_rejects_residue_missing_a_conformer(pdb_lines):
    residues = {ResidueId("A", n) for n in (2, 3, 4)}
    with pytest.raises(MotifError, match="A4"):
        choose_altlocs(find_altlocs(pdb_lines, residues), None)
    assert choose_altlocs(find_altlocs(pdb_lines, residues - {ResidueId("A", 4)}), None) == [
        "A",
        "B",
    ]


@pytest.mark.parametrize(
    ("motif", "expected"),
    [
        ("A181", [("A", "A", 181, None)]),
        ("B_57,AA250", [("B", "B", 57, None), ("AA", "AA", 250, None)]),
        ("10_100", [("10_", "10", 100, None)]),
        ("A1-3:DE", [("A", "A", n, "DE") for n in (1, 2, 3)]),
    ],
)
def test_expand_motif_parses_chains_ranges_and_substitutions(motif, expected):
    assert expand_motif(motif) == expected


def test_group_substitution_uses_each_residue_type(tmp_path: Path, pdb_lines):
    structure = tmp_path / "q.pdb"
    structure.write_text("".join(pdb_lines))
    query = MotifQuery(structure, "A1,A2,A3:H,A5", None, 1)
    assert substituted_motif(query, "group") == "A1:GP,A2:AVLIMC,A3:H,A5:DE"


def _hit(target: str, nodes: int, rmsd: float, residues: str = "A1,A2,A3,A4") -> dict:
    return {
        "target": f"/db/{target}.ent",
        "nodecount": nodes,
        "rmsd": rmsd,
        "targetresidues": residues,
    }


def test_summary_counts_unique_targets_numberings_and_shared():
    results = {
        "A": {
            "results": [
                {
                    "db": "pdb",
                    "alignments": [
                        _hit("1abc", 4, 0.2),
                        _hit("1abc", 4, 0.3, "B1,B2,B3,B4"),
                        _hit("2xyz", 4, 0.4, "A9,A2,A3,A4"),
                        _hit("3pqr", 3, 0.1),
                        _hit("4lmn", 4, 0.6),
                    ],
                }
            ]
        },
        "B": {
            "results": [{"db": "pdb", "alignments": [_hit("1abc", 4, 0.1), _hit("5stu", 4, 0.45)]}]
        },
    }
    rows = summarize(results, HitFilter(motif_size=4, min_matched=4, rmsd_max=0.5))
    counts = rows[0].counts
    assert counts["A"].targets == {"1abc", "2xyz"} and counts["A"].numberings == 2
    assert counts["B"].targets == {"1abc", "5stu"}
    assert rows[0].shared == 1
    assert totals(rows) == {"A": 2, "B": 2, "shared": 1}


def test_confident_filter_and_truncation_flag():
    hits = [_hit(f"t{i}", 2, 0.1) for i in range(999)] + [_hit("last", 4, 0.2)]
    loose = count_conformer(hits, HitFilter(4, 2, 0.5))
    confident = count_conformer(hits, HitFilter(4, 2, 0.5, confident=True))
    assert len(loose.targets) == 1000 and len(confident.targets) == 1
    assert confident.last_rank == 999 and confident.possibly_truncated


def test_upload_lists_segments_and_rejects_mmcif(tmp_path: Path, pdb_lines):
    state = AppState(tmp_path)
    view = handle_upload(state, {"filename": "../evil name.pdb", "content": "".join(pdb_lines)})
    assert view["filename"] == "evil_name.pdb"
    assert (tmp_path / "uploads" / view["upload_id"] / "evil_name.pdb").exists()
    assert view["segments"][0]["residues"] == "2-4" and view["segments"][0]["incomplete"] == ["A4"]
    with pytest.raises(MotifError, match="PDB-format"):
        handle_upload(state, {"filename": "x.cif", "content": "data_x"})
