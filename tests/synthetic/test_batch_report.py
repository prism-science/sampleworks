"""The batch report: which rows produced files, without reading the logs.

``process_batch`` runs rows independently and swallows per-row errors so one bad
structure does not abort the batch. That makes the exit status useless as a
signal -- a run where every row failed still exits 0 -- so the report is the only
place a caller can find out what happened.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")

from sampleworks.synthetic.generate_synthetic_sf_lunus import process_batch


RESOLUTION = 4.0  # coarse: these tests are about bookkeeping, not accuracy


@pytest.fixture
def batch(resources_dir: Path, tmp_path: Path):
    """A two-row batch: one real structure, one file that cannot be read.

    Returns ``(csv_path, base_dir, output_dir)``.
    """
    source = resources_dir / "1vme" / "1vme_final.cif"
    if not source.exists():
        pytest.skip(f"Source structure not found at {source}")

    base_dir = tmp_path / "structures"
    base_dir.mkdir()
    (base_dir / "good.cif").write_bytes(source.read_bytes())
    (base_dir / "broken.cif").write_text("this is not a structure file")

    csv_path = tmp_path / "batch.csv"
    csv_path.write_text("filename,selection\ngood.cif,chain A\nbroken.cif,\n")

    return csv_path, base_dir, tmp_path / "out"


def _run(batch, **kwargs) -> dict:
    csv_path, base_dir, output_dir = batch
    return process_batch(
        csv_path=csv_path,
        base_dir=base_dir,
        output_dir=output_dir,
        resolution=RESOLUTION,
        occupancy_mode="default",
        test_fraction=0.05,
        seed=0,
        device=torch.device("cpu"),
        n_jobs=1,
        strip_hydrogens=True,
        strip_waters=True,
        **kwargs,
    )


def test_report_is_written_to_the_output_directory(batch):
    _, _, output_dir = batch

    _run(batch)

    report_path = output_dir / "batch_report.json"
    assert report_path.exists()
    json.loads(report_path.read_text())  # must be valid JSON


def test_report_summarises_a_mixed_batch(batch):
    """The summary is the whole point: one line answering "did this batch work?"."""
    report = _run(batch)

    assert report["total"] == 2
    assert report["succeeded"] == 1
    assert report["failed"] == 1
    assert report["succeeded"] + report["failed"] == report["total"]


def test_report_holds_one_record_per_row_naming_both_outcomes(batch):
    report = _run(batch)

    by_name = {r["filename"]: r for r in report["rows"]}
    assert set(by_name) == {"good.cif", "broken.cif"}
    assert by_name["good.cif"]["status"] == "success"
    assert by_name["broken.cif"]["status"] == "failed"


def test_a_failed_row_does_not_suppress_a_successful_one(batch):
    """The reason rows are isolated at all: one bad structure must not lose the rest."""
    report = _run(batch)

    good = next(r for r in report["rows"] if r["filename"] == "good.cif")
    assert Path(good["output_path"]).exists(), "the good row's MTZ should be on disk"


def test_a_failed_row_records_the_stage_and_the_error(batch):
    """Enough to know where to look; the traceback stays in the log."""
    report = _run(batch)

    broken = next(r for r in report["rows"] if r["filename"] == "broken.cif")
    assert broken["failure_stage"] == "load"
    assert broken["error"], "the error string should not be empty"
    assert "output_path" not in broken


def test_the_written_report_matches_the_returned_one(batch):
    _, _, output_dir = batch

    returned = _run(batch)

    assert json.loads((output_dir / "batch_report.json").read_text()) == returned


def test_single_configuration_diffuse_rows_are_reported_as_validation_failures(batch):
    """--write-diffuse on a single-model structure is refused before any compute.

    It is a misconfiguration rather than a crash, so it must still show up in the
    report rather than passing silently.
    """
    report = _run(batch, write_diffuse=True)

    good = next(r for r in report["rows"] if r["filename"] == "good.cif")
    assert good["status"] == "failed"
    assert good["failure_stage"] == "validate"
    assert "output_path" not in good, "nothing should be written for a refused row"
