"""Shared test fixtures for preset-runner tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sampleworks.runs import runner


@pytest.fixture(autouse=True)
def force_pixi_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep argv assertions deterministic across dev and ACTL image environments."""
    monkeypatch.delenv("SAMPLEWORKS_GRID_SEARCH_SCRIPT", raising=False)
    monkeypatch.delenv("SAMPLEWORKS_PIXI_PROJECT_DIR", raising=False)
    monkeypatch.setattr(
        runner,
        "WORKSPACE_GRID_SEARCH_SCRIPT",
        str(tmp_path / "missing-workspace" / "run_grid_search.py"),
    )
    for var in (
        "RUNTIME_PIXI",
        "SAMPLEWORKS_ALLOW_RUNTIME_PIXI",
        "SAMPLEWORKS_REQUIRE_PREBUILT_PIXI",
        "SAMPLEWORKS_SKIP_ENV_PREPARE",
    ):
        monkeypatch.delenv(var, raising=False)
    for var in list(os.environ):
        if var.startswith("SAMPLEWORKS_") and var.endswith("_PYTHON"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SAMPLEWORKS_FORCE_PIXI", "1")


# A tuple, not a set: this parametrizes tests, and set iteration order varies
# with PYTHONHASHSEED, which would make test IDs differ between runs.
BUNDLED = (
    "boltz",
    "boltz1",
    "boltz2",
    "boltz2_md",
    "boltz2_xrd",
    "full_8gpu",
    "it_opt_1gpu",
    "it_opt_4gpu",
    "it_opt_8gpu",
    "protenix",
    "protenix_dual",
    "protpardelle",
    "rf3",
    "rf3_partial",
    "rf3_partial_chiral_off",
    "rf3_protenix",
)
