"""Tests for metadata-backed grid-search trial discovery."""

import json
from pathlib import Path

from sampleworks.eval.grid_search_eval_utils import load_job_metadata, scan_grid_search_results


def test_scan_grid_search_results_prefers_job_metadata(tmp_path) -> None:
    """Trial identity and input paths come from recorded job metadata."""
    trial_dir = tmp_path / "1ABC_0.5occA_0.5occB" / "wrong_model" / "wrong_scaler" / "ens1_gw0.1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "refined.cif").write_text("data_test")
    metadata = {
        "protein": "1ABC",
        "model_name": "boltz2",
        "method": "MD",
        "guidance_type": "pure_guidance",
        "ensemble_size": 8,
        "step_size": 0.25,
        "altloc_occupancies": {"A": 0.25, "B": 0.75},
        "structure": "/inputs/1abc.cif",
        "density": "/inputs/1abc.ccp4",
        "resolution": 1.8,
    }
    (trial_dir / "job_metadata.json").write_text(json.dumps(metadata))

    trials = scan_grid_search_results(trial_dir, current_depth=4, target_depth=4)

    assert len(trials) == 1
    trial = trials[0]
    assert trial.protein == "1ABC"
    assert trial.model == "boltz2"
    assert trial.method == "MD"
    assert trial.scaler == "pure_guidance"
    assert trial.ensemble_size == 8
    assert trial.guidance_weight == 0.25
    assert trial.altloc_occupancies == {"A": 0.25, "B": 0.75}
    assert trial.input_structure_path == Path("/inputs/1abc.cif")
    assert trial.density_path == Path("/inputs/1abc.ccp4")
    assert trial.resolution == 1.8


def test_load_job_metadata_rejects_non_object_json(tmp_path) -> None:
    """Metadata arrays are ignored rather than breaking trial discovery."""
    (tmp_path / "job_metadata.json").write_text("[]")

    assert load_job_metadata(tmp_path) is None


def test_scan_uses_path_fallback_for_empty_metadata_values(tmp_path) -> None:
    """Null or empty metadata fields do not replace usable path metadata."""
    trial_dir = tmp_path / "1ABC_1.0occA" / "boltz2_MD" / "pure_guidance" / "ens8_gw0.1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "refined.cif").write_text("data_test")
    (trial_dir / "job_metadata.json").write_text(
        json.dumps({"protein": None, "model": None, "method": "", "ensemble_size": "8.0"})
    )

    trials = scan_grid_search_results(trial_dir, current_depth=4, target_depth=4)

    assert len(trials) == 1
    assert trials[0].protein == "1abc"
    assert trials[0].model == "boltz2"
    assert trials[0].method == "MD"
    assert trials[0].ensemble_size == 8


def test_scan_reads_sampler_from_metadata(tmp_path) -> None:
    """The sampler comes from job metadata, and older runs without it default to af3edm."""
    for sampler, trial_name in (
        ("langevin", "ens8_gw0.1_langevin_reverse_sde_b4_l4"),
        (None, "ens8_gw0.1"),
    ):
        trial_dir = tmp_path / "1ABC_1.0occA" / "protenix" / "pure_guidance" / trial_name
        trial_dir.mkdir(parents=True)
        (trial_dir / "refined.cif").write_text("data_test")
        metadata = {"model_name": "protenix", "ensemble_size": 8, "step_size": 0.1}
        if sampler is not None:
            metadata["sampler"] = sampler
        (trial_dir / "job_metadata.json").write_text(json.dumps(metadata))

        trials = scan_grid_search_results(trial_dir, current_depth=4, target_depth=4)

        assert len(trials) == 1
        assert trials[0].sampler == (sampler or "af3edm")
        assert trials[0].ensemble_size == 8
        assert trials[0].guidance_weight == 0.1
