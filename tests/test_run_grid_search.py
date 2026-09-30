"""Tests for grid-search result serialization."""

import json
from argparse import Namespace
from pathlib import Path

import pytest
from run_grid_search import (
    build_args_for_process_pool,
    get_pixi_env,
    GridSearchConfig,
    sampler_dir_suffix,
    save_results,
)
from sampleworks.eval.grid_search_eval_utils import parse_trial_dir
from sampleworks.runs.schema import VALID_PIXI_ENVS
from sampleworks.utils.guidance_constants import StructurePredictor
from sampleworks.utils.guidance_script_arguments import JobConfig, JobResult


def test_save_results_normalizes_legacy_model_key(tmp_path) -> None:
    """Existing result records migrate to ``model_name`` without duplicates."""
    legacy_run = {
        "protein": "1abc",
        "model": "boltz2",
        "method": None,
        "scaler": "pure_guidance",
        "ensemble_size": 1,
        "gradient_weight": 0.1,
        "gd_steps": 1,
        "status": "failed",
    }
    (tmp_path / "results.json").write_text(json.dumps({"runs": [legacy_run]}))
    result = JobResult(
        protein="1abc",
        model_name="boltz2",
        method=None,
        scaler="pure_guidance",
        ensemble_size=1,
        gradient_weight=0.1,
        gd_steps=1,
        status="success",
        exit_code=0,
        runtime_seconds=1.0,
        started_at="2026-07-13T00:00:00",
        finished_at="2026-07-13T00:00:01",
        log_path="run.log",
        output_dir="output",
    )
    config = GridSearchConfig(
        model_name="boltz2",
        scalers=["pure_guidance"],
        ensemble_sizes=[1],
        gradient_weights=[0.1],
        gd_steps=[1],
        method="",
        proteins_file="proteins.csv",
        output_dir=str(tmp_path),
    )

    save_results([result], config, str(tmp_path), total_time=1.0)

    saved = json.loads((tmp_path / "results.json").read_text())
    assert len(saved["runs"]) == 1
    assert saved["runs"][0]["model_name"] == "boltz2"
    assert "model" not in saved["runs"][0]


def test_every_structure_predictor_has_a_valid_pixi_env() -> None:
    """Each supported model resolves to a pixi environment declared in pyproject.

    A model that reaches the grid search without a mapped environment only
    fails inside the worker subprocess, so this is checked up front.
    """
    for predictor in StructurePredictor:
        assert get_pixi_env(predictor) in VALID_PIXI_ENVS


def test_get_pixi_env_rejects_unknown_model() -> None:
    """An unrecognized model name fails with the valid options listed."""
    with pytest.raises(ValueError, match="Unknown model: not-a-model"):
        get_pixi_env("not-a-model")


LANGEVIN_ARGS = Namespace(
    sampler="langevin",
    sde_mode="reverse_sde",
    inverse_temperature=4.0,
    langevin_factor=4.0,
    temper_guidance=False,
    integrate_func="euler_maruyama",
)


def test_sampler_dir_suffix() -> None:
    """The default sampler keeps existing trial names; Langevin encodes its settings."""
    assert sampler_dir_suffix(Namespace(sampler="af3edm")) == ""
    assert sampler_dir_suffix(LANGEVIN_ARGS) == "_langevin_reverse_sde_b4_l4_untemp"
    heun = Namespace(**{**vars(LANGEVIN_ARGS), "temper_guidance": True, "integrate_func": "heun"})
    assert sampler_dir_suffix(heun) == "_langevin_reverse_sde_b4_l4_heun"


def test_langevin_suffix_keeps_trial_dir_parsing() -> None:
    """The suffix adds no token that parse_trial_dir reads as a grid value."""
    params = parse_trial_dir(Path(f"ens8_gw0.1{sampler_dir_suffix(LANGEVIN_ARGS)}"))
    assert params == {"ensemble_size": 8, "guidance_weight": 0.1, "gd_steps": None}


def _job_result(sampler: str) -> JobResult:
    return JobResult(
        protein="1abc",
        model_name="protenix",
        method=None,
        scaler="pure_guidance",
        ensemble_size=8,
        gradient_weight=0.1,
        gd_steps=1,
        status="success",
        exit_code=0,
        runtime_seconds=1.0,
        started_at="2026-09-30T00:00:00",
        finished_at="2026-09-30T00:00:01",
        log_path="run.log",
        output_dir=f"output_{sampler}",
        sampler=sampler,
    )


def test_save_results_keeps_runs_of_different_samplers(tmp_path) -> None:
    """Runs that differ only in sampler are separate records in results.json."""
    config = GridSearchConfig(
        model_name="protenix",
        scalers=["pure_guidance"],
        ensemble_sizes=[8],
        gradient_weights=[0.1],
        gd_steps=[1],
        method="",
        proteins_file="proteins.csv",
        output_dir=str(tmp_path),
    )
    save_results([_job_result("af3edm")], config, str(tmp_path), total_time=1.0)
    save_results([_job_result("langevin")], config, str(tmp_path), total_time=1.0)

    saved = json.loads((tmp_path / "results.json").read_text())
    assert sorted(run["sampler"] for run in saved["runs"]) == ["af3edm", "langevin"]


def test_build_args_for_process_pool_copies_langevin_options() -> None:
    """Grid-search Langevin options reach the worker's GuidanceConfig."""
    job = JobConfig(
        protein="1abc",
        structure_path="/tmp/structure.cif",
        density_path="/tmp/density.ccp4",
        resolution=1.0,
        model_name="protenix",
        scaler="pure_guidance",
        ensemble_size=8,
        gradient_weight=0.1,
        gd_steps=1,
        method=None,
        output_dir="/tmp/output",
        log_path="/tmp/output/run.log",
        sampler="langevin",
    )
    args = Namespace(
        **vars(LANGEVIN_ARGS),
        model_checkpoint="/tmp/custom.ckpt",
        step_scaler_type="dataspace",
        loss_order=2,
        partial_diffusion_step=300,
        guidance_start=-1,
        gradient_normalization=True,
        augmentation=True,
        align_to_input=True,
        recycling_steps=None,
        num_diffusion_steps=500,
    )

    config = build_args_for_process_pool(job, args)

    assert config.sampler == "langevin"
    assert config.inverse_temperature == 4.0
    assert config.langevin_factor == 4.0
    assert config.sde_mode == "reverse_sde"
    assert config.temper_guidance is False
    assert config.integrate_func == "euler_maruyama"
    assert config.num_diffusion_steps == 500
    assert config.as_dict()["sampler"] == "langevin"
