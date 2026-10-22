"""Tests for grid-search job generation and result serialization."""

import json
from pathlib import Path

import pytest
from run_grid_search import (
    build_args_for_process_pool,
    generate_jobs,
    get_pixi_env,
    GridSearchConfig,
    parse_args,
    save_results,
)
from sampleworks.runs.schema import VALID_PIXI_ENVS
from sampleworks.utils.guidance_constants import StructurePredictor
from sampleworks.utils.guidance_script_arguments import JobResult


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
        reward_config={"real_space_density": {}},
    )

    save_results([result], config, str(tmp_path), total_time=1.0)

    saved = json.loads((tmp_path / "results.json").read_text())
    assert len(saved["runs"]) == 1
    assert saved["runs"][0]["model_name"] == "boltz2"
    assert saved["runs"][0]["reward_type"] == "real_space_density"
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


@pytest.fixture
def grid_inputs(tmp_path: Path) -> Path:
    """A directory with one protein's structure, map, and MTZ."""
    for name in ("s.cif", "d.ccp4", "x.mtz"):
        (tmp_path / name).write_text("")
    return tmp_path


def _grid_args(inputs: Path, csv_text: str, *extra: str):
    csv_path = inputs / "proteins.csv"
    csv_path.write_text(csv_text)
    return parse_args(
        [
            "--proteins",
            str(csv_path),
            "--output-dir",
            str(inputs / "out"),
            "--scalers",
            "pure_guidance",
            "--ensemble-sizes",
            "4",
            "--gradient-weights",
            "0.1",
            "--model-checkpoint",
            str(inputs / "model.ckpt"),
            *extra,
        ]
    )


class TestRewardSelection:
    """Any registered reward can be swept, with its data coming from the proteins CSV."""

    def test_density_grids_keep_their_inputs_and_paths(self, grid_inputs):
        args = _grid_args(grid_inputs, "name,structure,density,resolution\n1ABC,s.cif,d.ccp4,1.8\n")

        (job,) = generate_jobs(args)

        options = job.reward_config["real_space_density"]["reward_options"]
        assert options["density"] == str(grid_inputs / "d.ccp4")
        assert options["resolution"] == 1.8
        assert Path(job.output_dir).name == "ens4_gw0.1"

    def test_loss_order_flag_still_reaches_the_density_reward(self, grid_inputs):
        args = _grid_args(
            grid_inputs,
            "name,structure,density,resolution\n1ABC,s.cif,d.ccp4,1.8\n",
            "--loss-order",
            "1",
        )

        (job,) = generate_jobs(args)
        config = build_args_for_process_pool(job, args)

        assert config.loss_order == 1
        assert config.density == str(grid_inputs / "d.ccp4")

    def test_structure_factor_grid_reads_the_mtz_from_the_csv(self, grid_inputs):
        args = _grid_args(
            grid_inputs,
            "name,structure,mtzfile\n1ABC,s.cif,x.mtz\n",
            "--reward-type",
            "structure_factor",
            "--bulk-solvent",
            "combined",
        )

        (job,) = generate_jobs(args)
        config = build_args_for_process_pool(job, args)

        options = config.resolved_reward_config().entries[0].options
        assert config.reward_type == "structure_factor"
        assert options["mtzfile"] == str(grid_inputs / "x.mtz")
        assert options["bulk_solvent"] == "combined"
        assert config.density is None
        assert Path(job.output_dir).name == "ens4_gw0.1_structure_factor"

    def test_one_csv_serves_density_and_structure_factor_sweeps(self, grid_inputs):
        csv_text = "name,structure,density,mtzfile,resolution\n1ABC,s.cif,d.ccp4,x.mtz,1.8\n"

        (density_job,) = generate_jobs(_grid_args(grid_inputs, csv_text))
        (sf_job,) = generate_jobs(
            _grid_args(grid_inputs, csv_text, "--reward-type", "structure_factor")
        )

        assert list(density_job.reward_config) == ["real_space_density"]
        assert list(sf_job.reward_config) == ["structure_factor"]
        assert density_job.output_dir != sf_job.output_dir

    def test_reward_config_file_combines_rewards(self, grid_inputs):
        (grid_inputs / "rewards.json").write_text(
            json.dumps(
                {
                    "real_space_density": {"weight": 0.5},
                    "structure_factor": {"weight": 0.5},
                }
            )
        )
        args = _grid_args(
            grid_inputs,
            "name,structure,density,mtzfile,resolution\n1ABC,s.cif,d.ccp4,x.mtz,1.8\n",
            "--reward-config",
            str(grid_inputs / "rewards.json"),
        )

        (job,) = generate_jobs(args)

        assert job.reward_config["structure_factor"]["reward_options"]["mtzfile"] == str(
            grid_inputs / "x.mtz"
        )
        assert Path(job.output_dir).name == "ens4_gw0.1_real_space_density+structure_factor"

    def test_a_missing_data_column_fails_before_any_job_is_queued(self, grid_inputs):
        args = _grid_args(
            grid_inputs,
            "name,structure,density,resolution\n1ABC,s.cif,d.ccp4,1.8\n",
            "--reward-type",
            "structure_factor",
        )

        with pytest.raises(ValueError, match="Protein '1ABC' has no value for mtzfile"):
            generate_jobs(args)

    def test_another_reward_s_flag_is_rejected(self, grid_inputs, capsys):
        with pytest.raises(SystemExit):
            _grid_args(
                grid_inputs,
                "name,structure,density,resolution\n1ABC,s.cif,d.ccp4,1.8\n",
                "--bulk-solvent",
                "combined",
            )

        assert "--bulk-solvent" in capsys.readouterr().err
