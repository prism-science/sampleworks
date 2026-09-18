"""Tests for guidance script argument handling."""

from __future__ import annotations

import pickle
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
from sampleworks.utils.guidance_constants import GuidanceType, StructurePredictor
from sampleworks.utils.guidance_script_arguments import (
    _remap_container_path,
    get_checkpoint,
    GuidanceConfig,
    JobConfig,
    JobResult,
    validate_model_checkpoint,
)


# ============================================================================
# get_checkpoint tests
# ============================================================================


def test_get_checkpoint_reads_model_checkpoint():
    """get_checkpoint should return the model_checkpoint value from the namespace."""
    args = Namespace(model_checkpoint="/tmp/model.ckpt")

    assert get_checkpoint(args) == "/tmp/model.ckpt"


def test_get_checkpoint_returns_none_when_missing():
    """get_checkpoint should return None when model_checkpoint is absent."""
    args = Namespace()

    assert get_checkpoint(args) is None


def test_get_checkpoint_treats_empty_string_as_missing():
    """Empty or whitespace-only model_checkpoint should be treated as missing."""
    args = Namespace(model_checkpoint="   ")

    assert get_checkpoint(args) is None


# ============================================================================
# populate_config_for_guidance_type tests
# ============================================================================


def _build_job(model: StructurePredictor) -> JobConfig:
    return JobConfig(
        protein="protein",
        structure_path="/tmp/structure.cif",
        density_path="/tmp/density.mrc",
        resolution=2.0,
        model_name=model,
        scaler=GuidanceType.PURE_GUIDANCE,
        ensemble_size=1,
        gradient_weight=0.1,
        gd_steps=1,
        method=None,
        output_dir="/tmp/output",
        log_path="/tmp/output/run.log",
    )


@patch(
    "sampleworks.utils.guidance_script_arguments._resolve_checkpoint",
    return_value="/checkpoints/mock.ckpt",
)
def test_populate_config_resolves_checkpoint_when_none_provided(_mock_resolve, model_wrapper_type):
    """populate_config_for_guidance_type should auto-resolve checkpoint if no arg exists."""
    config = GuidanceConfig(
        protein="protein",
        structure="/tmp/structure.cif",
        density="/tmp/density.mrc",
        model_name=model_wrapper_type,
        guidance_type=GuidanceType.PURE_GUIDANCE,
        log_path="/tmp/output/run.log",
    )

    config.populate_config_for_guidance_type(
        _build_job(model_wrapper_type),
        Namespace(use_tweedie=False, step_scaler_type="noisespace"),
    )

    assert config.model_checkpoint == "/checkpoints/mock.ckpt"


def test_populate_config_uses_model_checkpoint_argument(model_wrapper_type):
    """populate_config_for_guidance_type should read the model_checkpoint arg."""
    with patch(
        "sampleworks.utils.guidance_script_arguments._resolve_checkpoint",
        return_value="/checkpoints/mock.ckpt",
    ) as mock_resolve:
        config = GuidanceConfig(
            protein="protein",
            structure="/tmp/structure.cif",
            density="/tmp/density.mrc",
            model_name=model_wrapper_type,
            guidance_type=GuidanceType.PURE_GUIDANCE,
            log_path="/tmp/output/run.log",
        )
        mock_resolve.reset_mock()

        args = Namespace(
            model_checkpoint="/tmp/custom.ckpt",
            use_tweedie=False,
            step_scaler_type="noisespace",
        )
        config.populate_config_for_guidance_type(_build_job(model_wrapper_type), args)
        mock_resolve.assert_not_called()

    assert config.model_checkpoint == "/tmp/custom.ckpt"


# ============================================================================
# validate_model_checkpoint tests
# ============================================================================


@patch(
    "sampleworks.utils.guidance_script_arguments._resolve_checkpoint",
    side_effect=ValueError(
        "Running guidance requires a model checkpoint for 'model'. "
        "Provide --model-checkpoint or bake checkpoints into /checkpoints/."
    ),
)
def test_validate_model_checkpoint_requires_non_empty_value(_mock_resolve, model_wrapper_type):
    """Validation should fail fast when checkpoint is missing and can't be auto-resolved."""
    with pytest.raises(ValueError, match="requires a model checkpoint"):
        validate_model_checkpoint(model_wrapper_type, "")


def test_validate_model_checkpoint_requires_existing_file(model_wrapper_type, tmp_path: Path):
    """Validation should fail for missing files."""
    missing = tmp_path / "does_not_exist.ckpt"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_model_checkpoint(model_wrapper_type, str(missing))


def test_validate_model_checkpoint_rejects_directories(model_wrapper_type, tmp_path: Path):
    """Validation should reject directory paths."""
    with pytest.raises(ValueError, match="must be a file"):
        validate_model_checkpoint(model_wrapper_type, str(tmp_path))


def test_validate_model_checkpoint_returns_resolved_path(model_wrapper_type, tmp_path: Path):
    """Validation should return the resolved absolute checkpoint path."""
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("weights")

    validated = validate_model_checkpoint(model_wrapper_type, str(checkpoint))

    assert validated == str(checkpoint.resolve())


# ============================================================================
# _remap_container_path tests
# ============================================================================


def test_remap_noop_when_no_env_vars(monkeypatch):
    for var in (
        "SAMPLEWORKS_HOST_INPUT_DIR",
        "SAMPLEWORKS_HOST_RESULTS_DIR",
        "SAMPLEWORKS_HOST_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    assert _remap_container_path("/data/inputs/structure.cif") == "/data/inputs/structure.cif"
    assert _remap_container_path("/data/results/output") == "/data/results/output"


def test_remap_data_dir_env(monkeypatch):
    monkeypatch.delenv("SAMPLEWORKS_HOST_RESULTS_DIR", raising=False)
    monkeypatch.delenv("SAMPLEWORKS_HOST_DIR", raising=False)

    monkeypatch.setenv("SAMPLEWORKS_HOST_INPUT_DIR", "/mnt/nfs/proteins")
    assert _remap_container_path("/data/inputs/1abc.cif") == "/mnt/nfs/proteins/1abc.cif"
    # results path should pass through (no env var for it)
    assert _remap_container_path("/data/results/run1") == "/data/results/run1"


def test_remap_results_dir_env(monkeypatch):
    monkeypatch.delenv("SAMPLEWORKS_HOST_INPUT_DIR", raising=False)
    monkeypatch.delenv("SAMPLEWORKS_HOST_DIR", raising=False)

    monkeypatch.setenv("SAMPLEWORKS_HOST_RESULTS_DIR", "/results/exp1")
    result = _remap_container_path("/data/results/protein/boltz2/run.log")
    assert result == "/results/exp1/protein/boltz2/run.log"
    # inputs path should pass through
    assert _remap_container_path("/data/inputs/1abc.cif") == "/data/inputs/1abc.cif"


def test_remap_catchall_dir_env(monkeypatch):
    monkeypatch.delenv("SAMPLEWORKS_HOST_INPUT_DIR", raising=False)
    monkeypatch.delenv("SAMPLEWORKS_HOST_RESULTS_DIR", raising=False)

    monkeypatch.setenv("SAMPLEWORKS_HOST_DIR", "/mnt/storage")
    assert _remap_container_path("/data/inputs/1abc.cif") == "/mnt/storage/inputs/1abc.cif"
    assert _remap_container_path("/data/results/output") == "/mnt/storage/results/output"


def test_remap_specific_env_takes_priority_over_catchall(monkeypatch):
    monkeypatch.delenv("SAMPLEWORKS_HOST_RESULTS_DIR", raising=False)
    monkeypatch.setenv("SAMPLEWORKS_HOST_INPUT_DIR", "/specific/data")
    monkeypatch.setenv("SAMPLEWORKS_HOST_DIR", "/general")
    assert _remap_container_path("/data/inputs/foo.cif") == "/specific/data/foo.cif"
    # /data/results has no specific env var, so catch-all applies
    assert _remap_container_path("/data/results/bar") == "/general/results/bar"


def test_remap_leaves_checkpoint_unchanged(monkeypatch):
    monkeypatch.setenv("SAMPLEWORKS_HOST_INPUT_DIR", "/host/data")
    monkeypatch.setenv("SAMPLEWORKS_HOST_RESULTS_DIR", "/host/results")
    monkeypatch.delenv("SAMPLEWORKS_HOST_DIR", raising=False)

    assert _remap_container_path("/checkpoints/boltz2.ckpt") == "/checkpoints/boltz2.ckpt"


def test_remap_trailing_slash_normalization(monkeypatch):
    monkeypatch.delenv("SAMPLEWORKS_HOST_RESULTS_DIR", raising=False)
    monkeypatch.delenv("SAMPLEWORKS_HOST_DIR", raising=False)

    monkeypatch.setenv("SAMPLEWORKS_HOST_INPUT_DIR", "/mnt/data/")
    assert _remap_container_path("/data/inputs/foo.cif") == "/mnt/data/foo.cif"


def test_remap_exact_prefix_match(monkeypatch):
    monkeypatch.delenv("SAMPLEWORKS_HOST_RESULTS_DIR", raising=False)
    monkeypatch.delenv("SAMPLEWORKS_HOST_DIR", raising=False)

    monkeypatch.setenv("SAMPLEWORKS_HOST_INPUT_DIR", "/mnt/data")
    assert _remap_container_path("/data/inputs") == "/mnt/data"


def test_remap_ignores_empty_env_var(monkeypatch):
    monkeypatch.delenv("SAMPLEWORKS_HOST_RESULTS_DIR", raising=False)
    monkeypatch.delenv("SAMPLEWORKS_HOST_DIR", raising=False)
    monkeypatch.setenv("SAMPLEWORKS_HOST_INPUT_DIR", "")
    assert _remap_container_path("/data/inputs/foo.cif") == "/data/inputs/foo.cif"


def test_remap_whitespace_padded_env_var(monkeypatch):
    monkeypatch.delenv("SAMPLEWORKS_HOST_RESULTS_DIR", raising=False)
    monkeypatch.delenv("SAMPLEWORKS_HOST_DIR", raising=False)
    monkeypatch.setenv("SAMPLEWORKS_HOST_INPUT_DIR", " /mnt/data ")
    assert _remap_container_path("/data/inputs/foo.cif") == "/mnt/data/foo.cif"


def test_remap_root_env_var(monkeypatch):
    monkeypatch.delenv("SAMPLEWORKS_HOST_RESULTS_DIR", raising=False)
    monkeypatch.delenv("SAMPLEWORKS_HOST_DIR", raising=False)
    monkeypatch.setenv("SAMPLEWORKS_HOST_INPUT_DIR", "/")
    assert _remap_container_path("/data/inputs/foo.cif") == "/foo.cif"
    assert _remap_container_path("/data/inputs") == "/"


# ============================================================================
# as_dict path remapping tests
# ============================================================================


def test_as_dict_remaps_all_four_path_fields(monkeypatch):
    monkeypatch.setenv("SAMPLEWORKS_HOST_INPUT_DIR", "/host/data")
    monkeypatch.setenv("SAMPLEWORKS_HOST_RESULTS_DIR", "/host/results")
    monkeypatch.delenv("SAMPLEWORKS_HOST_DIR", raising=False)

    config = GuidanceConfig(
        protein="1abc",
        structure="/data/inputs/structures/1abc.cif",
        density="/data/inputs/maps/1abc.ccp4",
        model_name=StructurePredictor.BOLTZ_2,
        guidance_type=GuidanceType.PURE_GUIDANCE,
        log_path="/data/results/1abc/boltz2/run.log",
        output_dir="/data/results/1abc/boltz2",
    )
    d = config.as_dict()
    assert d["structure"] == "/host/data/structures/1abc.cif"
    assert d["density"] == "/host/data/maps/1abc.ccp4"
    assert d["output_dir"] == "/host/results/1abc/boltz2"
    assert d["log_path"] == "/host/results/1abc/boltz2/run.log"
    # Non-path fields unchanged
    assert d["protein"] == "1abc"


def test_as_dict_unchanged_when_no_env_vars(monkeypatch):
    for var in (
        "SAMPLEWORKS_HOST_INPUT_DIR",
        "SAMPLEWORKS_HOST_RESULTS_DIR",
        "SAMPLEWORKS_HOST_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    config = GuidanceConfig(
        protein="1abc",
        structure="/data/inputs/structures/1abc.cif",
        density="/data/inputs/maps/1abc.ccp4",
        model_name=StructurePredictor.BOLTZ_2,
        guidance_type=GuidanceType.PURE_GUIDANCE,
        log_path="/data/results/1abc/run.log",
        output_dir="/data/results/1abc",
    )
    d = config.as_dict()
    assert d["structure"] == "/data/inputs/structures/1abc.cif"
    assert d["density"] == "/data/inputs/maps/1abc.ccp4"
    assert d["output_dir"] == "/data/results/1abc"
    assert d["log_path"] == "/data/results/1abc/run.log"


def test_as_dict_remaps_with_catchall_host_dir(monkeypatch):
    monkeypatch.delenv("SAMPLEWORKS_HOST_INPUT_DIR", raising=False)
    monkeypatch.delenv("SAMPLEWORKS_HOST_RESULTS_DIR", raising=False)
    monkeypatch.setenv("SAMPLEWORKS_HOST_DIR", "/mnt/storage")
    config = GuidanceConfig(
        protein="1abc",
        structure="/data/inputs/structures/1abc.cif",
        density="/data/inputs/maps/1abc.ccp4",
        model_name=StructurePredictor.BOLTZ_2,
        guidance_type=GuidanceType.PURE_GUIDANCE,
        log_path="/data/results/1abc/run.log",
        output_dir="/data/results/1abc",
    )
    d = config.as_dict()
    assert d["structure"] == "/mnt/storage/inputs/structures/1abc.cif"
    assert d["density"] == "/mnt/storage/inputs/maps/1abc.ccp4"
    assert d["output_dir"] == "/mnt/storage/results/1abc"
    assert d["log_path"] == "/mnt/storage/results/1abc/run.log"
    assert d["protein"] == "1abc"


def test_guidance_config_migrates_legacy_model_pickle() -> None:
    """Old job queues restore ``model`` state as ``model_name``."""
    config = GuidanceConfig(
        protein="1abc",
        structure="structure.cif",
        density="density.ccp4",
        model_name=StructurePredictor.BOLTZ_2,
        guidance_type=GuidanceType.PURE_GUIDANCE,
        log_path="run.log",
    )
    config.__dict__["model"] = config.__dict__.pop("model_name")

    restored = pickle.loads(pickle.dumps(config))

    assert restored.model_name == StructurePredictor.BOLTZ_2
    assert "model" not in restored.__dict__
    assert "model" not in restored.as_dict()


def test_job_result_migrates_legacy_model_pickle() -> None:
    """Old result pickles restore ``model`` state as ``model_name``."""
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
    result.__dict__["model"] = result.__dict__.pop("model_name")

    restored = pickle.loads(pickle.dumps(result))

    assert restored.model_name == "boltz2"
    assert "model" not in restored.__dict__
    assert "model" not in restored.as_dict()


# ============================================================================
# latent-opt CLI flag plumbing (regression: flags must reach the config)
# ============================================================================


def test_latent_opt_cli_flags_reach_the_config():
    """Every --guidance-type latent_opt flag must land on the GuidanceConfig.

    Regression guard: these attribute names have to be listed in
    ``_DYNAMIC_ATTRS`` or ``from_cli`` silently drops the parsed values and
    ``_run_guidance``'s ``getattr(args, ...)`` falls back to the defaults, making
    every flag a no-op. Passing *non-default* values here catches that.
    """
    config = GuidanceConfig.from_cli(
        argv=[
            "--protein",
            "1vme",
            "--structure",
            "s.cif",
            "--density",
            "d.ccp4",
            "--resolution",
            "1.8",
            "--output-dir",
            "/tmp/out",
            # IT-opt flags, all set to values that differ from their defaults
            "--which-latent",
            "single",  # default "pair"
            "--learning-rate",
            "0.123",  # default 0.05
            "--outer-steps",
            "5",  # default 2
            "--anchor-weight",
            "0.7",  # default 0.0
            "--max-grad-norm",
            "2.0",  # default 1.0
            "--bond-length-weight",
            "0.001",  # default 5e-5
        ],
        model_name="protenix",
        guidance_type="latent_opt",
    )

    assert config.which_latent == "single"
    assert config.learning_rate == pytest.approx(0.123)
    assert config.outer_steps == 5
    assert config.anchor_weight == pytest.approx(0.7)
    assert config.max_grad_norm == pytest.approx(2.0)
    assert config.bond_length_weight == pytest.approx(0.001)
