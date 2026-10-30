"""Unit tests for sampleworks.runs.loader."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from sampleworks.runs import loader

from tests.runs.conftest import BUNDLED


def _get_model_env_prefixes() -> tuple[str, ...]:
    """Extract model environment prefixes from pyproject.toml.

    Note that this is NOT the same as sampleworks.runs.schema._load_valid_pixi_envs(),
    this method only pulls the prefixes, which are the environments used for running guidance.

    Returns unique base environment names (e.g., "boltz", "rf3") excluding
    "analysis" and dev/platform-specific variants.
    """
    current = Path(__file__).parent
    while current != current.parent:
        pyproject_path = current / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
            envs = data.get("tool", {}).get("pixi", {}).get("environments", {})

            prefixes = {
                env_name.split("-")[0]
                for env_name in envs.keys()
                if env_name.split("-")[0] != "analysis"
            }

            if not prefixes:
                raise ValueError("No valid model environment prefixes found in pyproject.toml")

            return tuple(sorted(prefixes))

        current = current.parent
    raise FileNotFoundError("Could not find pyproject.toml while searching from test_loader.py")


MODEL_ENV_PREFIXES = _get_model_env_prefixes()


def test_list_presets_returns_bundled_experiments() -> None:
    """Preset discovery returns the expected bundled experiment names."""
    names = loader.list_presets()
    assert set(names) == set(BUNDLED), f"unexpected experiment presets: {names}"


@pytest.mark.parametrize("name", BUNDLED)
def test_each_experiment_preset_loads(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every bundled preset loads into jobs with supported pixi environments."""
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset(name)
    assert preset.name == name
    assert preset.jobs, f"{name} has no jobs"
    for job in preset.jobs:
        assert job.env in MODEL_ENV_PREFIXES


def test_env_var_wins_over_defaults_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables override preset defaults during interpolation."""
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("DATA_DIR", "/from/env")
    preset = loader.load_preset("rf3_partial")
    assert preset.defaults["DATA_DIR"] == "/from/env"
    # PROTEINS_CSV expands to ${DATA_DIR}/proteins.csv; DATA_DIR overridden by env
    proteins = preset.shared_args["proteins"]
    assert proteins == "/from/env/proteins.csv"


def test_defaults_used_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preset defaults fill in interpolation variables absent from the environment."""
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("rf3_partial")
    assert preset.defaults["DATA_DIR"] == "/data/inputs"


def test_full_8gpu_uses_canonical_inputs_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flagship preset must use /data/inputs, matching the ACTL wrapper."""
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("full_8gpu")
    assert preset.defaults["DATA_DIR"] == "/data/inputs"
    assert preset.shared_args["proteins"] == "/data/inputs/proteins.csv"


def test_set_override_at_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--set defaults.*`` overrides participate in later interpolation."""
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("rf3_partial", overrides=["defaults.DATA_DIR=/custom"])
    assert preset.defaults["DATA_DIR"] == "/custom"
    assert preset.shared_args["proteins"] == "/custom/proteins.csv"


def test_set_override_at_job_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--set jobs.<name>.*`` updates the named job."""
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("full_8gpu", overrides=["jobs.rf3.gpus=7"])
    assert preset.job("rf3").gpus == "7"
    assert preset.job("rf3").gpu_count is None


def test_set_override_at_job_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--set jobs.<index>.*`` updates the indexed job."""
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("full_8gpu", overrides=["jobs.0.gpus=9"])
    assert preset.jobs[0].gpus == "9"
    assert preset.jobs[0].gpu_count is None


def test_set_override_gpu_count_clears_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--set jobs.<name>.gpu_count`` replaces an explicit GPU assignment."""
    monkeypatch.setenv("HOME", "/home/test")
    custom = tmp_path / "gpu_count.toml"
    custom.write_text(
        'description = "custom"\n'
        "[[jobs]]\n"
        'name = "j1"\n'
        'env = "rf3"\n'
        'gpus = "0"\n'
        'output_subdir = "j1"\n'
        "args = {}\n"
    )
    preset = loader.load_preset(str(custom), overrides=["jobs.j1.gpu_count=2"])
    assert preset.job("j1").gpu_count == 2
    assert preset.job("j1").gpus == ""


def test_set_override_at_args_inside_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dotted overrides can create or replace per-job CLI args."""
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset(
        "rf3_partial", overrides=["jobs.rf3.args.gradient-weights=0.0 0.01"]
    )
    assert preset.job("rf3").args["gradient-weights"] == "0.0 0.01"


def test_set_coerces_bool_and_float(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override values are coerced to bools and ints when unambiguous."""
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset(
        "rf3_partial",
        overrides=[
            "shared_args.gradient-normalization=false",
            "jobs.rf3.args.partial-diffusion-start=0.4",
        ],
    )
    assert preset.shared_args["gradient-normalization"] is False
    # job.args["partial-diffusion-start"] exists by default in rf3_partial, set to 0.6
    # but it should be overridden above
    assert preset.job("rf3").args["num-diffusion-steps"] == 0.4


def test_load_preset_from_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A filesystem TOML path loads as a custom preset."""
    monkeypatch.setenv("HOME", "/home/test")
    custom = tmp_path / "mycustom.toml"
    custom.write_text(
        'description = "custom"\n'
        "[defaults]\n"
        'DATA_DIR = "/x"\n'
        "[shared_args]\n"
        'model = "rf3"\n'
        "[[jobs]]\n"
        'name = "j1"\n'
        'env = "rf3"\n'
        'gpus = "0"\n'
        'output_subdir = "j1"\n'
        "args = {}\n"
    )
    preset = loader.load_preset(str(custom))
    assert preset.name == "mycustom"
    assert preset.defaults["DATA_DIR"] == "/x"


def test_load_preset_from_experiments_dir_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scientists can point the loader at a top-level experiments directory."""
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    (experiments_dir / "custom.toml").write_text(
        'description = "custom"\n'
        "[shared_args]\n"
        'model = "rf3"\n'
        "[[jobs]]\n"
        'name = "j1"\n'
        'env = "rf3"\n'
        'gpus = "0"\n'
        'output_subdir = "j1"\n'
        "args = {}\n"
    )
    monkeypatch.setenv("SAMPLEWORKS_EXPERIMENTS_DIR", str(experiments_dir))

    preset = loader.load_preset("custom")

    assert preset.name == "custom"
    assert preset.job("j1").env == "rf3"


def test_unknown_preset_raises() -> None:
    """Missing preset names raise ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        loader.load_preset("does_not_exist")


def test_undefined_variable_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unresolved ``${VAR}`` references fail instead of expanding to empty strings."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[shared_args]\nproteins = "${NEVER_DEFINED_VAR}/x"\n'
        '[[jobs]]\nname = "j"\nenv = "rf3"\ngpus = "0"\noutput_subdir = "j"\nargs = {}\n'
    )
    monkeypatch.delenv("NEVER_DEFINED_VAR", raising=False)
    with pytest.raises(KeyError, match="NEVER_DEFINED_VAR"):
        loader.load_preset(str(bad))


def test_set_without_equals_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed override specs must contain a ``KEY=VALUE`` separator."""
    monkeypatch.setenv("HOME", "/home/test")
    with pytest.raises(ValueError, match="KEY=VALUE"):
        loader.load_preset("rf3_partial", overrides=["bogus_no_equals"])


def test_set_with_unknown_top_level_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typos like ``--set job.rf3.gpus=0`` (missing 's' in jobs) must not silently no-op."""
    monkeypatch.setenv("HOME", "/home/test")
    with pytest.raises(KeyError, match="unknown top-level key"):
        loader.load_preset("rf3_partial", overrides=["job.rf3.gpus=0"])


def test_set_with_out_of_range_job_index_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-range list indices in overrides fail with a clear ``KeyError``."""
    monkeypatch.setenv("HOME", "/home/test")
    with pytest.raises(KeyError, match="index 99"):
        loader.load_preset("rf3_partial", overrides=["jobs.99.gpus=0"])


def test_cyclic_variable_expansion_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cyclic ``${VAR}`` references fail fast instead of looping forever."""
    bad = tmp_path / "cycle.toml"
    bad.write_text(
        "[shared_args]\n"
        'proteins = "${A}"\n'
        '[[jobs]]\nname = "j"\nenv = "rf3"\ngpus = "0"\noutput_subdir = "j"\nargs = {}\n'
    )
    monkeypatch.setenv("A", "${B}")
    monkeypatch.setenv("B", "${A}")
    with pytest.raises(ValueError, match="did not converge"):
        loader.load_preset(str(bad))


def test_bad_env_rejected(tmp_path: Path) -> None:
    """Preset jobs reject unsupported pixi environment names."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[[jobs]]\nname = "j"\nenv = "not_a_real_env"\ngpus = "0"\noutput_subdir = "j"\nargs = {}\n'
    )
    with pytest.raises(ValueError, match="env must be one of"):
        loader.load_preset(str(bad))
