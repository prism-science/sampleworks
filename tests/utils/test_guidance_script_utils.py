"""Tests for guidance_script_utils saving helpers."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import sampleworks.utils.guidance_script_utils as guidance_script_utils
import torch
from sampleworks.core.rewards.protocol import PreparableRewardFunctionProtocol
from sampleworks.utils.guidance_script_arguments import GuidanceConfig, JobResult
from sampleworks.utils.guidance_script_utils import (
    _three_state_resolver,
    _write_job_metadata,
    get_diffuse_reward_and_structure,
    get_model_and_device,
    get_reward_function_and_structure,
    save_everything,
)

from tests.utils.atom_array_builders import build_test_atom_array


@pytest.mark.parametrize(
    "override, is_boltz, expected",
    [
        (None, True, True),  # Boltz default: enabled
        (None, False, False),  # other models default: disabled
        (True, False, True),  # explicit opt-in on a non-Boltz model
        (True, True, True),  # explicit on, agrees with Boltz default
        (False, True, False),  # explicit opt-out overrides the Boltz default
        (False, False, False),  # explicit off, agrees with non-Boltz default
    ],
)
def test_resolve_alignment_reverse_diffusion(override, is_boltz, expected):
    """The override wins when set, None means is_boltz default."""
    assert _three_state_resolver(override, is_boltz) is expected


def test_get_model_and_device_forwards_preloaded_model_to_rf3(monkeypatch):
    """RF3 construction receives the pre-loaded model supplied by callers."""
    preloaded_model = object()

    class StubRF3Wrapper:
        """Capture RF3 constructor arguments without loading model dependencies."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(guidance_script_utils, "RF3Wrapper", StubRF3Wrapper)
    monkeypatch.setattr(
        guidance_script_utils,
        "validate_model_checkpoint",
        lambda model_type, checkpoint: checkpoint,
    )
    monkeypatch.setattr(guidance_script_utils, "MSAManager", lambda: object())

    device, wrapper = get_model_and_device(
        "cpu",
        "rf3.ckpt",
        "rf3",
        model=preloaded_model,
    )

    assert device == torch.device("cpu")
    assert wrapper.kwargs["model"] is preloaded_model


def test_save_everything_uses_model_atom_array_for_mismatch(tmp_path: Path):
    """Mismatch final_state should save with model template when provided."""
    refined_structure = {"asym_unit": build_test_atom_array(n_atoms=3, with_occupancy=True)}
    model_atom_array = build_test_atom_array(n_atoms=5, with_occupancy=False)

    final_state = torch.zeros((1, 5, 3), dtype=torch.float32)

    args = GuidanceConfig(
        protein="1l63",
        structure=Path("dummy"),
        density=Path("dummy"),
        model_name="boltz2",
        guidance_type="pure_guidance",
        log_path="dummy",
        output_dir=str(tmp_path),
    )

    save_everything(
        args,
        losses=[],
        refined_structure=refined_structure,
        traj_denoised=[],
        traj_next_step=[],
        scaler_type="pure_guidance",
        final_state=final_state,
        model_atom_array=model_atom_array,
    )

    assert (tmp_path / "refined.cif").exists()


def test_get_reward_function_keeps_original_structure_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The input structure file must survive when altloc resolution leaves it unchanged.

    ``resolve_mixed_hetatm_atom_altlocs`` returns a ``Path`` even when it makes no
    copy, so the original path may compare unequal to the returned one (``Path(x)
    != str(x)``). The unlink branch must compare by string value, otherwise the
    caller's real input file is deleted.
    """
    structure_file = tmp_path / "input.cif"
    structure_file.write_text("dummy structure")

    # Altloc resolution makes no copy: it returns the same path (as a Path object).
    monkeypatch.setattr(
        "sampleworks.utils.guidance_script_utils.resolve_mixed_hetatm_atom_altlocs",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        "sampleworks.utils.guidance_script_utils.parse_structure",
        lambda *args, **kwargs: {
            "asym_unit": build_test_atom_array(n_atoms=3, with_occupancy=True)
        },
    )
    monkeypatch.setattr(
        "sampleworks.utils.guidance_script_utils.XMap",
        MagicMock(fromfile=MagicMock(return_value=MagicMock())),
    )
    monkeypatch.setattr(
        "sampleworks.utils.guidance_script_utils.setup_scattering_params",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "sampleworks.utils.guidance_script_utils.RealSpaceRewardFunction",
        MagicMock(return_value=MagicMock()),
    )

    # Pass the path as a string to exercise the str-vs-Path comparison.
    get_reward_function_and_structure(
        density="dummy_density.mrc",
        device=torch.device("cpu"),
        em=False,
        loss_order=2,
        resolution=2.0,
        structure_path=str(structure_file),
    )

    assert structure_file.exists(), "original structure file must not be deleted"


def test_get_reward_function_removes_temporary_file_after_parse_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A temporary altloc-fixed CIF must be removed when parsing raises."""
    temporary_file = tmp_path / "fixed.cif"
    temporary_file.touch()

    def fail_parse(*args, **kwargs):
        raise ValueError("invalid structure")

    monkeypatch.setattr(
        "sampleworks.utils.guidance_script_utils.resolve_mixed_hetatm_atom_altlocs",
        lambda path: temporary_file,
    )
    monkeypatch.setattr("sampleworks.utils.guidance_script_utils.parse_structure", fail_parse)

    with pytest.raises(ValueError, match="invalid structure"):
        get_reward_function_and_structure(
            density="dummy_density.mrc",
            device=torch.device("cpu"),
            em=False,
            loss_order=2,
            resolution=2.0,
            structure_path=tmp_path / "input.cif",
        )

    assert not temporary_file.exists()


def test_get_reward_function_ignores_temporary_file_cleanup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failed temporary-file unlink must not turn a parsed structure into an error."""
    temporary_file = tmp_path / "fixed.cif"
    temporary_file.touch()
    unlink_attempts = []

    def fail_unlink(path, *args, **kwargs):
        unlink_attempts.append(path)
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr(
        "sampleworks.utils.guidance_script_utils.resolve_mixed_hetatm_atom_altlocs",
        lambda path: temporary_file,
    )
    monkeypatch.setattr(
        "sampleworks.utils.guidance_script_utils.parse_structure",
        lambda *args, **kwargs: {
            "asym_unit": build_test_atom_array(n_atoms=3, with_occupancy=True)
        },
    )
    monkeypatch.setattr(
        "sampleworks.utils.guidance_script_utils.XMap",
        MagicMock(fromfile=MagicMock(return_value=MagicMock())),
    )
    monkeypatch.setattr(
        "sampleworks.utils.guidance_script_utils.setup_scattering_params",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "sampleworks.utils.guidance_script_utils.RealSpaceRewardFunction",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    _, structure = get_reward_function_and_structure(
        density="dummy_density.mrc",
        device=torch.device("cpu"),
        em=False,
        loss_order=2,
        resolution=2.0,
        structure_path=tmp_path / "input.cif",
    )

    assert len(structure["asym_unit"]) == 3
    assert unlink_attempts == [temporary_file]


def test_write_job_metadata_with_job_result_appends_timing_and_status(
    tmp_path: Path, guidance_job_result: JobResult
):
    """JobResult fields (timing, status, exit_code) must be merged into job_metadata.json."""
    args = GuidanceConfig(
        protein="1l63",
        structure=Path("dummy"),
        density=Path("dummy"),
        model_name="boltz2",
        guidance_type="pure_guidance",
        log_path="dummy",
        output_dir=str(tmp_path),
    )

    _write_job_metadata(tmp_path, args, guidance_job_result)

    metadata = json.loads((tmp_path / "job_metadata.json").read_text())
    # GuidanceConfig keys are preserved
    assert metadata["protein"] == "1l63"
    assert metadata["guidance_type"] == "pure_guidance"
    assert metadata["model_name"] == "boltz2"
    assert "model" not in metadata
    # JobResult-only keys are appended
    assert metadata["started_at"] == "2026-05-05T10:00:00"
    assert metadata["finished_at"] == "2026-05-05T10:00:12.340000"
    assert metadata["runtime_seconds"] == 12.34
    assert metadata["status"] == "success"
    assert metadata["exit_code"] == 0


def test_write_job_metadata_creates_missing_output_dir(
    tmp_path: Path, guidance_job_result: JobResult
):
    """Helper should create the output directory if it doesn't exist (failure-path safety)."""
    nested = tmp_path / "does" / "not" / "exist"
    args = GuidanceConfig(
        protein="1l63",
        structure=Path("dummy"),
        density=Path("dummy"),
        model_name="boltz2",
        guidance_type="pure_guidance",
        log_path="dummy",
        output_dir=str(nested),
    )

    _write_job_metadata(nested, args, guidance_job_result)

    assert (nested / "job_metadata.json").exists()


def test_write_job_metadata_records_altloc_occupancies(
    tmp_path: Path, guidance_job_result: JobResult
):
    """Metadata stores occupancies explicitly instead of relying on directory names."""
    args = GuidanceConfig(
        protein="1l63_0.25occA_0.75occB",
        structure=Path("dummy"),
        density=Path("dummy"),
        model_name="boltz2",
        guidance_type="pure_guidance",
        log_path="dummy",
        output_dir=str(tmp_path),
    )

    _write_job_metadata(tmp_path, args, guidance_job_result)

    metadata = json.loads((tmp_path / "job_metadata.json").read_text())
    assert metadata["altloc_occupancies"] == {"A": 0.25, "B": 0.75}


def test_write_job_metadata_remaps_job_result_paths_to_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """JobResult's output_dir/log_path must be host-remapped, not left as container paths.

    Without this, the JobResult merge would overwrite the GuidanceConfig host paths with
    container paths, regressing job_metadata.json reproducibility outside the container.
    """
    host_results_dir = str(tmp_path)
    monkeypatch.setenv("SAMPLEWORKS_HOST_RESULTS_DIR", host_results_dir)

    container_output = "/data/results/run42"
    container_log = "/data/results/run42/run.log"
    expected_output = f"{host_results_dir}/run42"
    expected_log = f"{host_results_dir}/run42/run.log"

    args = GuidanceConfig(
        protein="1l63",
        structure=Path("dummy"),
        density=Path("dummy"),
        model_name="boltz2",
        guidance_type="pure_guidance",
        log_path=container_log,
        output_dir=container_output,
    )
    job_result = JobResult(
        protein="1l63",
        model_name="boltz2",
        method=None,
        scaler="pure_guidance",
        ensemble_size=8,
        gradient_weight=0.1,
        gd_steps=200,
        status="success",
        exit_code=0,
        runtime_seconds=12.34,
        started_at="2026-05-05T10:00:00",
        finished_at="2026-05-05T10:00:12.340000",
        log_path=container_log,
        output_dir=container_output,
    )

    _write_job_metadata(tmp_path, args, job_result)

    metadata = json.loads((tmp_path / "job_metadata.json").read_text())
    assert metadata["output_dir"] == expected_output
    assert metadata["log_path"] == expected_log


# ============================================================================
# get_diffuse_reward_and_structure tests
# ============================================================================
#
# The diffuse reward is partly two-phase: __init__ reads the targets and reduces
# them to one shared reflection list, while the scattering kernels and grid wait
# for prepare(), which needs the model atom array. These pin the half that runs
# at build time, plus the structure loading the builder shares with the
# real-space path.


@pytest.fixture
def structure_copy(resources_dir: Path, tmp_path: Path) -> Path:
    """A throwaway copy of 1VME, so a deletion bug cannot damage the resource."""
    source = resources_dir / "1vme" / "1vme_final.cif"
    if not source.exists():
        pytest.skip(f"Source structure not found at {source}")
    destination = tmp_path / "input.cif"
    destination.write_bytes(source.read_bytes())
    return destination


@pytest.fixture
def diffuse_targets(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal Bragg/diffuse MTZ pair on a shared reflection list.

    Written with the generator's own writers so the column names and MTZ dtypes
    match what the pipeline produces, but from invented amplitudes: the reward
    only has to read them here, not agree with them.
    """
    pytest.importorskip("lunus.sf", reason="lunus[sf] not installed")
    import gemmi
    from sampleworks.synthetic.generate_synthetic_sf_lunus import (
        dataset_from_amplitudes,
        dataset_from_intensities,
    )

    cell = gemmi.UnitCell(30.0, 40.0, 50.0, 90.0, 90.0, 90.0)
    space_group = gemmi.SpaceGroup("P 1")
    hkl = np.array(
        [(h, k, ell) for h in range(-2, 3) for k in range(-2, 3) for ell in range(-2, 3)],
        dtype=np.int32,
    )
    amplitudes = np.linspace(10.0, 100.0, len(hkl)).astype(np.complex64)
    intensities = np.linspace(1.0, 10.0, len(hkl)).astype(np.float32)

    bragg_path = tmp_path / "bragg.mtz"
    diffuse_path = tmp_path / "diffuse.mtz"
    dataset_from_amplitudes(hkl, amplitudes, cell, space_group, output_path=bragg_path)
    dataset_from_intensities(hkl, intensities, cell, space_group, output_path=diffuse_path)
    return bragg_path, diffuse_path


def test_get_diffuse_reward_and_structure_returns_reward_and_parsed_structure(
    structure_copy: Path, diffuse_targets: tuple[Path, Path]
):
    bragg_path, diffuse_path = diffuse_targets

    reward, structure = get_diffuse_reward_and_structure(
        structure_path=structure_copy,
        bragg_target=bragg_path,
        diffuse_target=diffuse_path,
        bragg_weight=0.25,
        resolution=None,
    )

    assert "asym_unit" in structure
    assert len(structure["asym_unit"]) > 0
    assert reward.bragg_weight == 0.25


def test_get_diffuse_reward_and_structure_builds_a_preparable_reward(
    structure_copy: Path, diffuse_targets: tuple[Path, Path]
):
    """The trajectory scalers dispatch on this protocol, so the builder must satisfy it.

    The reward is not usable yet: prepare() still has to supply the atom array.
    """
    bragg_path, diffuse_path = diffuse_targets

    reward, _ = get_diffuse_reward_and_structure(
        structure_path=structure_copy,
        bragg_target=bragg_path,
        diffuse_target=diffuse_path,
        bragg_weight=0.5,
        resolution=None,
    )

    assert isinstance(reward, PreparableRewardFunctionProtocol)
    assert reward.setup is None, "kernels must wait for prepare()"


def test_get_diffuse_reward_and_structure_reads_the_targets_at_build_time(
    structure_copy: Path, tmp_path: Path
):
    """A missing target fails here, before the model weights are loaded.

    The reflection list comes from the targets, so it is read during
    construction rather than deferred with the coordinate-dependent parts.
    """
    with pytest.raises(RuntimeError, match="bragg.mtz"):
        get_diffuse_reward_and_structure(
            structure_path=structure_copy,
            bragg_target=tmp_path / "bragg.mtz",
            diffuse_target=tmp_path / "diffuse.mtz",
            bragg_weight=0.5,
            resolution=None,
        )


def test_get_diffuse_reward_and_structure_rejects_a_missing_target_before_reading(
    structure_copy: Path,
):
    """bragg_weight < 1 needs a diffuse target; that is caught without touching disk."""
    with pytest.raises(ValueError, match="diffuse_target is required"):
        get_diffuse_reward_and_structure(
            structure_path=structure_copy,
            bragg_target=None,
            diffuse_target=None,
            bragg_weight=0.0,
            resolution=None,
        )


def test_get_diffuse_reward_and_structure_keeps_the_original_structure_file(
    structure_copy: Path, diffuse_targets: tuple[Path, Path]
):
    """Same contract as the real-space builder: the caller's input must survive.

    ``_load_structure`` deletes the temporary file that altloc resolution may
    write, and the comparison guarding that unlink is by string value because
    ``Path(x) != str(x)``.
    """
    bragg_path, diffuse_path = diffuse_targets

    get_diffuse_reward_and_structure(
        structure_path=str(structure_copy),
        bragg_target=bragg_path,
        diffuse_target=diffuse_path,
        bragg_weight=0.5,
        resolution=None,
    )

    assert structure_copy.exists(), "original structure file must not be deleted"


def test_get_diffuse_reward_and_structure_leaves_no_temporary_structures_behind(
    structure_copy: Path, diffuse_targets: tuple[Path, Path], tmp_path: Path
):
    """Altloc resolution may write a temporary cif; it must be cleaned up."""
    bragg_path, diffuse_path = diffuse_targets

    get_diffuse_reward_and_structure(
        structure_path=structure_copy,
        bragg_target=bragg_path,
        diffuse_target=diffuse_path,
        bragg_weight=0.5,
        resolution=None,
    )

    stray = [p.name for p in tmp_path.glob("*.cif") if p != structure_copy]
    assert stray == [], f"temporary structure files left behind: {stray}"


def test_get_diffuse_reward_and_structure_removes_hydrogens(
    structure_copy: Path, diffuse_targets: tuple[Path, Path]
):
    """The structure is loaded with hydrogens stripped, as every target type expects."""
    bragg_path, diffuse_path = diffuse_targets

    _, structure = get_diffuse_reward_and_structure(
        structure_path=structure_copy,
        bragg_target=bragg_path,
        diffuse_target=diffuse_path,
        bragg_weight=0.5,
        resolution=None,
    )

    assert "H" not in set(structure["asym_unit"].element)
