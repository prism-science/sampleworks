"""End-to-end checks that a CLI invocation produces a usable reward function.

These cover the whole path a run takes: command line -> RewardConfig -> the built
reward -> prepare() -> a value. They are the tests that would have caught the
structure-factor reward being unreachable from the CLI.
"""

from pathlib import Path

import pytest
import torch
from sampleworks.core.rewards.config import build_reward
from sampleworks.core.rewards.protocol import (
    PreparableRewardFunctionProtocol,
    prepare_reward_if_needed,
    RewardFunctionProtocol,
    RewardInputs,
)
from sampleworks.core.rewards.real_space_density import RealSpaceRewardFunction
from sampleworks.core.rewards.registry import RewardBuildContext
from sampleworks.core.rewards.structure_factor import StructureFactorRewardFunction
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import load_guidance_structure

from tests.rewards.reward_input_helpers import build_reward_input_tensors_without_coords


# Building either reward loads real experimental data through the qFit / SFcalculator
# stacks, the same code the gpu-marked reward tests exercise.
pytestmark = pytest.mark.gpu


BASE_ARGV = [
    "--model",
    "boltz2",
    "--guidance-type",
    "pure_guidance",
    "--protein",
    "1VME",
]


def build_from_argv(argv: list[str], structure_path: Path, device: torch.device):
    """Run the CLI path, then build the reward the resulting configuration describes."""
    config = GuidanceConfig.from_cli(argv)
    structure = load_guidance_structure(structure_path)
    return build_reward(
        config.resolved_reward_config(),
        RewardBuildContext(structure=structure, device=device),
    )


def test_density_reward_is_reachable_from_the_command_line(resources_dir: Path, device):
    structure_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    density_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"

    reward = build_from_argv(
        BASE_ARGV
        + [
            "--structure",
            str(structure_path),
            "--density",
            str(density_path),
            "--resolution",
            "1.8",
            "--loss-order",
            "1",
        ],
        structure_path,
        device,
    )

    assert isinstance(reward, RealSpaceRewardFunction)
    assert isinstance(reward.loss, torch.nn.L1Loss)


def test_structure_factor_reward_scores_a_structure_end_to_end(
    sf_1vme_cif_and_mtz_paths: tuple[Path, Path],
    structure_1vme_sf,
    device: torch.device,
):
    """--reward-type structure_factor: parse, build, prepare, and score."""
    structure_path, mtz_path = sf_1vme_cif_and_mtz_paths

    reward = build_from_argv(
        BASE_ARGV
        + [
            "--structure",
            str(structure_path),
            "--reward-type",
            "structure_factor",
            "--mtzfile",
            str(mtz_path),
            "--expcolumns",
            "Fprotein",
            "SIGFprotein",
            "--normalize-amplitude",
        ],
        structure_path,
        device,
    )

    assert isinstance(reward, StructureFactorRewardFunction)
    assert isinstance(reward, PreparableRewardFunctionProtocol)

    # The scalers prepare against the reward inputs built in model atom order
    # (SampleworksProcessedStructure.to_reward_inputs); here that is the structure the
    # synthetic MTZ was computed from, altlocs and all.
    atom_array = structure_1vme_sf
    prepare_reward_if_needed(
        reward,
        RewardInputs.from_atom_array(atom_array, ensemble_size=1, device=device),
        device=device,
    )
    elements, b_factors, occupancies = build_reward_input_tensors_without_coords(atom_array, device)
    coords = torch.from_numpy(atom_array.coord).to(device=device, dtype=torch.float32)
    # One conformer, as a batch of one: rewards are always called batched.
    per_atom = (elements.unsqueeze(0), b_factors.unsqueeze(0), occupancies.unsqueeze(0))

    value = reward(coords.unsqueeze(0), *per_atom)
    perturbed = reward((coords + torch.randn_like(coords) * 0.5).unsqueeze(0), *per_atom)

    # The MTZ was computed from these coordinates, so a run configured this way scores
    # them as a match (normalized amplitudes are unit-variance per shell), and moving
    # away from them costs more.
    assert torch.isfinite(value)
    assert 0.0 <= value.item() < 0.1
    assert perturbed > value


def test_a_configuration_file_composes_two_rewards(
    tmp_path: Path,
    resources_dir: Path,
    sf_1vme_cif_and_mtz_paths: tuple[Path, Path],
    device: torch.device,
):
    """The two rewards score different data about the same structure, and combine."""
    structure_path, mtz_path = sf_1vme_cif_and_mtz_paths
    density_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
    config_file = tmp_path / "rewards.yaml"
    config_file.write_text(
        "real_space_density:\n"
        "  weight: 0.4\n"
        f"  reward_options: {{density: {density_path}, resolution: 1.8}}\n"
        "structure_factor:\n"
        "  weight: 0.6\n"
        f"  reward_options: {{mtzfile: {mtz_path}, expcolumns: [Fprotein, SIGFprotein]}}\n"
    )

    reward = build_from_argv(
        BASE_ARGV + ["--structure", str(structure_path), "--reward-config", str(config_file)],
        structure_path,
        device,
    )

    assert isinstance(reward, RewardFunctionProtocol)
    assert [type(term).__name__ for term in reward.rewards] == [
        "RealSpaceRewardFunction",
        "StructureFactorRewardFunction",
    ]
    assert reward.weights == [0.4, 0.6]
