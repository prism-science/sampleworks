"""End-to-end checks that a CLI invocation produces a usable loss function.

These cover the whole path a run takes: command line -> LossConfig -> the built
loss -> prepare() -> a value. They are the tests that would have caught the
structure-factor loss being unreachable from the CLI.
"""

from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch
from sampleworks.core.losses.config import build_loss
from sampleworks.core.losses.protocol import (
    LossFunctionProtocol,
    LossInputs,
    PreparableLossFunctionProtocol,
    prepare_loss_if_needed,
)
from sampleworks.core.losses.real_space_density import RealSpaceLossFunction
from sampleworks.core.losses.registry import LossBuildContext
from sampleworks.core.losses.structure_factor import StructureFactorLossFunction
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import load_guidance_structure

from tests.losses.loss_input_helpers import build_loss_input_tensors_without_coords


# Building either loss loads real experimental data through the qFit / SFcalculator
# stacks, the same code the gpu-marked loss tests exercise.
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
    """Run the CLI path, then build the loss the resulting configuration describes."""
    config = GuidanceConfig.from_cli(argv)
    structure = load_guidance_structure(structure_path)
    return build_loss(
        config.resolved_loss_config(),
        LossBuildContext(structure=structure, device=device),
    )


def test_density_loss_is_reachable_from_the_command_line(
    resources_dir: Path, density_path_1vme: Path, device
):
    structure_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"

    loss = build_from_argv(
        BASE_ARGV
        + [
            "--structure",
            str(structure_path),
            "--density",
            str(density_path_1vme),
            "--resolution",
            "1.8",
            "--loss-order",
            "1",
        ],
        structure_path,
        device,
    )

    assert isinstance(loss, RealSpaceLossFunction)
    assert isinstance(loss.loss, torch.nn.L1Loss)


def test_density_loss_target_does_not_depend_on_map_units(resources_dir: Path, device, tmp_path):
    """Experimental maps come in arbitrary units. The built loss should rescale them to the input
    structure's computed density, so the same map stored 30x smaller or larger is the same
    target."""
    structure_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    density_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
    targets = []
    for factor in (1 / 30, 30):
        ccp4 = gemmi.read_ccp4_map(str(density_path))
        grid = np.asarray(ccp4.grid)  # view: scaling it scales the map in place
        grid *= factor
        ccp4.update_ccp4_header()
        scaled_path = tmp_path / f"scaled_{factor:g}.ccp4"
        ccp4.write_ccp4_map(str(scaled_path))
        argv = ["--structure", str(structure_path), "--density", str(scaled_path)]
        loss = build_from_argv(BASE_ARGV + argv + ["--resolution", "1.8"], structure_path, device)
        targets.append(loss.transformer.xmap.array)

    torch.testing.assert_close(targets[0], targets[1], rtol=1e-4, atol=1e-6)


def test_structure_factor_loss_scores_a_structure_end_to_end(
    sf_1vme_cif_and_mtz_paths: tuple[Path, Path],
    structure_1vme_sf,
    device: torch.device,
):
    """--loss-type structure_factor: parse, build, prepare, and score."""
    structure_path, mtz_path = sf_1vme_cif_and_mtz_paths

    loss = build_from_argv(
        BASE_ARGV
        + [
            "--structure",
            str(structure_path),
            "--loss-type",
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

    assert isinstance(loss, StructureFactorLossFunction)
    assert isinstance(loss, PreparableLossFunctionProtocol)

    # The scalers prepare against the loss inputs built in model atom order
    # (SampleworksProcessedStructure.to_loss_inputs); here that is the structure the
    # synthetic MTZ was computed from, altlocs and all.
    atom_array = structure_1vme_sf
    prepare_loss_if_needed(
        loss,
        LossInputs.from_atom_array(atom_array, ensemble_size=1, device=device),
        device=device,
    )
    elements, b_factors, occupancies = build_loss_input_tensors_without_coords(atom_array, device)
    coords = torch.from_numpy(atom_array.coord).to(device=device, dtype=torch.float32)
    # One conformer, as a batch of one: losses are always called batched.
    per_atom = (elements.unsqueeze(0), b_factors.unsqueeze(0), occupancies.unsqueeze(0))

    value = loss(coords.unsqueeze(0), *per_atom)
    perturbed = loss((coords + torch.randn_like(coords) * 0.5).unsqueeze(0), *per_atom)

    # The MTZ was computed from these coordinates, so a run configured this way scores
    # them as a match (normalized amplitudes are unit-variance per shell), and moving
    # away from them costs more.
    assert torch.isfinite(value)
    assert 0.0 <= value.item() < 0.1
    assert perturbed > value


def test_a_configuration_file_composes_two_losses(
    tmp_path: Path,
    density_path_1vme: Path,
    sf_1vme_cif_and_mtz_paths: tuple[Path, Path],
    device: torch.device,
):
    """The two losses score different data about the same structure, and combine."""
    structure_path, mtz_path = sf_1vme_cif_and_mtz_paths
    config_file = tmp_path / "losses.yaml"
    config_file.write_text(
        "real_space_density:\n"
        "  weight: 0.4\n"
        f"  loss_options: {{density: {density_path_1vme}, resolution: 1.8}}\n"
        "structure_factor:\n"
        "  weight: 0.6\n"
        f"  loss_options: {{mtzfile: {mtz_path}, expcolumns: [Fprotein, SIGFprotein]}}\n"
    )

    loss = build_from_argv(
        BASE_ARGV + ["--structure", str(structure_path), "--loss-config", str(config_file)],
        structure_path,
        device,
    )

    assert isinstance(loss, LossFunctionProtocol)
    assert [type(term).__name__ for term in loss.losses] == [
        "RealSpaceLossFunction",
        "StructureFactorLossFunction",
    ]
    assert loss.weights == [0.4, 0.6]
