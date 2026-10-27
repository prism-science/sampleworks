"""Checkpoint-free BioEmu2 fixtures.

The real BioEmu2 checkpoint needs cuEquivariance kernels and a GPU. These fixtures
build a randomly initialized model from BioEmu2's own test configuration
(``trimul_variant: full``), which runs on CPU and exercises every code path in the
wrapper except the trained weights.
"""

from pathlib import Path

import pytest
import torch
from sampleworks.utils.imports import BIOEMU2_AVAILABLE


if not BIOEMU2_AVAILABLE:
    pytest.skip("BioEmu2 dependencies not installed in this environment", allow_module_level=True)

from hydra.utils import instantiate
from omegaconf import OmegaConf
from sampleworks.models.bioemu2.wrapper import BioEmu2Wrapper
from torch.optim.swa_utils import AveragedModel


@pytest.fixture(scope="session")
def tiny_bioemu2_checkpoint(tmp_path_factory: pytest.TempPathFactory, resources_dir: Path) -> Path:
    """Write a random-weight BioEmu2 checkpoint and its config.yaml.

    Returns
    -------
    Path
        Path to ``checkpoint.ckpt``; ``config.yaml`` sits beside it.
    """
    directory = tmp_path_factory.mktemp("tiny_bioemu2")
    score_model_config = OmegaConf.load(resources_dir / "bioemu2" / "tiny_config.yaml")
    OmegaConf.save({"score_model": score_model_config}, directory / "config.yaml")
    torch.manual_seed(0)
    state = AveragedModel(instantiate(score_model_config)).state_dict()
    checkpoint_path = directory / "checkpoint.ckpt"
    torch.save({"model_ema": state}, checkpoint_path)
    return checkpoint_path


@pytest.fixture(scope="session")
def tiny_bioemu2_wrapper(tiny_bioemu2_checkpoint: Path) -> BioEmu2Wrapper:
    """BioEmu2 wrapper around the random-weight model, on CPU."""
    return BioEmu2Wrapper(checkpoint_path=tiny_bioemu2_checkpoint, device="cpu")
