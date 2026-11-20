"""Structure predictors with their released architectures and random weights.

How a wrapper featurizes, steps, caches and passes gradients does not depend on trained
weights, so tests of that behavior can use models built from each release's own
configuration with random weights. They then run wherever the model runs, without the
checkpoints (#599).

Each builder reproduces the model a wrapper loads from a release checkpoint, minus the
weights: Boltz from the checkpoint's ``hyper_parameters`` plus the wrapper's own inference
overrides, Protenix from the configs it ships, RF3 from the checkpoint's ``train_cfg``.
The configs that do not ship with a package are copied from the released checkpoints into
``tests/resources/model_configs``.
"""

import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from omegaconf import DictConfig, OmegaConf
from sampleworks.utils.guidance_script_arguments import _CHECKPOINT_CANDIDATES
from sampleworks.utils.imports import RF3_AVAILABLE


if TYPE_CHECKING:
    from boltz.model.models.boltz1 import Boltz1
    from boltz.model.models.boltz2 import Boltz2


MODEL_CONFIGS = Path(__file__).parent / "resources" / "model_configs"

# Protenix derives its architecture from the checkpoint file name.
PROTENIX_MODEL_NAME = "protenix_base_default_v0.5.0"


def fill_zero_initialized_parameters(model: torch.nn.Module, seed: int = 0, std: float = 0.02):
    """Give every all-zero parameter small random values, in place.

    AF3-style models zero-initialize output projections, gates and adaptive norms so that
    a fresh network starts out as an identity map. Left at zero, the denoiser's output
    ignores the trunk and no gradient reaches the trunk representations.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose zero-initialized parameters are replaced.
    seed : int, optional
        Seed for the replacement values, by default 0.
    std : float, optional
        Standard deviation of the replacement values, by default 0.02.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.any():
                noise = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                parameter.copy_(noise * std)


def find_boltz_cache(model_key: str, required: str) -> Path | None:
    """Find a Boltz cache directory that holds the data a Boltz wrapper reads.

    The wrappers read Boltz's chemical component data from the directory that holds the
    checkpoint, so look next to each checkpoint candidate, then in Boltz's own cache.

    Parameters
    ----------
    model_key : str
        ``"boltz1"`` or ``"boltz2"``.
    required : str
        Entry the cache must contain: ``"ccd.pkl"`` for Boltz1, ``"mols"`` for Boltz2.

    Returns
    -------
    Path | None
        The first directory containing ``required``, or None if there is none.
    """
    candidates = [Path(path).expanduser().parent for path in _CHECKPOINT_CANDIDATES[model_key]]
    candidates.insert(0, Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser())
    return next((path for path in candidates if (path / required).exists()), None)


def _load_boltz_hparams(model_key: str, inference_hparams: dict[str, Any]) -> dict[str, Any]:
    """Merge a released Boltz checkpoint's hyperparameters with the wrapper's overrides.

    Parameters
    ----------
    model_key : str
        ``"boltz1"`` or ``"boltz2"``.
    inference_hparams : dict[str, Any]
        Overrides the wrapper passes to ``load_from_checkpoint``.

    Returns
    -------
    dict[str, Any]
        Constructor arguments, with nested settings as ``DictConfig`` like a checkpoint's.
    """
    released = OmegaConf.to_container(OmegaConf.load(MODEL_CONFIGS / f"{model_key}_hparams.yaml"))
    assert isinstance(released, dict)
    hparams = {**released, **inference_hparams}
    return {
        key: OmegaConf.create(value) if isinstance(value, dict) else value
        for key, value in hparams.items()
    }


def random_boltz1() -> "Boltz1":
    """Build the Boltz1 model ``Boltz1Wrapper`` loads, with random weights.

    Returns
    -------
    Boltz1
        Model on CPU, for ``Boltz1Wrapper(model=...)``.
    """
    from boltz.main import BoltzDiffusionParams, BoltzSteeringParams
    from boltz.model.models.boltz1 import Boltz1
    from sampleworks.models.boltz.wrapper import boltz1_inference_hparams

    overrides = boltz1_inference_hparams(BoltzDiffusionParams(), BoltzSteeringParams())
    torch.manual_seed(0)
    model = Boltz1(**_load_boltz_hparams("boltz1", overrides))
    fill_zero_initialized_parameters(model)
    return model


def random_boltz2() -> "Boltz2":
    """Build the Boltz2 model ``Boltz2Wrapper`` loads, with random weights.

    Returns
    -------
    Boltz2
        Model on CPU, for ``Boltz2Wrapper(model=...)``.
    """
    from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams
    from boltz.model.models.boltz2 import Boltz2
    from sampleworks.models.boltz.wrapper import boltz2_inference_hparams

    overrides = boltz2_inference_hparams(Boltz2DiffusionParams(), BoltzSteeringParams())
    torch.manual_seed(0)
    model = Boltz2(**_load_boltz_hparams("boltz2", overrides))
    fill_zero_initialized_parameters(model)
    return model


def write_random_protenix_checkpoint(directory: Path) -> Path:
    """Write a Protenix checkpoint that holds no weights.

    ``ProtenixWrapper`` builds the network from Protenix's configs and then loads the
    checkpoint's state dict; with ``--load_strict false`` an empty one keeps the network's
    random initialization. The file name selects the released base architecture.

    Parameters
    ----------
    directory : Path
        Directory to write the checkpoint into.

    Returns
    -------
    Path
        Checkpoint for ``ProtenixWrapper(checkpoint_path=..., args_str="--load_strict false")``.
    """
    checkpoint = directory / f"{PROTENIX_MODEL_NAME}.pt"
    torch.save({"model": {}}, checkpoint)
    return checkpoint


if RF3_AVAILABLE:
    from rf3.trainers.rf3 import RF3TrainerWithConfidence

    class RandomWeightsRF3Trainer(RF3TrainerWithConfidence):
        """RF3 trainer that keeps the network's random initialization."""

        def load_checkpoint(self, checkpoint, weight_loading_config=None, reset_optimizer=False):
            """Skip loading, leaving the freshly constructed weights in place.

            Parameters
            ----------
            checkpoint
                Ignored.
            weight_loading_config
                Ignored.
            reset_optimizer
                Ignored.
            """


def write_random_rf3_checkpoint(directory: Path) -> Path:
    """Write an RF3 checkpoint that holds the released config and no weights.

    ``RF3InferenceEngine`` builds its pipeline and network from a checkpoint's
    ``train_cfg``, then has the trainer load the weights. Pointing the trainer at
    :class:`RandomWeightsRF3Trainer` skips that load.

    Parameters
    ----------
    directory : Path
        Directory to write the checkpoint into.

    Returns
    -------
    Path
        Checkpoint for ``RF3Wrapper(checkpoint_path=...)``.
    """
    train_cfg = OmegaConf.load(MODEL_CONFIGS / "rf3_train_cfg.yaml")
    assert isinstance(train_cfg, DictConfig)
    train_cfg.trainer._target_ = f"{__name__}.RandomWeightsRF3Trainer"
    checkpoint = directory / "rf3_random_init.ckpt"
    torch.save({"train_cfg": train_cfg}, checkpoint)
    return checkpoint
