from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from esm.models.esmfold2.model import EsmFold2Model


@dataclass(frozen=True, slots=True)
class ESMFold2WrapperConfig:
    """Configure checkpoint loading and model execution.

    Attributes
    ----------
    checkpoint_path
        Hugging Face repository ID or local checkpoint directory.
    device
        Device on which to place the model. Defaults to CUDA.
    dtype
        Checkpoint loading dtype. None preserves the native loader's default.
    esmc_precision
        Precision for the ESMC language-model backbone, configured separately
        from the folding model. Defaults to the native FP8 setting; use
        ``"fp32"`` for CPU execution.
    revision
        Optional Hugging Face revision for the folding checkpoint.
    """

    checkpoint_path: str | Path = "biohub/ESMFold2"
    device: str | torch.device = "cuda"
    dtype: torch.dtype | None = None
    esmc_precision: Literal["fp8", "bf16", "fp32"] = "fp8"
    revision: str | None = None


class ESMFold2Wrapper:
    """Adapt ESMFold2 using its native checkpoint loader.

    Parameters
    ----------
    config
        Checkpoint location and runtime settings.

    Notes
    -----
    Construction loads model weights and may download the folding checkpoint
    and ESMC backbone. Parameters are frozen and the model is set to evaluation
    mode. Freezing weights does not disable autograd globally; coordinate and
    latent gradients depend on the denoising entry point used by ``step()``.
    """

    def __init__(self, config: ESMFold2WrapperConfig):
        """Load the native model with the requested runtime settings.

        Parameters
        ----------
        config
            Checkpoint location, device, and precision settings.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.model = EsmFold2Model.from_pretrained(
            config.checkpoint_path,
            device=self.device,
            dtype=config.dtype,
            esmc_precision=config.esmc_precision,
            revision=config.revision,
        )
        self.model.eval()
        self.model.requires_grad_(False)
