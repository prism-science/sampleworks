"""Fixtures for Protpardelle wrapper tests.

Build a small randomly-initialized ``ai-allatom`` model. The root conftest points
``PROTPARDELLE_MODEL_PARAMS`` at a throwaway directory so protpardelle imports
without downloaded weights.
"""

import textwrap
from pathlib import Path

import pytest


protpardelle_models = pytest.importorskip(
    "protpardelle.core.models", reason="Protpardelle not installed in this environment"
)


# A minimal cc89-style (``task: ai-allatom``, sequence-conditioned all-atom)
# config with the network dimensions shrunk so a randomly-initialized model is
# cheap to build and run on CPU. The data/diffusion fields that the model reads
# at inference time mirror cc89.yaml.
_SMALL_CC89_YAML = textwrap.dedent(
    """
    train:
      seed: null
      ckpt_path: null
      batch_size: 1
      max_epochs: 1
      eval_freq: 1
      checkpoint_freq: 1
      checkpoints: []
      lr: 0.0001
      warmup_steps: 1
      decay_steps: 1
      use_amp: False
      clip_grad_norm: True
      grad_clip_val: 1.0
      weight_decay: 0.0
      self_cond_train_prob: 0.9
      crop_conditional: True
      crop_cond:
        contiguous_prob: 0.05
        discontiguous_prob: 0.9
        sidechain_prob: 0.9
        sidechain_only_prob: 0.0
        max_span_len: 12
        max_discontiguous_res: 24
        dist_threshold: 45.0
        recenter_coords: True

    data:
      pdb_paths: ["/nonexistent"]
      mixing_ratios: [1.0]
      fixed_size: 64
      n_aatype_tokens: 21
      short_epoch: 0
      num_workers: 0
      se3_data_augment: True
      translation_scale: 1.0
      chain_residx_gap: 200
      sigma_data: 10.01
      auto_calc_sigma_data: False
      n_examples_for_sigma_data: 1
      dummy_fill_mode: "zero"
      subset: ["designable"]

    diffusion:
      training:
        function: "lognormal"
        psigma_mean: -0.5
        psigma_std: 1.5
      sampling:
        function: "uniform"
        s_min: 0.001
        s_max: 80.0

    model:
      task: "ai-allatom"
      pretrained_modules: []
      struct_model_checkpoint: ""
      mpnn_model_checkpoint: ""
      crop_conditional: True
      conditioning_style: "concat"
      compute_loss_on_all_atoms: false
      struct_model:
        arch: "dit"
        n_atoms: 37
        n_channel: 32
        noise_cond_mult: 2
        uvit:
          patch_size: 1
          n_layers: 2
          n_heads: 2
          dim_head: 16
          n_filt_per_layer: []
          n_blocks_per_layer: 2
          cat_pwd_to_conv: False
          conv_skip_connection: False
          position_embedding_type: "rotary"
          position_embedding_max: 32
      mpnn_model:
        use_self_conditioning: True
        label_smoothing: 0.1
        n_channel: 32
        n_layers: 2
        n_neighbors: 8
        noise_cond_mult: 2
    """
)


@pytest.fixture(scope="session")
def small_cc89_config_path(tmp_path_factory) -> Path:
    """Write the small ai-allatom config YAML to a temp file and return its path."""
    config_dir = tmp_path_factory.mktemp("protpardelle_config")
    config_path = config_dir / "cc89_small.yaml"
    config_path.write_text(_SMALL_CC89_YAML)
    return config_path


@pytest.fixture(scope="session")
def protpardelle_model(small_cc89_config_path: Path):
    """A small, randomly-initialized ai-allatom Protpardelle model on CPU."""
    import torch
    from protpardelle.configs import TrainingConfig
    from protpardelle.utils import load_config

    config = load_config(small_cc89_config_path, TrainingConfig)
    model = protpardelle_models.Protpardelle(config, device=torch.device("cpu"))
    model.eval()
    return model


@pytest.fixture(scope="session")
def protpardelle_wrapper(protpardelle_model, small_cc89_config_path):
    """A ProtpardelleWrapper backed by the small random model.

    A pre-built ``model`` is supplied, so ``checkpoint_path`` is only kept for
    record-keeping and need not point at a real file.
    """
    import torch
    from sampleworks.models.protpardelle.wrapper import ProtpardelleWrapper

    return ProtpardelleWrapper(
        checkpoint_path=small_cc89_config_path.parent / "unused.pth",
        config_path=small_cc89_config_path,
        device=torch.device("cpu"),
        model=protpardelle_model,
    )
