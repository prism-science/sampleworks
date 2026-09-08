import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from atomworks.enums import ChainType
from atomworks.ml.samplers import LoadBalancedDistributedSampler
from biotite.structure import AtomArray, AtomArrayStack
from jaxtyping import Float
from loguru import logger
from rf3.inference_engines import RF3InferenceEngine
from rf3.loss.loss import calc_chiral_grads_flat_impl
from rf3.model.RF3 import RF3WithConfidence
from rf3.trainers.rf3 import assert_no_nans, RF3TrainerWithConfidence
from rf3.utils.inference import InferenceInput, InferenceInputDataset
from torch import Tensor
from torch.utils.data import DataLoader

from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.framework_utils import match_batch
from sampleworks.utils.guidance_constants import StructurePredictor
from sampleworks.utils.msa import MSAManager


# Attached to the trainer-owned model after RF3InferenceEngine initialization in
# RF3Wrapper.__init__; reused wrappers retrieve the complete runtime through it.
_INFERENCE_ENGINE_ATTR = "_sampleworks_rf3_inference_engine"


@dataclass(frozen=True, slots=True)
class RF3Conditioning:
    """Conditioning tensors from RF3 trunk forward pass.

    Passable to diffusion module forward.

    Attributes
    ----------
    s_inputs : Tensor
        Input embeddings (S_inputs_I).
    s_trunk : Tensor
        Single representation from trunk (S_I).
    z_trunk : Tensor
        Pair representation from trunk (Z_II).
    features : dict[str, Any]
        Raw feature dict (f tensor).
    true_atom_array : AtomArray | None
        The AtomArray of the true structure, used for determining proper atom counts.
    model_atom_array : AtomArray | None
        The AtomArray of the atoms the model actually operates on for the current
        sequence. May differ from ``true_atom_array`` (padding, added/missing atoms,
        element differences, etc.). Used for atom reconciliation against the true structure.
    """

    s_inputs: Tensor
    s_trunk: Tensor
    z_trunk: Tensor
    features: dict[str, Any]
    true_atom_array: AtomArray | None = None
    model_atom_array: AtomArray | None = None


@dataclass
class RF3Config:
    """Configuration for RF3 featurization.

    Attributes
    ----------
    msa_path : str | Path | dict | None
        MSA specification. Can be:
        - dict: chain_id -> MSA file path mapping
        - str/Path to .json: JSON file with chain_id -> MSA path mapping
        - str/Path to .a3m: Single MSA file applied to all protein chains
        - None (default): No MSA information is used
    recycling_steps : int | None
        Number of recycling steps to perform. Default is None, uses model default.
    disable_chiral_features : bool
        If True, zero out chiral_centers in features dict so the chiral gradient
        feature contributes nothing to the atom single representation during
        diffusion. Useful during guidance where reward gradients may push
        coordinates into out-of-distribution chiral configurations. Default is False.
    track_chiral_features : bool
        If True, log the chiral gradient L2 norm at each denoising step. The
        chiral gradient (output of calc_chiral_grads_flat_impl on the EDM scaled
        coordinates) is the input feature to the model's chiral processing layer.
        Uses original features when disable_chiral_features is True to
        determine if guidance would be breaking them (e.g. magnitude is much larger when guidance
        is on than off). Default is False.
    """

    msa_path: str | Path | dict | None = None
    recycling_steps: int | None = None
    disable_chiral_features: bool = False
    track_chiral_features: bool = False


def annotate_structure_for_rf3(
    structure: dict,
    *,
    msa_path: str | Path | dict | None = None,
    recycling_steps: int | None = None,
    disable_chiral_features: bool = False,
    track_chiral_features: bool = False,
) -> dict:
    """Annotate an Atomworks structure with RF3-specific configuration.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary.
    msa_path : str | Path | dict | None
        MSA specification. Can be:
        - dict: chain_id -> MSA file path mapping
        - str/Path to .json: JSON file with chain_id -> MSA path mapping
        - str/Path to .a3m: Single MSA file applied to all protein chains
        - None (default): No MSA information is used
    recycling_steps : int | None
        Number of recycling steps to perform. Default is None, uses model default.
    disable_chiral_features : bool
        If True, zero out chiral features during guidance. Default is False.
    track_chiral_features : bool
        If True, log the chiral gradient L2 norm at each denoising step. Default is False.

    Returns
    -------
    dict
        Structure dict with "_rf3_config" key added.
    """
    config = RF3Config(
        msa_path=msa_path,
        recycling_steps=recycling_steps,
        disable_chiral_features=disable_chiral_features,
        track_chiral_features=track_chiral_features,
    )
    return {**structure, "_rf3_config": config}


# TODO: This should go in some sort of atomworks utils module
def add_msa_to_chain_info(chain_info: dict, msa_path: str | Path | dict | None) -> dict:
    """Add MSA paths to chain_info dictionary.

    Parameters
    ----------
    chain_info: dict
        Original chain_info dictionary.
    msa_path: dict | str | Path | None
        MSA specification. Can be:
        - dict: chain_id -> MSA file path mapping
        - str/Path to .json: JSON file with chain_id -> MSA path mapping
        - str/Path to .a3m: Single MSA file applied to all protein chains
        - None: No MSA information is used

    Returns
    -------
    dict
        Updated chain_info dictionary with MSA paths.
    """
    updated_chain_info = chain_info.copy()

    if msa_path is None:
        return updated_chain_info

    # If msa_path is a JSON file, read it to get chain_id -> msa_path mapping
    if isinstance(msa_path, (str, Path)):
        msa_path_obj = Path(msa_path)
        if msa_path_obj.suffix == ".json" and msa_path_obj.exists():
            with open(msa_path_obj) as f:
                msa_path = json.load(f)

    # InferenceInput expects msa_path in chain_info
    for chain_id in updated_chain_info:
        if updated_chain_info[chain_id]["chain_type"] == ChainType.POLYPEPTIDE_L:
            if isinstance(msa_path, dict):
                chain_msa_path = msa_path.get(chain_id, None)
            else:
                chain_msa_path = msa_path

            if chain_msa_path is not None:
                updated_chain_info[chain_id]["msa_path"] = chain_msa_path

    return updated_chain_info


def _cuda_index(device: torch.device | str) -> int:
    """Extract the CUDA device index from a ``torch.device`` or string.

    Parameters
    ----------
    device: torch.device | str
        Device spec, e.g. ``"cuda:3"``, ``torch.device("cuda", 2)``, or ``"cuda"``.

    Returns
    -------
    int
        Device index (``0`` when unspecified, matching Torch defaults).

    Raises
    ------
    ValueError
        If ``device`` is not a CUDA device. Currently, sampleworks only supports
        CUDA systems, so non-CUDA devices will fail here.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"RF3Wrapper requires a CUDA device, got {dev!r}")
    return dev.index if dev.index is not None else 0


class RF3Wrapper:
    """Wrapper for RosettaFold 3 (Baker Lab AlphaFold 3 replication)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        msa_manager: MSAManager | None = None,
        device: torch.device | str | None = None,
        model: Any | None = None,
    ):
        """
        Parameters
        ----------
        checkpoint_path: str | Path
            Filesystem path to the checkpoint containing trained weights.
        msa_manager: MSAManager | None
            MSA manager for retrieving MSAs for input structures.
        device: torch.device | str | None
            CUDA device to bind the underlying Lightning Fabric to (e.g. ``"cuda:3"``).
            When ``None``, Fabric picks the first available device. Required for
            parallel jobs that must target distinct GPUs — passing an ``int``
            to Fabric (the default) always resolves to GPU 0, which serialises
            otherwise-parallel workers onto a single device.
        model: Any | None
            Model previously obtained from another ``RF3Wrapper``. RF3 model
            reuse also reuses the originating inference engine because its
            trainer, preprocessing pipeline, Fabric strategy, and model are a
            single initialized runtime context.

            References: https://lightning.ai/docs/fabric/stable/fundamentals/launch.html
              devices argument to fabric run
            https://github.com/RosettaCommons/foundry/blob/b071919caa19ff334bc04b1b41145cac61eba819/src/foundry/trainers/fabric.py#L92
        """
        logger.info("Loading RF3 Inference Engine")

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.msa_manager = msa_manager
        self.msa_pairing_strategy = "greedy"
        self.inference_engine: RF3InferenceEngine

        if model is None:
            engine_kwargs: dict[str, Any] = {
                "ckpt_path": str(self.checkpoint_path),
                "diffusion_batch_size": 1,
            }
            if device is not None:
                engine_kwargs["devices_per_node"] = [_cuda_index(device)]

            self.inference_engine = RF3InferenceEngine(**engine_kwargs)
            self.inference_engine.initialize()
        else:
            inference_engine = getattr(model, _INFERENCE_ENGINE_ATTR, None)
            if inference_engine is None:
                raise ValueError(
                    "RF3 model reuse requires a model created by RF3Wrapper so its "
                    "initialized inference engine and Fabric context can also be reused"
                )
            self.inference_engine = cast(RF3InferenceEngine, inference_engine)
            engine_checkpoint = Path(self.inference_engine.ckpt_path).expanduser().resolve()
            if engine_checkpoint != self.checkpoint_path:
                raise ValueError(
                    f"Pre-loaded RF3 model uses checkpoint {engine_checkpoint}, "
                    f"not requested checkpoint {self.checkpoint_path}"
                )

        self.inference_engine.trainer = cast(
            RF3TrainerWithConfidence, self.inference_engine.trainer
        )
        if model is not None and self.inference_engine.trainer.state["model"] is not model:
            raise ValueError(
                "Pre-loaded RF3 model does not belong to its attached inference engine"
            )
        self._device = self.inference_engine.trainer.fabric.device
        if device is not None and _cuda_index(device) != _cuda_index(self._device):
            raise ValueError(f"RF3 runtime is on {self._device}, not requested device {device}")

        if model is None:
            self.model = self.inference_engine.trainer.state["model"]
            object.__setattr__(self.model, _INFERENCE_ENGINE_ATTR, self.inference_engine)
        else:
            self.model = model

        # Chiral feature state, set in featurize()
        self._track_chiral_features: bool = False
        self._chiral_grad_stats: list[dict[str, float]] = []
        self._original_chiral_centers: torch.Tensor | None = None
        self._original_chiral_dihedral_angles: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def _inner_model(self) -> RF3WithConfidence:
        """Access the unwrapped RF3WithConfidence model through EMA wrappers."""
        model = self.model
        if hasattr(model, "shadow"):
            # The model is wrapped with an ExponentialMovingAverage (EMA) wrapper
            # To access the EMA weights that we want to use for inference, we need
            # to access the `shadow` attribute (see AF3 Supplement section 5.6, RF3 preprint
            # supplement A.4.2)
            model = model.shadow
        return cast(RF3WithConfidence, model)

    def featurize(self, structure: dict) -> GenerativeModelInput[RF3Conditioning]:
        """From an Atomworks structure, calculate RF3 input features.

        Runs the trunk forward pass to produce conditioning features.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary. Can be annotated with RF3 config
            via `annotate_structure_for_rf3()`. Config is read from
            `structure["_rf3_config"]` if present, otherwise default RF3Config
            values are used.

        Returns
        -------
        GenerativeModelInput[RF3Conditioning]
            Model input with trunk conditioning.
        """
        config = structure.get("_rf3_config", RF3Config())
        if isinstance(config, dict):
            config = RF3Config(**config)

        # Reset tracking state for this featurization run
        self._track_chiral_features = config.track_chiral_features
        self._chiral_grad_stats = []

        msa_path = config.msa_path
        recycling_steps = config.recycling_steps

        if "asym_unit" not in structure:
            raise ValueError("structure must contain 'asym_unit' key")

        atom_array = structure["asym_unit"]
        chain_info = structure.get("chain_info", {})

        # if we have an MSAManager, then use it to get msa_paths unless they've been overridden
        # I'm using one of two possible sequences, which has
        #  non-canonicals filtered out.
        if self.msa_manager is not None and msa_path is None and chain_info:
            polypeptides = {
                chain_id: item["processed_entity_canonical_sequence"]
                for chain_id, item in chain_info.items()
                if item["chain_type"] == ChainType.POLYPEPTIDE_L
            }
            msa_path = self.msa_manager.get_msa(
                polypeptides, self.msa_pairing_strategy, structure_predictor=StructurePredictor.RF3
            )

            # These are debugging assertions.
            assert all(isinstance(pp, str) for pp in polypeptides.values())

        logger.info(f"Using MSA paths: {msa_path}")

        chain_info = add_msa_to_chain_info(chain_info, msa_path)

        inference_input = InferenceInput.from_atom_array(atom_array, chain_info=chain_info)

        inference_dataset = InferenceInputDataset([inference_input])
        trainer = cast(RF3TrainerWithConfidence, self.inference_engine.trainer)

        sampler = LoadBalancedDistributedSampler(
            dataset=inference_dataset,
            key_to_balance=inference_dataset.key_to_balance,
            num_replicas=trainer.fabric.world_size,
            rank=trainer.fabric.global_rank,
            drop_last=False,
        )

        loader = DataLoader(
            dataset=inference_dataset,
            sampler=sampler,
            batch_size=1,
            # multiprocessing is disabled since it shouldn't be hard to read
            # InferenceInput objects
            num_workers=0,
            collate_fn=lambda x: x,  # no collation since we're not batching
            pin_memory=True,
            drop_last=False,
        )

        input_batch = next(iter(loader))
        input_spec = cast(
            InferenceInput, input_batch[0]
        )  # since we're not batching, the loader returns a list of length 1

        # Hydra instantiation leaves pipeline imprecisely typed even though it is callable
        # at runtime.
        pipeline = cast(Any, self.inference_engine.pipeline)
        pipeline_output = pipeline(input_spec.to_pipeline_input())
        pipeline_output = trainer.fabric.to_device(pipeline_output)

        features = trainer._assemble_network_inputs(pipeline_output)

        assert_no_nans(
            features,
            msg=f"network_input for example_id: {pipeline_output['example_id']}",
        )

        pairformer_out = self._pairformer_pass(
            features, grad_needed=False, recycling_steps=recycling_steps or 10
        )

        true_atom_array: AtomArray = (
            cast(AtomArray, atom_array[0]) if isinstance(atom_array, AtomArrayStack) else atom_array
        )

        num_atoms = len(pairformer_out["features"]["atom_to_token_map"])

        # Use the pipeline output array that RF3's native inference
        # uses (rf3/inference_engines/rf3.py line 594).  The pipeline removes
        # atoms (H, OXT, etc.) that the model doesn't operate on automatically,
        # so we will use this for the "model_atom_array" that refers to the set of
        # atoms that the model operates on during sampling.
        if "atom_array" not in pipeline_output:
            raise ValueError(
                "pipeline_output is missing 'atom_array' key, cannot determine model_atom_array"
            )
        model_aa = pipeline_output["atom_array"].copy()
        if len(model_aa) != num_atoms:
            raise ValueError(
                f"model_atom_array has {len(model_aa)} atoms but the model's "
                f"atom_to_token_map has {num_atoms} entries. These must match exactly "
                "for correct coordinate-to-atom mapping."
            )

        # atomworks's add_missing_atoms adds unresolved atoms with
        # occupancy=0.0 and NaN coordinates when we get our atom array with
        # InferenceInput.from_atom_array. RF3 operates on these atoms (they're
        # in atom_to_token_map), so initialize their coordinates with noise and
        # set occupancy to 1.0 so they participate in guidance and don't get
        # masked out in reward functions.
        nan_coord_mask = np.any(np.isnan(model_aa.coord), axis=-1)
        if nan_coord_mask.any():
            resolved_coords = model_aa.coord[~nan_coord_mask]
            centroid = resolved_coords.mean(axis=0) if len(resolved_coords) > 0 else np.zeros(3)
            n_nan = int(nan_coord_mask.sum())
            noise = np.random.normal(loc=0.0, scale=1.0, size=(n_nan, 3)).astype(np.float32)
            new_coords = model_aa.coord.copy()
            new_coords[nan_coord_mask] = centroid + noise
            model_aa.coord = new_coords
            logger.info(
                f"Initialized {n_nan} unresolved atoms with noise "
                f"(had NaN coordinates from add_missing_atoms)"
            )

        # All atoms in model_aa are operated on by RF3 during diffusion.
        # Set occupancy to 1.0 regardless of what atomworks assigned (unresolved
        # atoms from add_missing_atoms get occupancy=0.0, but RF3 should use them)
        model_aa.set_annotation("occupancy", np.ones(len(model_aa), dtype=np.float32))
        if not hasattr(model_aa, "b_factor") or model_aa.b_factor is None:
            model_aa.set_annotation("b_factor", np.full(len(model_aa), 20.0, dtype=np.float32))
        else:
            nan_b_mask = np.isnan(model_aa.b_factor)
            if nan_b_mask.any():
                b_factors = model_aa.b_factor.copy()
                b_factors[nan_b_mask] = 20.0
                model_aa.set_annotation("b_factor", b_factors)
                logger.info(
                    f"Replaced {int(nan_b_mask.sum())} NaN B-factors with default 20.0 "
                    f"(from unresolved atoms added by add_missing_atoms)"
                )

        conditioning = RF3Conditioning(
            s_inputs=pairformer_out["s_inputs"],
            s_trunk=pairformer_out["s_trunk"],
            z_trunk=pairformer_out["z_trunk"],
            features=pairformer_out["features"],
            true_atom_array=true_atom_array,
            model_atom_array=model_aa,
        )

        # Store original chiral features for tracking before optionally zeroing them out
        self._original_chiral_centers: torch.Tensor | None = conditioning.features.get(
            "chiral_centers", None
        )
        self._original_chiral_dihedral_angles: torch.Tensor | None = conditioning.features.get(
            "chiral_center_dihedral_angles", None
        )

        if config.disable_chiral_features:
            # When chiral_centers.shape[0] == 0, calc_chiral_grads_flat_impl
            # returns zeros, and process_ch in RF3 (linearNoBias layer) maps zeros to zeros,
            # so the chiral contribution to Q_L is exactly zero.
            chiral_disabled_features = dict(conditioning.features)
            chiral_disabled_features["chiral_centers"] = torch.zeros(
                (0, 4), dtype=torch.long, device=self.device
            )
            chiral_disabled_features["chiral_center_dihedral_angles"] = torch.zeros(
                0, dtype=torch.float32, device=self.device
            )
            conditioning = RF3Conditioning(
                s_inputs=conditioning.s_inputs,
                s_trunk=conditioning.s_trunk,
                z_trunk=conditioning.z_trunk,
                features=chiral_disabled_features,
                true_atom_array=conditioning.true_atom_array,
                model_atom_array=conditioning.model_atom_array,
            )
            logger.info("Chiral features disabled: zeroed out chiral_centers in features dict")

        return GenerativeModelInput(conditioning=conditioning)

    def _pairformer_pass(
        self, features: dict[str, Any], grad_needed: bool = False, recycling_steps: int = 10
    ) -> dict[str, Any]:
        """Perform a pass through the RF3 trunk to obtain representations.

        Internal method that computes trunk representations.

        Parameters
        ----------
        features: dict[str, Any]
            Model features dict that is computed internally in featurize
            (raw features, not GenerativeModelInput).
        grad_needed: bool, optional
            Whether gradients are needed for this pass, by default False.
        recycling_steps: int, optional
            Number of recycling steps to perform. Defaults to 10.

        Returns
        -------
        dict[str, Any]
            Trunk outputs (s_inputs, s_trunk, z_trunk, features).
        """

        with (
            torch.set_grad_enabled(grad_needed),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):  # TODO: bfloat16 will require newer GPU generations and new CUDA
            recycling_output_generator = self._inner_model.trunk_forward_with_recycling(
                features["f"], n_recycles=recycling_steps
            )

            # (We use `deque` with maxlen=1 to ensure that we only keep the last output
            #  in memory)
            try:
                recycling_outputs = deque(recycling_output_generator, maxlen=1).pop()
            except IndexError:
                # Handle the case where the generator is empty
                raise RuntimeError("Recycling generator produced no outputs")

        s_inputs = recycling_outputs["S_inputs_I"]
        s_trunk = recycling_outputs["S_I"]
        z_trunk = recycling_outputs["Z_II"]

        return {
            "s_inputs": s_inputs,
            "s_trunk": s_trunk,
            "z_trunk": z_trunk,
            "features": features["f"],
        }

    def step(
        self,
        x_t: Float[Tensor, "batch atoms 3"],
        t: Float[Tensor, "*batch"] | float,
        *,
        features: GenerativeModelInput[RF3Conditioning] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        r"""Perform denoising at given timestep/noise level.

        Returns predicted clean sample :math:`\hat{x}_\theta`.

        Parameters
        ----------
        x_t : Float[Tensor, "batch atoms 3"]
            Noisy structure at timestep :math:`t`.
        t : Float[Tensor, "*batch"] | float
            Current timestep/noise level (:math:`\hat{t}` from noise schedule).
        features : GenerativeModelInput[RF3Conditioning] | None
            Model features as returned by ``featurize``.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Predicted clean sample coordinates.
        """
        if features is None or features.conditioning is None:
            raise ValueError("features with conditioning required for step()")

        cond = features.conditioning
        if not isinstance(x_t, torch.Tensor):
            x_t = torch.tensor(x_t, device=self.device, dtype=torch.float32)

        if isinstance(t, (int, float)):
            t_tensor = torch.tensor([t], device=self.device, dtype=x_t.dtype)
        else:
            t_tensor = t.to(device=self.device, dtype=x_t.dtype)
            if t_tensor.ndim == 0:
                t_tensor = t_tensor.unsqueeze(0)

        t_tensor = match_batch(t_tensor, target_batch_size=x_t.shape[0])

        with torch.autocast(
            "cuda", dtype=torch.float32
        ):  # TODO: bfloat16 will require newer GPU generations and new CUDA
            atom_coords_denoised: Tensor = self._inner_model.diffusion_module(
                X_noisy_L=x_t,
                t=t_tensor,
                f=cond.features,
                S_inputs_I=cond.s_inputs,
                S_trunk_I=cond.s_trunk,
                Z_trunk_II=cond.z_trunk,
            )

        # Track chiral gradient statistics using original features to report what the chiral
        # gradient would have been for diagnostics
        if self._track_chiral_features:
            if (
                self._original_chiral_centers is None
                or self._original_chiral_dihedral_angles is None
                or self._original_chiral_centers.shape[0] == 0
            ):
                raise ValueError(
                    "Chiral feature tracking is enabled, but original features are missing or empty"
                    ". Cannot compute chiral gradients for tracking. This may be due to an upstream"
                    " change in RF3 or RF3Wrapper.featurize()."
                )

            sigma_data = self._inner_model.diffusion_module.sigma_data
            f_pred = self._inner_model.diffusion_module.f_pred
            if f_pred != "edm":
                raise ValueError(f"Chiral tracking assumes EDM scaling, got {f_pred=}")
            # DiffusionModule.forward(): R_noisy_L = X_noisy_L / sqrt(t^2 + sigma_data^2)
            # t_tensor: (batch) to (batch, 1, 1) to broadcast with (batch, atoms, 3)
            R_L = x_t.detach() / torch.sqrt(t_tensor[..., None, None] ** 2 + sigma_data**2)

            chiral_grads = calc_chiral_grads_flat_impl(
                R_L,
                self._original_chiral_centers,
                self._original_chiral_dihedral_angles,
                self._inner_model.diffusion_module.atom_attention_encoder.no_grad_on_chiral_center,
            ).nan_to_num()

            # L2 norm per sample, then average across the batch. This tracks magnitudes,
            # doesn't correspond to anything directly in the RF3 code.
            # chiral_grads: [batch, atoms, 3]
            per_sample_norm = chiral_grads.flatten(1).norm(dim=1)  # [batch]
            l2_norm = per_sample_norm.mean().item()

            stats = {
                "t": t_tensor[0].item(),
                "l2_norm": l2_norm,
            }
            self._chiral_grad_stats.append(stats)
            logger.debug(f"Chiral grad stats (t={stats['t']:.4f}): L2={l2_norm:.4f}")

        return atom_coords_denoised.float()

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[RF3Conditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Create initial samples from the prior distribution.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        features : GenerativeModelInput[RF3Conditioning] | None, optional
            Model features as returned by `featurize`. Useful for determining shape.
        shape : tuple[int, ...] | None, optional
            Explicit shape of the generated state (in the form [num_atoms, 3]).
            NOTE: shape will override features if both are provided.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Gaussian initialized coordinates.

        Raises
        ------
        ValueError
            If both features and shape are None, or if shape is invalid.
        """
        if shape is not None:
            if len(shape) != 2 or shape[1] != 3:
                raise ValueError("shape must be of the form (num_atoms, 3)")
            return torch.randn((batch_size, *shape), device=self.device)

        if features is None or features.conditioning is None:
            raise ValueError("Either features or shape must be provided to initialize_from_prior()")

        cond = features.conditioning
        num_atoms = len(cond.features["atom_to_token_map"])

        return torch.randn((batch_size, num_atoms, 3), device=self.device)
