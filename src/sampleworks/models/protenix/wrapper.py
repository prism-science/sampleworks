from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from biotite.structure import AtomArray
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from jaxtyping import Float
from loguru import logger
from ml_collections import ConfigDict
from protenix.config import parse_configs
from protenix.data.data_pipeline import DataPipeline
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.model.protenix import (
    Protenix,
    update_input_feature_dict,
)
from protenix.utils.torch_utils import autocasting_disable_decorator, dict_to_tensor

# "runner" is actually part of protenix, but for some reason is a separate package
from runner.inference import download_infercence_cache as download_inference_cache
from torch import Tensor

from sampleworks.models.protenix.structure_processing import (
    add_terminal_oxt_atoms,
    create_protenix_input_from_structure,
    ensure_atom_array,
    filter_zero_occupancy,
)
from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.framework_utils import match_batch
from sampleworks.utils.guidance_constants import StructurePredictor
from sampleworks.utils.msa import MSAManager
from sampleworks.utils.torch_utils import send_tensors_in_dict_to_device


@dataclass(frozen=True, slots=True)
class ProtenixConditioning:
    """Conditioning tensors from Protenix Pairformer pass.

    Passable to diffusion module forward.

    Attributes
    ----------
    s_inputs : Tensor
        Input embeddings.
    s_trunk : Tensor
        Single representation from Pairformer.
    z_trunk : Tensor
        Pair representation from Pairformer.
    features : dict[str, Any]
        Raw feature dict for diffusion module.
    num_atoms : int
        Number of atoms in the Protenix model's representation. This is the authoritative
        atom count for the diffusion module and prior initialization.
    pair_z : Tensor | None
        Cached pair representation for diffusion conditioning.
    p_lm : Tensor | None
        Cached atom attention encoder output.
    c_l : Tensor | None
        Cached atom attention encoder output.
    true_atom_array : AtomArray | None
        The AtomArray of the true structure, used for alignment/evaluation.
    model_atom_array : AtomArray | None
        The AtomArray of the atoms the model actually operates on for the current
        sequence. May differ from ``true_atom_array`` (padding, added/missing atoms,
        element differences, etc.). Used for atom reconciliation against the true structure.
    """

    s_inputs: Tensor
    s_trunk: Tensor
    z_trunk: Tensor
    features: dict[str, Any]
    num_atoms: int
    pair_z: Tensor | None = None
    p_lm: Tensor | None = None
    c_l: Tensor | None = None
    true_atom_array: AtomArray | None = None
    model_atom_array: AtomArray | None = None


@dataclass
class ProtenixConfig:
    """Configuration for Protenix featurization.

    Attributes
    ----------
    out_dir : str | Path | None
        Output directory for intermediate JSON file.
    recycling_steps : int | None
        Number of recycling steps to perform. If None, uses model default.
    use_msa : bool
        Whether to generate MSA features.
    enable_diffusion_shared_vars_cache : bool
        Enable caching of shared variables in diffusion module.
    """

    out_dir: str | Path | None = None
    recycling_steps: int | None = None
    use_msa: bool = True
    enable_diffusion_shared_vars_cache: bool = True


def annotate_structure_for_protenix(
    structure: dict,
    *,
    out_dir: str | Path | None = None,
    recycling_steps: int | None = None,
    use_msa: bool = True,
    enable_diffusion_shared_vars_cache: bool = True,
) -> dict:
    """Annotate an Atomworks structure with Protenix-specific configuration.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary.
    out_dir : str | Path | None
        Output directory for intermediate files.
        Defaults to structure metadata ID or "protenix_output".
    recycling_steps : int | None
        Number of recycling steps to perform. If None, uses model default.
    use_msa : bool
        Whether to generate MSA features.
    enable_diffusion_shared_vars_cache : bool
        Enable caching of shared variables in diffusion module.

    Returns
    -------
    dict
        Structure dict with "_protenix_config" key added.
    """
    config = ProtenixConfig(
        out_dir=out_dir or structure.get("metadata", {}).get("id", "protenix_output"),
        recycling_steps=recycling_steps,
        use_msa=use_msa,
        enable_diffusion_shared_vars_cache=enable_diffusion_shared_vars_cache,
    )
    return {**structure, "_protenix_config": config}


def _has_ccd_components(json_dict: dict) -> bool:
    """Check whether any sequence entry contains truncated CCD codes.

    Detects entries where a CCD code has been truncated with a tilde
    (i.e., the string contains both ``"CCD_"`` and ``"~"``), which
    indicates a CCD identifier that was clipped during serialization.

    Parameters
    ----------
    json_dict : dict
        Protenix input JSON dict containing sequence information.

    Returns
    -------
    bool
        True if any ligand or ion entry contains a truncated CCD code.
    """
    return any(
        "CCD_" in c and "~" in c
        for entry in json_dict.get("sequences", [])
        for c in (
            entry.get("ligand", {}).get("ligand", ""),
            entry.get("ion", {}).get("ion", ""),
        )
    )


def _make_ccd_parse_error(json_dict: dict) -> TypeError:
    """Create an actionable TypeError for CCD parse failures.

    Parameters
    ----------
    json_dict : dict
        The Protenix input JSON dict containing sequence information.

    Returns
    -------
    TypeError
        With diagnostic message listing ligand/ion codes and suggesting mmCIF input.
    """
    ligand_codes = [
        entry.get("ligand", {}).get("ligand", "unknown")
        for entry in json_dict.get("sequences", [])
        if "ligand" in entry
    ]
    ion_codes = [
        entry.get("ion", {}).get("ion", "unknown")
        for entry in json_dict.get("sequences", [])
        if "ion" in entry
    ]
    component_summary = []
    if ligand_codes:
        component_summary.append(f"Ligand codes: {ligand_codes}")
    if ion_codes:
        component_summary.append(f"Ion codes: {ion_codes}")
    base_msg = "Protenix failed to parse one or more CCD components."
    if component_summary:
        base_msg += f" {'; '.join(component_summary)}."
    return TypeError(
        base_msg
        + " This commonly happens with tilde-prefixed CCD codes (e.g., '~QS') "
        + "from PDB format files where the full extended CCD code (e.g., 'A1AQS') "
        + "was truncated. Consider using mmCIF format input instead."
    )


class ProtenixWrapper:
    """
    Wrapper for Protenix (ByteDance AlphaFold3 implementation)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        args_str: str = "",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model: Protenix | None = None,
    ):
        """
        Parameters
        ----------
        checkpoint_path: str | Path
            Filesystem path to the Protenix checkpoint containing trained weights.
        args_str: str, optional
            Command-line style argument string to override default configurations.
        device: torch.device, optional
            Device to run the model on, by default CUDA if available.
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)

        # Protenix puts extra required information in the .a3m files about species/taxonomy,
        # easiest to just use their server for now, but I'm not a fan.
        self.msa_manager = MSAManager(msa_server_url="https://protenix-server.com/api/msa")

        self.cache_path = Path(checkpoint_path).parent.expanduser().resolve()

        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.cached_representations: dict[str, Any] = {}

        args = args_str.split()
        verified_arg_str = ""
        for k, v in zip(args[::2], args[1::2]):
            assert k.startswith("--")
            verified_arg_str += f"{k} {v} "

        configs = {**configs_base, **{"data": data_configs}, **inference_configs}
        self.configs: ConfigDict = parse_configs(
            configs=configs,
            arg_str=verified_arg_str,
            fill_required_with_null=True,
        )

        # Protenix inference logging
        self.configs.model_name = str(self.checkpoint_path).split("/")[-1].replace(".pt", "")
        self.configs.load_checkpoint_dir = str(self.cache_path)
        model_name = self.configs.model_name
        _, model_size, model_feature, model_version = model_name.split("_")
        logger.info(
            f"Inference by Protenix: model_size: {model_size}, with_feature: "
            f"{model_feature.replace('-', ', ')}, model_version: {model_version}"
        )
        model_specifics_configs = ConfigDict(model_configs[model_name])
        # update model specific configs
        self.configs.update(model_specifics_configs)
        logger.info(
            f"Triangle_multiplicative kernel: {self.configs.triangle_multiplicative}, "
            f"Triangle_attention kernel: {self.configs.triangle_attention}"
        )
        logger.info(
            f"enable_diffusion_shared_vars_cache: "
            f"{self.configs.enable_diffusion_shared_vars_cache}, "
            f"enable_efficient_fusion: {self.configs.enable_efficient_fusion}, "
            f"enable_tf32: {self.configs.enable_tf32}"
        )
        download_inference_cache(self.configs)

        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        if model is None:
            self.model = Protenix(self.configs).to(self.device)
        else:
            self.model = model.to(self.device)

        checkpoint_path_str = f"{self.configs.load_checkpoint_dir}/{self.configs.model_name}.pt"
        logger.info(f"Loading checkpoint from {checkpoint_path_str}")
        checkpoint = torch.load(checkpoint_path_str, map_location=self.device)

        if any(k.startswith("module.") for k in checkpoint["model"].keys()):
            checkpoint["model"] = {k[len("module.") :]: v for k, v in checkpoint["model"].items()}

        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=cast(dict, self.configs).get("load_strict", True),
        )
        self.model.eval()
        logger.info("Finished loading checkpoint")

        if cast(dict, self.configs.get("esm", {})).get("enable", False):
            from protenix.data.esm_featurizer import ESMFeaturizer

            esm_config = self.configs.esm
            self.esm_featurizer = ESMFeaturizer(
                embedding_dir=esm_config.embedding_dir,
                sequence_fpath=esm_config.sequence_fpath,
                embedding_dim=esm_config.embedding_dim,
                error_dir="./esm_embeddings/",
            )
        else:
            self.esm_featurizer = None

    def featurize(self, structure: dict) -> GenerativeModelInput[ProtenixConditioning]:
        """From an Atomworks structure, calculate Protenix input features.

        Runs the Pairformer pass to produce conditioning features.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary. Can be annotated with Protenix config
            via `annotate_structure_for_protenix()`. Config is read from
            `structure["_protenix_config"]` if present, otherwise default ProtenixConfig
            values are used.

        Returns
        -------
        GenerativeModelInput[ProtenixConditioning]
            Model input with Pairformer conditioning.
        """
        config = structure.get("_protenix_config", ProtenixConfig())
        if isinstance(config, dict):
            config = ProtenixConfig(**config)

        out_dir = config.out_dir or structure.get("metadata", {}).get("id", "protenix_output")
        use_msa = config.use_msa
        recycling_steps = config.recycling_steps
        enable_diffusion_shared_vars_cache = config.enable_diffusion_shared_vars_cache

        json_path, json_dict = create_protenix_input_from_structure(structure, out_dir)

        if use_msa:
            # first try the MSAManager

            import json

            updated_json_path = json_path.with_name(f"{json_path.stem}-add-msa.json")
            if not updated_json_path.exists():
                # get all the required sequences from the json_dict
                sequence_data: dict[str | int, str] = {
                    n: seq_data["proteinChain"]["sequence"]
                    for n, seq_data in enumerate(json_dict["sequences"])
                    if "proteinChain" in seq_data
                }
                msa_paths = self.msa_manager.get_msa(
                    sequence_data,
                    msa_pairing_strategy="complete",  # not actually passed through for Protenix
                    structure_predictor=StructurePredictor.PROTENIX,
                )

                for idx in sequence_data:
                    # see https://github.com/bytedance/Protenix/blob/main/runner/msa_search.py#L57
                    json_dict["sequences"][idx]["proteinChain"]["msa"] = {
                        "precomputed_msa_dir": str(msa_paths[idx]),
                        "pairing_db": "uniref100",
                    }

                # Dump the new config to a file--for consistency with Protenix, not really needed.
                with open(updated_json_path, "w") as f:
                    json.dump([json_dict], f)

            with open(updated_json_path) as f:
                json_data = json.load(f)
                json_dict = json_data[0]

        try:
            sample2feat = SampleDictToFeatures(json_dict)
            features_dict, atom_array_protenix, token_array = sample2feat.get_feature_dict()
        except TypeError as e:
            if _has_ccd_components(json_dict):
                raise _make_ccd_parse_error(json_dict) from e
            raise
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array_protenix.distogram_rep_atom_mask
        ).long()

        entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array_protenix)
        msa_features: dict[str, Any] = {}
        if use_msa:
            msa_features_raw = InferenceMSAFeaturizer.make_msa_feature(
                bioassembly=json_dict["sequences"],
                entity_to_asym_id=cast(dict[str, Any], entity_to_asym_id),
                token_array=token_array,
                atom_array=atom_array_protenix,
            )
            if msa_features_raw is not None:
                msa_features = cast(dict[str, Any], msa_features_raw)

        if self.esm_featurizer is not None:
            x_esm = self.esm_featurizer(
                token_array=token_array,
                atom_array=atom_array_protenix,
                bioassembly_dict=json_dict,
                inference_mode=True,
            )
            features_dict["esm_token_embedding"] = x_esm

        dummy_feats = ["template"]
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        feat = cast(dict[str, Any], data_type_transform(feat_or_label_dict=features_dict))

        if "constraint_feature" in feat and isinstance(feat["constraint_feature"], dict):
            for k, v in feat["constraint_feature"].items():
                feat[f"constraint_feature_{k}"] = v
            del feat["constraint_feature"]

        input_feature_dict = dict_to_tensor(feat)

        # TODO: all this processing is very jank, and we still get cases where the atom
        # numbers mismatch. I imagine this will be a common source of bugs for other models too...
        atom_array = ensure_atom_array(structure["asym_unit"])
        atom_array = filter_zero_occupancy(atom_array)
        atom_array = add_terminal_oxt_atoms(atom_array, structure.get("chain_info", {}))

        features = cast(dict[str, Any], input_feature_dict)

        if "asym_unit" in structure:
            # true_coords feeds into the model and must match the Protenix
            # feature dimensions (N_protenix atoms).
            true_coords = atom_array_protenix.coord
            if true_coords is None:
                raise ValueError("Protenix atom array has no coordinates")
            if not isinstance(true_coords, torch.Tensor):
                true_coords = torch.as_tensor(true_coords, device=self.device, dtype=torch.float32)
            features["true_coords"] = true_coords

        features = self.model.relative_position_encoding.generate_relp(features)
        features = update_input_feature_dict(features)
        features = send_tensors_in_dict_to_device(features, self.device, inplace=False)

        pairformer_kwargs: dict[str, Any] = {
            "enable_diffusion_shared_vars_cache": enable_diffusion_shared_vars_cache,
        }
        if recycling_steps is not None:
            pairformer_kwargs["recycling_steps"] = recycling_steps

        pairformer_out = self._pairformer_pass(features, grad_needed=False, **pairformer_kwargs)

        p_lm_c_l = pairformer_out.get("p_lm/c_l", [None, None])
        p_lm = p_lm_c_l[0] if p_lm_c_l else None
        c_l = p_lm_c_l[1] if p_lm_c_l else None

        # Build model atom array for mismatch reconciliation
        model_aa = cast(AtomArray, atom_array_protenix)
        if not hasattr(model_aa, "occupancy") or model_aa.occupancy is None:
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

        num_atoms_protenix = len(atom_array_protenix)
        conditioning = ProtenixConditioning(
            s_inputs=pairformer_out["s_inputs"],
            s_trunk=pairformer_out["s_trunk"],
            z_trunk=pairformer_out["z_trunk"],
            features=pairformer_out["features"],
            num_atoms=num_atoms_protenix,
            pair_z=pairformer_out.get("pair_z"),
            p_lm=p_lm,
            c_l=c_l,
            true_atom_array=atom_array if "asym_unit" in structure else None,
            model_atom_array=model_aa,
        )

        return GenerativeModelInput(conditioning=conditioning)

    def _pairformer_pass(
        self, features: dict[str, Any], grad_needed: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Perform a pass through the Protenix Pairformer to obtain representations.

        Internal method that computes Pairformer representations. Called by
        `featurize()` and cached for reuse across denoising steps.

        Parameters
        ----------
        features: dict[str, Any]
            Model features dict (raw features, not GenerativeModelInput).
        grad_needed: bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs: dict, optional
            Additional arguments.

            - recycling_steps: int
                Number of recycling steps to perform. Defaults to the value in
                the Protenix model config.

            - enable_diffusion_shared_vars_cache: bool, optional
                Enable caching of shared variables in diffusion module
                (default True).


        Returns
        -------
        dict[str, Any]
            Pairformer outputs (s_inputs, s_trunk, z_trunk, features, pair_z, p_lm/c_l).
        """
        inplace_safe = not grad_needed
        chunk_size = (
            cast(ConfigDict, self.configs.infer_setting).chunk_size if inplace_safe else None
        )
        with torch.set_grad_enabled(grad_needed):
            s_inputs, s, z = self.model.get_pairformer_output(
                input_feature_dict=features,
                N_cycle=kwargs.get("recycling_steps", cast(ConfigDict, self.configs.model).N_cycle),
                inplace_safe=inplace_safe,  # Default in Protenix is True
                chunk_size=cast(int, chunk_size),  # Default in Protenix is 4
            )

        keys_to_delete = []
        for key in features.keys():
            if "template_" in key or key in [
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
                "token_bonds",
            ]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del features[key]
        torch.cuda.empty_cache()

        outputs: dict[str, Any] = {
            "s_inputs": s_inputs,
            "s_trunk": s,
            "z_trunk": z,
            "features": features,
        }

        if kwargs.get(
            "enable_diffusion_shared_vars_cache",
            self.configs.enable_diffusion_shared_vars_cache,
        ):
            outputs["pair_z"] = autocasting_disable_decorator(
                self.model.configs.skip_amp.sample_diffusion
            )(self.model.diffusion_module.diffusion_conditioning.prepare_cache)(
                features["relp"], z, False
            )
            outputs["p_lm/c_l"] = autocasting_disable_decorator(
                self.model.configs.skip_amp.sample_diffusion
            )(self.model.diffusion_module.atom_attention_encoder.prepare_cache)(
                features["ref_pos"],
                features["ref_charge"],
                features["ref_mask"],
                features["ref_element"],
                features["ref_atom_name_chars"],
                features["atom_to_token_idx"],
                features["d_lm"],
                features["v_lm"],
                features["pad_info"],
                "",
                outputs["pair_z"],
                False,
            )
        else:
            outputs["pair_z"] = None
            outputs["p_lm/c_l"] = [None, None]

        return outputs

    def step(
        self,
        x_t: Float[Tensor, "batch atoms 3"],
        t: Float[Tensor, "*batch"] | float,
        *,
        features: GenerativeModelInput[ProtenixConditioning] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        r"""Perform denoising at given timestep/noise level.

        Returns predicted clean sample :math:`\hat{x}_\theta`.

        Parameters
        ----------
        x_t : Float[Tensor, "batch atoms 3"]
            Noisy structure at timestep :math:`t`.
        t : Float[Tensor, "*batch"] | float
            Current timestep/noise level (:math:`\hat{t}` from noise schedule).
        features : GenerativeModelInput[ProtenixConditioning] | None
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

        # Detach cached pairformer outputs under grad so the many denoising steps that reuse them
        # don't backprop through the trunk twice -- EXCEPT a latent that IT-opt has injected as an
        # optimizable leaf (requires_grad=True), which is kept attached so its gradient survives.
        # In the guidance path featurize() runs the trunk under no_grad, so every cached latent
        # is a requires_grad=False constant and detaches -- the exact behavior before IT-opt.
        grad_needed = torch.is_grad_enabled()

        def detach_unless_leaf(latent: Tensor) -> Tensor:
            """Detach a cached latent under grad, unless it is an optimizable IT-opt leaf."""
            return latent.detach() if grad_needed and not latent.requires_grad else latent

        s_inputs = detach_unless_leaf(cond.s_inputs)
        s_trunk = detach_unless_leaf(cond.s_trunk)
        z_trunk = detach_unless_leaf(cond.z_trunk)
        # pair_z / p_lm / c_l are z-derived caches. When optimizing z_trunk, featurize with
        # enable_diffusion_shared_vars_cache=False so these are None and the diffusion module
        # recomputes them from the live z_trunk; otherwise the z gradient is only partial.
        # See docs/IT_OPT_TESTING.md.
        pair_z = cond.pair_z.detach() if grad_needed and cond.pair_z is not None else cond.pair_z
        p_lm = cond.p_lm.detach() if grad_needed and cond.p_lm is not None else cond.p_lm
        c_l = cond.c_l.detach() if grad_needed and cond.c_l is not None else cond.c_l

        atom_coords_denoised = self.model.diffusion_module.forward(
            x_noisy=x_t,
            t_hat_noise_level=t_tensor,
            input_feature_dict=cond.features,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
        )

        # TODO: is there a way to handle this more cleanly?
        # remove protenix ensemble dim, shape (N_ensemble, 1, N_atoms, 3)
        # -> (N_ensemble, N_atoms, 3)
        atom_coords_denoised = atom_coords_denoised.squeeze(1)

        return atom_coords_denoised

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[ProtenixConditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Create a noisy version of a state at given noise level.

        Parameters
        ----------
        batch_size : int
            Number of noisy samples to generate.
        features : GenerativeModelInput[ProtenixConditioning] | None, optional
            Model features as returned by `featurize`. Useful for determining shape, etc. for
            the state.
        shape : tuple[int, ...] | None, optional
            Explicit shape of the generated state (in the form [num_atoms, 3]), if features is None
            or does not provide shape info. NOTE: shape will override features if both are provided.

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
        return torch.randn((batch_size, cond.num_atoms, 3), device=self.device)
