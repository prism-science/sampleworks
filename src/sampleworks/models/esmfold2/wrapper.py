from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import torch
import torch.nn.functional as F
from biotite.structure import AtomArray, AtomArrayStack
from esm.models.esmfold2.layers import (
    CHAR_VOCAB_SIZE,
    MAX_ATOMIC_NUMBER,
    maybe_apply_msa_column_masking,
    NUM_RES_TYPES,
)
from esm.models.esmfold2.model import EsmFold2Model
from esm.models.esmfold2.processor import _seed_context

from sampleworks.models.esmfold2.structure_processing import (
    esmfold2_features_to_atom_array,
    ESMFold2InputBuilderBlockLogs,
    structure_to_esmfold2_input,
)
from sampleworks.models.protocol import GenerativeModelInput


@dataclass
class ESMFold2Conditioning:
    """Reusable inputs to the native ESMFold2 diffusion module.

    Attributes
    ----------
    s_inputs
        Input embeddings, shape (batch, tokens, input_channels).
    z_trunk
        Trunk pair embeddings, shape (batch, tokens, tokens, pair_channels).
    relative_position_encoding
        Relative positions, shape (batch, tokens, tokens, pair_channels).
    features
        Batched, padded features with one-hot element and atom-name encodings.
    num_atoms
        Number of unpadded model atoms.
    model_atom_array
        Atom identities in unpadded model order, with CCD reference coordinates.
    true_atom_array
        Copy of the input structure's first frame, for reconciliation and alignment.
    """

    s_inputs: torch.Tensor
    z_trunk: torch.Tensor
    relative_position_encoding: torch.Tensor
    features: dict[str, torch.Tensor]
    num_atoms: int
    model_atom_array: AtomArray
    true_atom_array: AtomArray


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


def compute_esmfold2_trunk(
    model: EsmFold2Model, features: dict[str, torch.Tensor], num_loops: int | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the conditioning portion of ESMFold2Model's forward pass.

    Parameters
    ----------
    features
        Prepared features with padded, one-hot atom encodings.
    num_loops
        Native recycling setting, or None for the checkpoint default.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Input embeddings, trunk pair embeddings, and relative position encoding.

    Notes
    -----
    Mirrors esm 3.4.2 EsmFold2Model.forward up to structure_head.sample,
    reusing its embedding, recycling, and final trunk modules.
    """
    token_mask = features["token_attention_mask"]
    residue_types = F.one_hot(features["res_type"].long(), NUM_RES_TYPES).float()
    residue_types = residue_types * token_mask.unsqueeze(-1)

    msa_profile = F.one_hot(features["msa"].long(), NUM_RES_TYPES).float()
    msa_mask = features["msa_attention_mask"].float()
    profile = (msa_profile * msa_mask.unsqueeze(-1)).sum(dim=1)
    profile = profile / msa_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)

    with torch.autocast("cuda", enabled=model.device.type == "cuda", dtype=torch.bfloat16):
        # Build token inputs from sequence, MSA profile, and reference atom chemistry.
        s_inputs = model.inputs_embedder(
            aatype=residue_types,
            profile=profile,
            deletion_mean=features["deletion_mean"].float(),
            ref_pos=features["ref_pos"],
            atom_attention_mask=features["atom_attention_mask"],
            ref_space_uid=features["ref_space_uid"],
            ref_charge=features["ref_charge"],
            ref_element=features["ref_element"],
            ref_atom_name_chars=features["ref_atom_name_chars"],
            atom_to_token=features["atom_to_token"],
        )

        # Initialize each token pair from both tokens, relative positions, and bonds.
        z_init = model.z_init_1(s_inputs).unsqueeze(2) + model.z_init_2(s_inputs).unsqueeze(1)
        relative_position_encoding = model.rel_pos(
            residue_index=features["residue_index"],
            asym_id=features["asym_id"],
            sym_id=features["sym_id"],
            entity_id=features["entity_id"],
            token_index=features["token_index"],
        )
        z_init = (
            z_init + relative_position_encoding + model.token_bonds(features["token_bonds"].float())
        )

        # ESMC supplies sequence representations that are projected into pair features
        # for the folding trunk. This expensive computation is outside denoising.
        lm_hidden_states = model._compute_lm_hidden_states(
            features["input_ids"],
            features["asym_id"],
            features["residue_index"],
            features["mol_type"],
            token_mask,
            lm_mask_pct=model.config.lm_mask_pct,
        )
        lm_z = model.language_model(lm_hidden_states.detach())
        del lm_hidden_states

        # Pair mask: (batch, tokens, tokens).
        pair_mask = token_mask[:, :, None].float() * token_mask[:, None, :].float()
        z = model._init_pair_state(z_init)
        a, b = model._discretized_dynamics()
        msa_inputs = None
        if model.msa_encoder is not None:
            msa_config = model.config.msa_encoder
            msa_inputs = {
                "msa": features["msa"],
                "msa_attention_mask": maybe_apply_msa_column_masking(
                    features["msa_attention_mask"], rate=msa_config.column_mask_rate
                ),
                "has_deletion": features["has_deletion"],
                "deletion_value": features["deletion_value"],
                "x_inputs": s_inputs,
                "max_depth": msa_config.max_depth,
                "subsample_enabled": msa_config.max_depth is not None,
            }

        # ESMFold2 runs the trunk once, then repeats it num_loops more times. The
        # readout and coda layers further process the resulting pair embeddings. The
        # denoiser uses these embeddings to predict clean atom coordinates.
        loops = model.config.num_loops if num_loops is None else num_loops
        z = model._run_one_loop(
            z=z,
            z_init=z_init,
            lm_z=lm_z,
            _msa_inputs=msa_inputs,
            pair_mask=pair_mask,
            a=a.view(1, 1, 1, -1).to(z),
            b_mat=b.to(z),
            tok_mask=token_mask,
            total_steps=max(1, loops + 1),
        )
        z = model.parcae_readout(z)
        z = model.parcae_coda(z, pair_attention_mask=pair_mask)

    return s_inputs, z.float(), relative_position_encoding


@dataclass
class ESMFold2Wrapper:
    """A wrapper for ESMFold2.

    Attributes
    ----------
    model
        ESMFold2 model. Direct construction preserves its training mode and parameter
        gradient settings. Callers prepare the model for their use.
        :meth:`from_config()` creates an evaluation-mode model with frozen parameters.
    """

    model: EsmFold2Model

    @classmethod
    def from_config(cls, config: ESMFold2WrapperConfig) -> Self:
        """Load and freeze a model for inference and guidance.

        Parameters
        ----------
        config
            Various settings, including what's needed to instantiate/modify
            ``ESMFold2Model``.

        Returns
        -------
        Self
            Wrapper containing an evaluation-mode model with frozen parameters.
        """
        model = EsmFold2Model.from_pretrained(
            config.checkpoint_path,
            device=torch.device(config.device),
            dtype=config.dtype,
            esmc_precision=config.esmc_precision,
            revision=config.revision,
        )
        model.eval()
        model.requires_grad_(False)

        return cls(model)

    @torch.no_grad()
    def featurize(
        self,
        structure: dict[str, Any],
        *,
        seed: int = 0,
        num_loops: int | None = None,
    ) -> GenerativeModelInput[ESMFold2Conditioning]:
        """Prepare model atoms and cache conditioning before diffusion sampling.

        Parameters
        ----------
        structure
            Atomworks-parsed structure dictionary.
        seed
            Seed for input preparation and trunk randomness. The caller's random
            state is restored afterward.
        num_loops
            Native trunk recycling setting; None uses the checkpoint default.

        Returns
        -------
        GenerativeModelInput[ESMFold2Conditioning]
            Detached conditioning and atom identities for subsequent denoising.

        Notes
        -----
        Uses no_grad rather than inference_mode so cached tensors can participate
        in later gradient calculations. Does not sample or align coordinates.
        """
        model_input = structure_to_esmfold2_input(structure)
        atoms = structure["asym_unit"]
        if isinstance(atoms, AtomArrayStack):
            atoms = atoms[0]

        builder = ESMFold2InputBuilderBlockLogs()

        # This context manager seeds python, numpy, pytorch (CPU/CUDA) for reproducible
        # featurization and restores the previous random states upon exit.
        with _seed_context(seed):
            features, chains = builder.prepare_input(
                model_input,
                seed=seed,
                device=self.model.device,
            )

            # Recovers atom identities in native order before replacing integer feature
            # IDs with one-hot encodings.
            model_atoms = esmfold2_features_to_atom_array(features, chains)
            atom_mask = features["atom_attention_mask"].float()

            # This matches the encoding normally done inside native ESMFold2.forward().
            # Padded rows get zeroed so their placeholder IDs don't become real
            # element/name features.

            features["ref_element"] = F.one_hot(
                features["ref_element"].long(), num_classes=MAX_ATOMIC_NUMBER
            ).float() * atom_mask.unsqueeze(-1)  # (batch, atoms, elements)

            features["ref_atom_name_chars"] = F.one_hot(
                features["ref_atom_name_chars"].long(), num_classes=CHAR_VOCAB_SIZE
            ).float() * atom_mask.unsqueeze(-1).unsqueeze(-1)  # (batch, atoms, 4, characters)

            features["atom_to_token"] = features["atom_to_token"] * atom_mask.long()

            # Run the language model and trunk to get their conditioning.
            s_inputs, z_trunk, relative_position_encoding = compute_esmfold2_trunk(
                self.model, features, num_loops
            )

        return GenerativeModelInput(
            conditioning=ESMFold2Conditioning(
                s_inputs=s_inputs.detach(),
                z_trunk=z_trunk.detach(),
                relative_position_encoding=relative_position_encoding.detach(),
                features=features,
                num_atoms=len(model_atoms),
                model_atom_array=model_atoms,
                true_atom_array=atoms.copy(),
            )
        )
