"""Wrapper for Protpardelle-1c models.

Follows the ``FlowModelWrapper`` protocol in
``sampleworks.models.protocol`` so Protpardelle can be used interchangeably
with other structure predictors in sampling pipelines.

This wrapper targets the *sequence-conditioned* all-atom Protpardelle-1c
models (``task: "ai-allatom"``, e.g. the model defined by ``cc89.yaml``).
These models take a protein sequence as their only conditioning input and
generate an all-atom structure by running the full reverse-diffusion
trajectory internally. However, note that we pass a "sequence" as coordinates
for all possible atoms (Protpardelle's atom37 representation), with unused
atoms zeroed out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import biotite.structure as struc
import numpy as np
import torch
from atomworks.enums import ChainType
from jaxtyping import Float, Int
from loguru import logger
from protpardelle.common import residue_constants
from protpardelle.core.models import load_model, Protpardelle
from protpardelle.data.sequence import seq_to_aatype
from torch import Tensor

from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.framework_utils import match_batch
from sampleworks.utils.structure_utils import get_asym_unit_from_structure, get_valid_atom_mask


# Number of atom37 slots in the all-atom (atom37) representation. Matches
# ``residue_constants.atom_type_num`` and ``struct_model.n_atoms`` in the configs.
ATOM37_NUM_ATOMS = residue_constants.atom_type_num


# Number of aatype tokens for the 21-token alphabet (20 canonical + X).
# Matches ``data.n_aatype_tokens`` in the Protpardelle-1c configs (e.g. cc89.yaml).
NUM_AATYPE_TOKENS = 21

# Atomworks keeps selenomethionine atoms as ``SE`` while Protpardelle's atom37
# representation uses the canonical methionine sulfur slot, ``SD``.
ATOM37_ATOM_NAME_ALIASES = {"SE": "SD"}


@dataclass(slots=True)
class ProtpardelleConditioning:
    """Sequence-derived conditioning for a Protpardelle sampling run.

    All tensors carry a leading batch (ensemble) dimension and share the same
    padded length ``L``. Padding positions are indicated by ``seq_mask == 0``.

    Attributes
    ----------
    aatype : Int[Tensor, "batch L"]
        Per-residue amino-acid type indices (21-token alphabet). This is the
        sequence the structure is conditioned on (passed as ``gt_aatype``).
    seq_mask : Float[Tensor, "batch L"]
        1.0 for real residues, 0.0 for padding.
    residue_index : Float[Tensor, "batch L"]
        Residue ordering, with chain gaps applied by the model.
    chain_index : Float[Tensor, "batch L"]
        Integer chain id per residue.
    atom_mask : Float[Tensor, "batch L 37"]
        atom37 occupancy mask implied by ``aatype``; defines which atoms the
        model places and how predicted coordinates are flattened.
    atom37_residue_index : Int[Tensor, "atoms"]
        For each of the ``N`` flat atoms in the sampler's ``x_t`` (shape
        ``batch x N x 3``), the residue position ``0..L-1`` that atom belongs to.
        Together with :attr:`atom37_atom_index` this maps the flat coordinate
        tensor into the model's ``batch x L x 37 x 3`` atom37 layout (see
        :func:`_convert_to_atom37`). Shared across the batch
        (a single input structure), so it carries no batch dimension.
    atom37_atom_index : Int[Tensor, "atoms"]
        For each of the ``N`` flat atoms, the atom37 slot ``0..36`` it occupies,
        i.e. ``residue_constants.atom_order[atom_name]``.
    sequences : tuple[str, ...]
        The input one-letter sequences, one per protein chain (for reference).
    x_self_conditioning: Float[Tensor, "batch L 37 3"]
        Self-conditioning for the input structure,
        essentially the previously denoised coordinates (optional).
    """

    aatype: Int[Tensor, "batch L"]
    seq_mask: Float[Tensor, "batch L"]
    residue_index: Float[Tensor, "batch L"]
    chain_index: Float[Tensor, "batch L"]
    atom_mask: Float[Tensor, "batch L 37"]
    atom37_residue_index: Int[Tensor, " atoms"]
    atom37_atom_index: Int[Tensor, " atoms"]
    sequences: tuple[str, ...]
    x_self_conditioning: Float[Tensor, "batch L 37 3"] | None = None
    _initialized: bool = field(default=False, init=False, repr=False)

    _FROZEN = frozenset(
        {
            "aatype",
            "seq_mask",
            "residue_index",
            "chain_index",
            "atom_mask",
            "sequences",
        }
    )

    def __post_init__(self) -> None:
        """Mark construction complete so selected conditioning fields become immutable."""
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, key, value):
        """
        Prevent setting of frozen attributes after initialization.
        """
        if key in self._FROZEN and getattr(self, "_initialized", False):
            raise AttributeError(
                f"Cannot set attribute {key!r} on {self.__class__.__name__}, it is frozen!"
            )
        object.__setattr__(self, key, value)


@dataclass
class ProtpardelleConfig:
    """Configuration for Protpardelle featurization.

    Currently empty. The sampleworks trajectory scaler owns all sampling
    controls (ensemble size, schedule, stochasticity, step scaling), and
    featurization is fully determined by the input structure, so this config
    carries no fields. It is retained as this model's config type — mirroring
    the other wrappers — so it can gain fields in the future without changing
    the featurization interface.
    """


def annotate_structure_for_protpardelle(structure: dict) -> dict:
    """Annotate an Atomworks structure with Protpardelle-specific configuration.

    Retained for parity with the other model wrappers (and for future config
    fields); the current :class:`ProtpardelleConfig` carries no settings, so this
    is presently a structural no-op.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary.

    Returns
    -------
    dict
        Structure dict with a ``"_protpardelle_config"`` key added.
    """
    return {**structure, "_protpardelle_config": ProtpardelleConfig()}


def extract_protein_sequences(structure: dict) -> list[str]:
    """Extract canonical one-letter protein sequences from an Atomworks structure dictionary,
    created by the Atomworks ``parse`` method.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary. Sequences are read from
        ``structure["chain_info"]`` using the
        ``processed_entity_canonical_sequence`` of each protein (POLYPEPTIDE_L)
        chain, in chain order.

    Returns
    -------
    list[str]
        One sequence per protein chain.

    Raises
    ------
    ValueError
        If no ``chain_info`` is present or no protein chains are found.
    """
    chain_info = structure.get("chain_info")
    if not chain_info:
        raise ValueError(
            "Protpardelle featurization requires 'chain_info' with canonical "
            "sequences; none was found on the structure."
        )

    sequences: list[str] = []
    for chain_id, info in chain_info.items():
        chain_type: ChainType = info["chain_type"]
        if chain_type == ChainType.POLYPEPTIDE_L:
            sequence = info["processed_entity_canonical_sequence"]
            if sequence:
                sequences.append(sequence)
        else:
            logger.warning(
                f"Skipping non-protein chain {chain_id!r} (chain_type={chain_type}); "
                "Protpardelle only models L-polypeptide chains."
            )

    if not sequences:
        raise ValueError("No L-polypeptide (protein) chains found for Protpardelle.")

    return sequences


def _protein_chain_ids(structure: dict) -> list[str]:
    """Return the chain IDs of the protein chains, in chain order.

    Applies the same L-polypeptide / non-empty-sequence filter as
    :func:`extract_protein_sequences`, so the asym_unit can be restricted to
    exactly the chains that contribute sequence conditioning. This keeps the
    flat per-atom layout aligned with ``seq_mask`` / ``aatype`` and prevents
    ligand or other non-protein atoms from entering the atom37 mapping.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary with a ``"chain_info"`` mapping.

    Returns
    -------
    list[str]
        Protein chain IDs, in ``chain_info`` order.
    """
    chain_info = structure.get("chain_info", {})
    return [
        chain_id
        for chain_id, info in chain_info.items()
        if info["chain_type"] == ChainType.POLYPEPTIDE_L
        and info["processed_entity_canonical_sequence"]
    ]


def _atom37_indices_from_atom_array(
    atom_array,
    *,
    device: torch.device | str | None = None,
) -> tuple[Int[Tensor, " atoms"], Int[Tensor, " atoms"]]:
    """Derive per-atom atom37 destination indices from an Atomworks atom array.

    For each atom in ``atom_array`` (the order the sampler's flat ``x_t``
    follows), computes the residue position ``0..L-1`` and the atom37 slot
    ``0..36`` (``residue_constants.atom_order[atom_name]``). This mirrors how
    :func:`protpardelle.data.pdb_io.read_pdb` scatters atoms into the per-residue
    ``pos`` array of shape ``(37, 3)``.

    The returned maps define the canonical *input* atom order; feeding them to
    :func:`_convert_to_atom37` then :func:`_convert_atom37_to_flat` is an
    order-preserving round trip.

    Parameters
    ----------
    atom_array : biotite.structure.AtomArray
        Single-model atom array with ``atom_name`` / ``res_id`` / ``chain_id``
        annotations (a single frame; stacks are reduced to frame 0 upstream).
    device : torch.device | str | None
        Device for the returned tensors. Defaults to CPU.

    Returns
    -------
    tuple[Int[Tensor, "atoms"], Int[Tensor, "atoms"]]
        ``(atom37_residue_index, atom37_atom_index)``, both ``long`` tensors of
        length ``N``.

    Raises
    ------
    ValueError
        If any atom name is not a recognized atom37 atom type, which would
        break the one-to-one correspondence with ``x_t``.
    """
    atom_names = np.asarray(atom_array.atom_name)
    atom37_names = np.asarray(
        [ATOM37_ATOM_NAME_ALIASES.get(str(name), str(name)) for name in atom_names]
    )

    unknown = sorted(set(atom37_names) - set(residue_constants.atom_order))
    if unknown:
        raise ValueError(
            "asym_unit contains atom names not present in the atom37 "
            f"representation: {unknown}. Protpardelle's atom37 conversion "
            "requires every atom to map to a canonical atom type."
        )

    atom_slots = np.array(
        [residue_constants.atom_order[name] for name in atom37_names], dtype=np.int64
    )

    # Contiguous residue ordinal (0..L-1) per atom, in atom order across chains.
    # Note that this _is not_ a residue index. This is just to find indices that convert
    # to atom37 representation. Any reconciliation of missing residues happens elsewhere.
    residue_starts = struc.get_residue_starts(atom_array)
    is_start = np.zeros(len(atom_names), dtype=np.int64)
    is_start[residue_starts] = 1
    residue_ordinals = np.cumsum(is_start) - 1

    atom37_residue_index = torch.as_tensor(residue_ordinals, dtype=torch.long, device=device)
    atom37_atom_index = torch.as_tensor(atom_slots, dtype=torch.long, device=device)
    return atom37_residue_index, atom37_atom_index


def _convert_to_atom37(
    x_flat: Float[Tensor, "batch atoms 3"],
    atom37_residue_index: Int[Tensor, " atoms"],
    atom37_atom_index: Int[Tensor, " atoms"],
    num_residues: int,
) -> Float[Tensor, "batch L 37 3"]:
    """Scatter flat per-atom coordinates into Protpardelle's atom37 layout.

    Our sampler represents structures as ``x_flat`` of shape ``batch x N x 3``
    (``N`` = number of atoms), whereas Protpardelle expects
    ``batch x L x 37 x 3`` (``L`` = residues, 37 = atom37 slots). Using the
    per-atom residue/slot maps from :func:`_atom37_indices_from_atom_array`, this
    places each atom's coordinate at ``[batch, residue, slot]`` exactly like the
    ``pos`` tensor built in :func:`protpardelle.data.pdb_io.read_pdb`. Slots with
    no corresponding atom stay zero.

    The conversion is a differentiable scatter (``index_put``), so gradients flow
    from the returned tensor back to ``x_flat``. It is the exact inverse of
    :func:`_convert_atom37_to_flat`.

    Parameters
    ----------
    x_flat : Float[Tensor, "batch atoms 3"]
        Flat per-atom coordinates.
    atom37_residue_index : Int[Tensor, "atoms"]
        Per-atom residue position ``0..L-1`` (see
        :func:`_atom37_indices_from_atom_array`).
    atom37_atom_index : Int[Tensor, "atoms"]
        Per-atom atom37 slot ``0..36``.
    num_residues : int
        Padded residue count ``L`` of the destination layout.

    Returns
    -------
    Float[Tensor, "batch L 37 3"]
        Coordinates in atom37 layout, on the same device/dtype as ``x_flat``.

    Raises
    ------
    ValueError
        If ``x_flat``'s atom count disagrees with the atom37 index maps.
    """
    batch_size, num_atoms, _ = x_flat.shape

    residue_index = atom37_residue_index.to(x_flat.device)
    atom_index = atom37_atom_index.to(x_flat.device)

    if residue_index.shape[0] != num_atoms:
        raise ValueError(
            f"x has {num_atoms} atoms but the atom37 index maps describe "
            f"{residue_index.shape[0]} atoms; these must match."
        )

    x_atom37 = torch.zeros(
        (batch_size, num_residues, ATOM37_NUM_ATOMS, 3),
        dtype=x_flat.dtype,
        device=x_flat.device,
    )

    # Flatten (batch, atom) so a single index_put scatters every coordinate.
    batch_index = torch.arange(batch_size, device=x_flat.device).repeat_interleave(num_atoms)
    flat_residue_index = residue_index.repeat(batch_size)
    flat_atom_index = atom_index.repeat(batch_size)
    values = x_flat.reshape(batch_size * num_atoms, 3)

    # Out-of-place index_put keeps the op differentiable w.r.t. ``values``
    # (hence ``x_flat``); the freshly-zeroed destination needs no gradient.
    x_atom37 = x_atom37.index_put((batch_index, flat_residue_index, flat_atom_index), values)
    return x_atom37


def _convert_atom37_to_flat(
    x_atom37: Float[Tensor, "batch L 37 3"],
    atom37_residue_index: Int[Tensor, " atoms"],
    atom37_atom_index: Int[Tensor, " atoms"],
) -> Float[Tensor, "batch atoms 3"]:
    """Gather Protpardelle's atom37 coordinates back into flat per-atom order.

    Exact inverse of :func:`_convert_to_atom37`: for each atom ``i`` in the
    original input order, reads the coordinate stored at
    ``[atom37_residue_index[i], atom37_atom_index[i]]``. Because it indexes with
    the same per-atom maps used to scatter, the returned atoms appear in the
    *same order* as the source atom array.

    This is deliberately not a boolean-mask gather (``x_atom37[:, mask]``): a
    mask emits atoms in ascending ``(residue, slot)`` order, which reorders atoms
    within a residue whenever the input order differs from the atom37 slot order
    (e.g. PDB/CIF store ``O`` before ``CB``, but slot ``CB=3`` precedes ``O=4``).

    Differentiable (advanced-indexing gather), so gradients flow from the flat
    output back to ``x_atom37``.

    Parameters
    ----------
    x_atom37 : Float[Tensor, "batch L 37 3"]
        Coordinates in Protpardelle's atom37 layout.
    atom37_residue_index : Int[Tensor, "atoms"]
        Per-atom residue position ``0..L-1``.
    atom37_atom_index : Int[Tensor, "atoms"]
        Per-atom atom37 slot ``0..36``.

    Returns
    -------
    Float[Tensor, "batch atoms 3"]
        Flat per-atom coordinates in the original input atom order, on the same
        device/dtype as ``x_atom37``.
    """
    residue_index = atom37_residue_index.to(x_atom37.device)
    atom_index = atom37_atom_index.to(x_atom37.device)
    return x_atom37[:, residue_index, atom_index]


class ProtpardelleWrapper:
    """Wrapper for sequence-conditioned Protpardelle-1c all-atom models."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        device: torch.device | str,
        model: Protpardelle | None = None,
    ):
        """
        Parameters
        ----------
        config_path : str | Path
            Path to the Protpardelle model config YAML (e.g. ``cc89.yaml``).
            Required unless ``model`` is provided.
        checkpoint_path : str | Path
            Path to the matching ``.pth`` checkpoint with trained weights.
            Required unless ``model`` is provided.
        device : torch.device | str
            Device to load the model on. When ``None``, Protpardelle picks the
            default device (CUDA if available).
        model : Protpardelle | None
            A pre-built Protpardelle model. When given, ``config_path`` and
            ``checkpoint_path`` are still required (but only used for record keeping);
            the model is used directly (useful for testing and advanced reuse).
        """

        self._device = torch.device(device)
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if model is not None:
            self.model = model.to(self._device).eval()
        else:
            logger.info(
                f"Loading Protpardelle model from {self.config_path.name} "
                f"(checkpoint {self.checkpoint_path.name})"
            )
            self.model = load_model(self.config_path, self.checkpoint_path, device=self._device)

        if self.model.use_mpnn_model:
            logger.warning(
                "Loaded Protpardelle model has a MiniMPNN (sequence design) head; "
                "this wrapper treats the input sequence as fixed conditioning "
                "(gt_aatype) and does not redesign it."
            )

    @property
    def device(self) -> torch.device:
        return self._device

    def featurize(self, structure: dict) -> GenerativeModelInput[ProtpardelleConditioning]:
        """From an Atomworks structure, derive Protpardelle sequence conditioning.

        Reads the canonical protein sequence(s) from the structure and builds
        the sequence mask, residue/chain indices, aatype, and atom37 mask
        required to condition sampling. No diffusion is run here.

        The conditioning is built with a single (batch=1) ensemble dimension;
        the trajectory scaler sets the ensemble size on the sampled coordinates,
        and :meth:`step` broadcasts the conditioning to match.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary, optionally annotated via
            :func:`annotate_structure_for_protpardelle`.

        Returns
        -------
        GenerativeModelInput[ProtpardelleConditioning]
            Sequence conditioning for :meth:`step`.
        """
        if "asym_unit" not in structure:
            raise ValueError(
                "Protpardelle featurization requires an 'asym_unit' atom array to "
                "map flat coordinates into the atom37 layout; none was found on "
                "the structure."
            )

        sequences = extract_protein_sequences(structure)

        # Lay out chains exactly as Protpardelle's own sampling helper does so
        # residue/chain indexing (including inter-chain gaps) matches training.
        # Keep the length tensor on CPU because Protpardelle's helper constructs
        # intermediate residue indices on CPU before moving its outputs to the
        # model device.
        prot_lens_per_chain = torch.tensor([[len(seq) for seq in sequences]], dtype=torch.long)
        seq_mask, residue_index, chain_index = self.model.make_seq_mask_for_sampling(
            prot_lens_per_chain=prot_lens_per_chain
        )

        # Concatenate per-chain aatypes in chain order; chains are placed
        # contiguously at the front of the padded sequence by the helper above.
        chain_aatypes = [seq_to_aatype(seq, num_tokens=NUM_AATYPE_TOKENS) for seq in sequences]
        flat_aatype = torch.cat(chain_aatypes).to(self.device)
        padded_len = seq_mask.shape[1]
        aatype = torch.zeros((1, padded_len), dtype=torch.long, device=self.device)
        aatype[0, : flat_aatype.shape[0]] = flat_aatype

        # Conditioning stays at batch=1 here (``make_seq_mask_for_sampling``
        # returns ``[1, L]``); the trajectory scaler owns the ensemble size and
        # sizes the sampled coordinates accordingly, and ``step`` broadcasts the
        # conditioning to match that batch.

        # Map the flat per-atom coordinates the sampler works with (``x_t``,
        # shape ``batch x N x 3``) onto the model's ``batch x L x 37 x 3`` atom37
        # layout. The mapping is derived from the input structure's atom names.
        atom_array = get_asym_unit_from_structure(structure, atom_array_index=0)

        # Restrict to protein chains so ligand / non-polymer atoms from mixed
        # inputs never enter the atom37 mapping, which assumes a protein-only
        # layout aligned with the sequence-derived ``seq_mask`` / ``aatype``.
        protein_chain_ids = _protein_chain_ids(structure)
        atom_array = atom_array[np.isin(atom_array.chain_id, protein_chain_ids)]

        # Compute atom37 indices from the full protein-chain atoms BEFORE filtering to
        # valid atoms. This ensures residue ordinals match the padded sequence even if
        # entire residues are missing. Then filter both atoms and indices to remove
        # zero-occupancy or NaN-coordinate atoms.
        atom37_residue_index, atom37_atom_index = _atom37_indices_from_atom_array(
            atom_array, device=self.device
        )

        # Filter to only valid atoms (occupancy > 0, non-NaN coordinates) when available.
        # Structures from real files have occupancy; mock test structures may not.
        if getattr(atom_array, "occupancy", None) is not None:
            valid_atom_mask = get_valid_atom_mask(atom_array)
            atom_array = atom_array[valid_atom_mask]
            # Also filter the indices to match the filtered atoms.
            valid_mask_tensor = torch.as_tensor(
                valid_atom_mask, dtype=torch.bool, device=self.device
            )
            atom37_residue_index = atom37_residue_index[valid_mask_tensor]
            atom37_atom_index = atom37_atom_index[valid_mask_tensor]

        # the atom_mask created here is used in two ways. First it is used to generate
        # the initial noisy coordinates (see initialize_from_prior() below) simply for the
        # atom count. Second it is used in .step() to convert back from atom37 coordinates
        # (B x res x 37 x 3) to (B x atoms x 3) coordinates. It is only used to operate on
        # the structure we will model, never the reference structure which may have missing
        # atoms.
        #
        # This should be the way to make the atom mask:
        #   atom_mask = atom37_mask_from_aatype(aatype, seq_mask) but it doesn't handle OXT,
        # so:
        atom_mask = torch.zeros(
            (padded_len, ATOM37_NUM_ATOMS), dtype=torch.float, device=self.device
        )
        atom_mask[atom37_residue_index, atom37_atom_index] = 1
        atom_mask = atom_mask[None, :, :]  # [1, L, 37]

        conditioning = ProtpardelleConditioning(
            aatype=aatype,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            atom_mask=atom_mask,
            atom37_residue_index=atom37_residue_index,
            atom37_atom_index=atom37_atom_index,
            sequences=tuple(sequences),
            x_self_conditioning=None,
        )

        return GenerativeModelInput(conditioning=conditioning)

    def _expand_noise_level(
        self,
        t: Float[Tensor, "*batch"] | float,
        seq_mask: Float[Tensor, "batch L"],
        dtype: torch.dtype,
    ) -> Float[Tensor, "batch L"]:
        """Convert a sampler timestep/noise scalar to Protpardelle's ``B x L`` tensor.

        Parameters
        ----------
        t : Float[Tensor, "*batch"] | float
            Sampler noise level for the current EDM step. May be a scalar or one
            value per ensemble member.
        seq_mask : Float[Tensor, "batch L"]
            Sequence mask (already broadcast to the sampler's ensemble batch)
            defining the target batch and padded length.
        dtype : torch.dtype
            Floating dtype to use for the model call.

        Returns
        -------
        Float[Tensor, "batch L"]
            Noise level broadcast across valid/padded sequence positions on the
            wrapper device.
        """
        seq_mask = seq_mask.to(device=self.device, dtype=dtype)
        noise_level = torch.as_tensor(t, device=self.device, dtype=dtype)

        if noise_level.ndim == 0:
            return noise_level.expand_as(seq_mask).clone()

        if noise_level.ndim == 1:
            if noise_level.shape[0] != seq_mask.shape[0]:
                raise ValueError(
                    f"Noise level batch size {noise_level.shape[0]} does not match "
                    f"conditioning batch size {seq_mask.shape[0]}."
                )
            return noise_level[:, None].expand_as(seq_mask).clone()

        if noise_level.shape != seq_mask.shape:
            raise ValueError(
                f"Noise level shape {tuple(noise_level.shape)} must be scalar, "
                f"batch-sized, or match seq_mask shape {tuple(seq_mask.shape)}."
            )
        return noise_level

    def step(
        self,
        x_t: Float[Tensor, "batch atoms 3"],
        t: Float[Tensor, "*batch"] | float,
        *,
        features: GenerativeModelInput[ProtpardelleConditioning] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """
        Prepare data for and run the forward pass of the Protpardelle model.
        To do this, it converts our coordinate representation to its "atom37" format
        which has one row per _residue_ and a coordinate for all 37 possible atom types.
        See protpardelle-1c/src/protpardelle/core/models.py:L1760
        (commit ee378400f25b801fa481028000f9060183d7fb4c on branch main)

        The returned tensor is the final all-atom prediction, flattened to the
        atoms implied by the input sequence (the ``seq_mask``  attribute in the
        conditioning).

        Parameters
        ----------
        x_t : Float[Tensor, "batch atoms 3"]
            Noisy structure at timestep :math:`t`.
        t : Float[Tensor, "*batch"] | float
            Current timestep/noise level (:math:`\hat{t}` from EDM schedule).
        features : GenerativeModelInput[ProtpardelleConditioning] | None
            Model features as returned by ``featurize``.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Predicted coordinates for the present atoms, one row of length
            ``atoms`` per ensemble member.
        """
        if features is None or features.conditioning is None:
            raise ValueError("features with conditioning required for step()")

        cond = features.conditioning
        x_t = x_t.to(device=self.device)
        # The trajectory scaler sets the ensemble batch on ``x_t``; the stored
        # conditioning is batch=1, so broadcast it to match (mirrors how the
        # other wrappers derive the batch from ``x_t`` rather than featurization).
        batch_size = x_t.shape[0]

        # Our x_t is B x N x 3 (N = number of atoms); Protpardelle expects
        # B x L x 37 x 3. Scatter into the atom37 layout (gradient-preserving).
        x_t_atom37 = _convert_to_atom37(
            x_t, cond.atom37_residue_index, cond.atom37_atom_index, cond.seq_mask.shape[1]
        )

        seq_mask = match_batch(
            cond.seq_mask.to(device=self.device, dtype=x_t_atom37.dtype),
            target_batch_size=batch_size,
        )
        residue_index = match_batch(
            cond.residue_index.to(device=self.device), target_batch_size=batch_size
        )
        chain_index = match_batch(
            cond.chain_index.to(device=self.device), target_batch_size=batch_size
        )
        noise_level = self._expand_noise_level(t, seq_mask, x_t_atom37.dtype)
        struct_self_cond = cond.x_self_conditioning
        if struct_self_cond is not None:
            struct_self_cond = struct_self_cond.to(device=self.device, dtype=x_t_atom37.dtype)

        # To understand all these arguments better, you can study the (complicated)
        # Protpardelle.sample method. Hints: partial diffusion is enabled and cc.enabled = False!
        # Inside that method, the call to .forward() is at
        # https://github.com/ProteinDesignLab/protpardelle-1c/blob/ee378400f25b801fa481028000f9060183d7fb4c/src/protpardelle/core/models.py#L1766
        x0, s_logprobs, x_self_cond, s_self_cond = self.model.forward(
            noisy_coords=x_t_atom37,
            noise_level=noise_level,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            hotspot_mask=None,  # we don't support this yet
            struct_self_cond=(
                struct_self_cond
                if self.model.config.train.self_cond_train_prob > 0.5  # true for cc89
                else None
            ),
            struct_crop_cond=None,
            sse_cond=None,  # secondary structure conditioning, TODO maybe use?
            adj_cond=None,  # adjacency conditioning, # TODO not sure what this is for exactly
            seq_self_cond=None,  # we don't do sequence design.
            seq_crop_cond=None,
            run_mpnn_model=False,  # we don't want to do sequence design
        )

        # pass the self-conditioning to the next step by updating the features.
        # Detach it since we don't want the gradient flowing back to a previous step.
        # TODO: I wonder if we need to adjust this since we will apply additional guidance.
        features.conditioning.x_self_conditioning = x_self_cond.detach()

        # x0: [batch, L, 37, 3] -> flat [batch, atoms, 3]. Gather with the same
        # per-atom (residue, slot) maps used to scatter in, so the returned atoms
        # keep the original input order. A boolean-mask gather would instead emit
        # atoms in atom37-slot order and silently reorder them within a residue
        # (e.g. CB before O), breaking correspondence with the input structure.
        flat_coords = _convert_atom37_to_flat(x0, cond.atom37_residue_index, cond.atom37_atom_index)

        return flat_coords

    # This just generates the input noise. When doing partial diffusion, the
    # noise is added to the input coordinates with some weight..
    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[ProtpardelleConditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Sample reference coordinates from the prior distribution.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        features : GenerativeModelInput[ProtpardelleConditioning] | None
            Features as returned by :meth:`featurize`, used to infer the atom
            count when ``shape`` is not given.
        shape : tuple[int, ...] | None
            Explicit ``(num_atoms, 3)`` shape. Overrides ``features``.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Gaussian-initialized coordinates.

        Raises
        ------
        ValueError
            If neither ``features`` nor a valid ``shape`` is provided.
        """
        if shape is not None:
            if len(shape) != 2 or shape[1] != 3:
                raise ValueError("shape must be of the form (num_atoms, 3)")
            return torch.randn((batch_size, *shape), device=self.device)

        if features is None or features.conditioning is None:
            raise ValueError("Either features or shape must be provided to initialize_from_prior()")

        num_atoms = int(features.conditioning.atom_mask[0].sum().item())
        return torch.randn((batch_size, num_atoms, 3), device=self.device)
