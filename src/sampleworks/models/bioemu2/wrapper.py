"""Wrapper for Microsoft BioEmu2, an all-atom EDM diffusion model.

BioEmu2 supplies the trunk (``embed``) and the denoiser (``denoise``); sampleworks'
``AF3EDMSampler`` drives the sampling loop. BioEmu2's own samplers, steering and
trajectory writers are not used. Importing ``bioemu2`` patches mdtraj process-wide,
so keep that import confined to this module.
"""

import itertools as it
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Literal

import mdtraj
import numpy as np
import torch
import torch.nn.functional as F
from atomworks.enums import ChainType
from bioemu2 import DNA, Ligand, load_model, MSA, Protein, RNA, System, TrunkEmbeddings
from bioemu2.biomoa.models.score_model import BioMoaCModel
from bioemu2.biomoa.modules.diffusion import StaticContext
from bioemu2.msa.alignment import get_msa, load_a3m_msa
from bioemu2.structure import BatchAllAtomStructure
from bioemu2.structure.preparation import prepare_sampling_inputs
from biotite.structure import AtomArray, AtomArrayStack
from jaxtyping import Float
from loguru import logger
from pydantic import ValidationError
from torch import Tensor

from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.framework_utils import match_batch


SamplingMode = Literal["equilibrium", "structure"]

_POLYMER_ENTITIES = {
    ChainType.POLYPEPTIDE_L: Protein,
    ChainType.RNA: RNA,
    ChainType.DNA: DNA,
}

_DEFAULT_B_FACTOR = 20.0


@dataclass(frozen=True)
class BioEmu2Config:
    """Featurization settings for BioEmu2, attached to a structure dict.

    Attributes
    ----------
    mode : {"equilibrium", "structure"}
        Sampling mode baked into the trunk embeddings. ``"equilibrium"`` samples the
        conformational ensemble; ``"structure"`` predicts a single folded state.
    msa_path : str | Path | dict | None
        A3M file for every protein chain, a ``{chain_id: path}`` mapping, or a JSON
        file holding that mapping. ``None`` fetches MSAs from the ColabFold server.
    use_msa_server : bool
        When False and no ``msa_path`` is given, use single-sequence MSAs
        (offline, lower quality).
    include_ligands : bool
        Model non-polymer, non-water chains as CCD ligands.
    """

    mode: SamplingMode = "equilibrium"
    msa_path: str | Path | dict | None = None
    use_msa_server: bool = True
    include_ligands: bool = True


@dataclass(frozen=True, slots=True)
class BioEmu2Conditioning:
    """Trunk embeddings and static sampling inputs for one BioEmu2 system.

    Attributes
    ----------
    s_inputs : Tensor
        Input embeddings, shape ``(1, tokens, s_input_dim)``.
    single : Tensor
        Trunk single representation, shape ``(1, tokens, token_s)``.
    pair : Tensor
        Trunk pair representation, shape ``(1, tokens, tokens, token_z)``.
    relative_position_encoding : Tensor
        Relative-position features, shape ``(1, tokens, tokens, token_z)``.
    distogram : Tensor
        Distogram logits, shape ``(1, tokens, tokens, bins)``.
    static_context : StaticContext
        Coordinate-independent structure-module inputs computed from the embeddings.
    static_context_source : tuple[Tensor, Tensor]
        The ``pair`` and ``relative_position_encoding`` tensors ``static_context`` was built from.
        :meth:`BioEmu2Wrapper.step` rebuilds the context when either has been replaced.
    batch : BatchAllAtomStructure
        Model input structure, batch size 1, atoms right-padded to a multiple of 32.
    num_atoms : int
        Number of real atoms.
    num_padded_atoms : int
        Number of atoms including padding, as the denoiser expects.
    topology : mdtraj.Topology
        BioEmu2's atom topology; atom ``i`` is coordinate row ``i``.
    model_atom_array : AtomArray
        Atoms the model operates on, in model order, for atom reconciliation.
    true_atom_array : AtomArray
        Input structure atoms.
    mode : str
        Sampling mode the embeddings were computed for.
    """

    s_inputs: Tensor
    single: Tensor
    pair: Tensor
    relative_position_encoding: Tensor
    distogram: Tensor
    static_context: StaticContext
    static_context_source: tuple[Tensor, Tensor]
    batch: BatchAllAtomStructure
    num_atoms: int
    num_padded_atoms: int
    topology: mdtraj.Topology
    model_atom_array: AtomArray
    true_atom_array: AtomArray
    mode: str

    @property
    def trunk_embeddings(self) -> TrunkEmbeddings:
        """Reassemble BioEmu2's ``TrunkEmbeddings`` from the stored tensors."""
        return TrunkEmbeddings(
            s_inputs=self.s_inputs,
            single=self.single,
            pair=self.pair,
            relative_position_encoding=self.relative_position_encoding,
            distogram=self.distogram,
        )


def annotate_structure_for_bioemu2(
    structure: dict,
    *,
    mode: SamplingMode = "equilibrium",
    msa_path: str | Path | dict | None = None,
    use_msa_server: bool = True,
    include_ligands: bool = True,
) -> dict:
    """Attach a :class:`BioEmu2Config` to an Atomworks structure.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary.
    mode : {"equilibrium", "structure"}
        Sampling mode.
    msa_path : str | Path | dict | None
        MSA specification, see :class:`BioEmu2Config`.
    use_msa_server : bool
        Whether to query ColabFold when no ``msa_path`` is given.
    include_ligands : bool
        Whether to model non-polymer chains as ligands.

    Returns
    -------
    dict
        Copy of ``structure`` with a ``"_bioemu2_config"`` entry.
    """
    config = BioEmu2Config(
        mode=mode,
        msa_path=msa_path,
        use_msa_server=use_msa_server,
        include_ligands=include_ligands,
    )
    return {**structure, "_bioemu2_config": config}


def ligand_residues(structure: dict) -> dict[str, tuple[str, int, str]]:
    """Assign a BioEmu2 ``asym_id`` to every ligand residue in ``chain_info``.

    BioEmu2 ligands are single CCD components, but PDB files often put several
    hetero groups in one chain. A one-residue chain keeps its chain ID; each residue
    of a longer chain gets ``<chain_id><n>`` (or ``L<n>``, if that is taken or longer
    than BioEmu2's four-character limit).

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary with a ``chain_info`` mapping.

    Returns
    -------
    dict[str, tuple[str, int, str]]
        ``asym_id -> (input chain_id, res_id, CCD code)``, in ``chain_info`` order.
    """
    chain_info = structure["chain_info"]
    taken = set(chain_info)
    assigned: dict[str, tuple[str, int, str]] = {}
    for chain_id, info in chain_info.items():
        if info["chain_type"] != ChainType.NON_POLYMER:
            continue
        res_names = [str(name) for name in info.get("res_name", [])]
        res_ids = [int(r) for r in info.get("res_id", [])] or list(range(1, len(res_names) + 1))
        if len(res_names) == 1:
            assigned[chain_id] = (chain_id, res_ids[0], res_names[0])
            continue
        for res_id, res_name in zip(res_ids, res_names):
            candidates = it.chain(
                (f"{chain_id}{n}" for n in range(1, 1_000)), (f"L{n}" for n in range(1, 1_000))
            )
            asym_id = next(c for c in candidates if len(c) <= 4 and c not in taken)
            taken.add(asym_id)
            assigned[asym_id] = (chain_id, res_id, res_name)
    return assigned


def system_from_structure(structure: dict, *, include_ligands: bool = True) -> System:
    """Build a BioEmu2 ``System`` from an Atomworks structure's ``chain_info``.

    Polymer chains use ``processed_entity_canonical_sequence``, so a sequence override
    applied with ``apply_sequence_override`` carries through and modified residues
    (e.g. MSE) become their canonical parents. Every non-polymer residue becomes a CCD
    ligand (see :func:`ligand_residues` for its ``asym_id``); ligands BioEmu2 cannot
    build, such as residues missing from the CCD, are skipped with a warning. Waters
    are dropped.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary with a ``chain_info`` mapping.
    include_ligands : bool
        Whether to include non-polymer chains.

    Returns
    -------
    System
        BioEmu2 system whose polymer ``asym_ids`` are the input chain IDs.

    Raises
    ------
    ValueError
        If ``chain_info`` is missing or a chain type is unsupported.
    """
    chain_info = structure.get("chain_info")
    if not chain_info:
        raise ValueError("BioEmu2 featurization requires structure['chain_info']")

    entities = []
    for chain_id, info in chain_info.items():
        chain_type = info["chain_type"]
        if chain_type in _POLYMER_ENTITIES:
            sequence = info["processed_entity_canonical_sequence"]
            entities.append(_POLYMER_ENTITIES[chain_type](sequence=sequence, asym_ids=[chain_id]))
        elif chain_type not in (ChainType.WATER, ChainType.NON_POLYMER):
            raise ValueError(f"BioEmu2 does not support chain {chain_id} of type {chain_type!r}")

    if include_ligands:
        for asym_id, (chain_id, res_id, ccd_code) in ligand_residues(structure).items():
            try:
                entities.append(Ligand(ccd_code=ccd_code, asym_ids=[asym_id]))
            except ValidationError as error:
                # e.g. atomworks' "~"-prefixed names for residues missing from the CCD.
                logger.warning(
                    f"Skipping ligand {ccd_code} (chain {chain_id}, residue {res_id}): "
                    f"BioEmu2 rejected it: {error.errors()[0]['msg']}"
                )

    if not entities:
        raise ValueError("Structure has no chains BioEmu2 can model")
    return System(entities=entities)


def model_atom_array_from_topology(topology: mdtraj.Topology, structure: dict) -> AtomArray:
    """Build the AtomArray of atoms BioEmu2 operates on, in model order.

    Polymer residues are renumbered from the structure's ``chain_info`` residue IDs
    when they cover the chain's sequence, otherwise from 1. Ligands go back to their
    input chain and residue ID, so a chain split by :func:`ligand_residues` reads as
    the original chain.

    Parameters
    ----------
    topology : mdtraj.Topology
        Topology from ``prepare_sampling_inputs``, built with the input chain IDs.
    structure : dict
        Atomworks structure dictionary the system was built from.

    Returns
    -------
    AtomArray
        Atoms with chain, residue, atom and element annotations, zero coordinates,
        occupancy 1 and B-factor 20.
    """
    chain_info = structure["chain_info"]
    ligands = ligand_residues(structure)

    # topology residue index -> (output chain_id, output res_id)
    residue_label: dict[int, tuple[str, int]] = {}
    for chain in topology.chains:
        residues = list(chain.residues)
        if chain.chain_id in ligands:
            input_chain, res_id, _ = ligands[chain.chain_id]
            for residue in residues:
                residue_label[residue.index] = (input_chain, res_id)
            continue
        listed_ids = [int(r) for r in chain_info[chain.chain_id].get("res_id", [])]
        use_listed = len(listed_ids) == len(residues)
        for position, residue in enumerate(residues):
            res_id = listed_ids[position] if use_listed else position + 1
            residue_label[residue.index] = (chain.chain_id, res_id)

    atoms = list(topology.atoms)
    labels = [residue_label[atom.residue.index] for atom in atoms]
    atom_array = AtomArray(len(atoms))
    atom_array.chain_id = np.array([chain_id for chain_id, _ in labels])
    atom_array.res_id = np.array([res_id for _, res_id in labels], dtype=int)
    atom_array.res_name = np.array([atom.residue.name for atom in atoms])
    atom_array.atom_name = np.array([atom.name for atom in atoms])
    atom_array.element = np.array([atom.element.symbol.upper() for atom in atoms])
    atom_array.hetero = np.array([atom.residue.chain.chain_id in ligands for atom in atoms])
    atom_array.coord = np.zeros((len(atoms), 3), dtype=np.float32)
    atom_array.set_annotation("occupancy", np.ones(len(atoms), dtype=np.float32))
    atom_array.set_annotation("b_factor", np.full(len(atoms), _DEFAULT_B_FACTOR, dtype=np.float32))
    return atom_array


def resolve_msa(system: System, config: BioEmu2Config) -> MSA:
    """Get the MSA BioEmu2's trunk needs for the system's protein chains.

    Parameters
    ----------
    system : System
        BioEmu2 system.
    config : BioEmu2Config
        Supplies ``msa_path`` and ``use_msa_server``.

    Returns
    -------
    MSA
        Alignments loaded from ``msa_path``, fetched from ColabFold (cached under
        ``BIOEMU2_MSA_CACHE``), or single-sequence alignments when offline.
    """
    msa_path = config.msa_path
    if isinstance(msa_path, (str, Path)) and Path(msa_path).suffix == ".json":
        msa_path = json.loads(Path(msa_path).read_text())

    if isinstance(msa_path, dict):
        return load_a3m_msa(system, {chain: Path(path) for chain, path in msa_path.items()})
    if msa_path is not None:
        return get_msa(system, source=Path(msa_path))
    if config.use_msa_server:
        return get_msa(system)

    return MSA(
        {
            chain_id: f">query\n{entity.sequence}\n"
            for entity in system.entities
            if isinstance(entity, Protein)
            for chain_id in entity.asym_ids
        }
    )


class BioEmu2Wrapper:
    """Wrapper for Microsoft BioEmu2 as a sampleworks ``FlowModelWrapper``."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | str | None = None,
        precision: str = "bf16-mixed",
        model=None,
    ):
        """
        Parameters
        ----------
        checkpoint_path : str | Path
            BioEmu2 ``checkpoint.ckpt`` with its ``config.yaml`` in the same directory.
        device : torch.device | str | None
            Inference device. Defaults to CUDA when available.
        precision : str
            Trunk precision policy passed to ``bioemu2.load_model``. The denoiser
            always runs in float32.
        model : optional
            A model loaded by another ``BioEmu2Wrapper``, reused instead of loading
            the checkpoint again.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if model is None:
            logger.info(f"Loading BioEmu2 checkpoint {self.checkpoint_path}")
            # TF32 matmuls are only available (and only accepted by load_model) on CUDA.
            matmul_precision = "high" if self._device.type == "cuda" else None
            model = load_model(
                checkpoint_path=self.checkpoint_path,
                device=self._device,
                precision=precision,
                matmul_precision=matmul_precision,
            )
        self.model = model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def edm_schedule(self):
        """The checkpoint's EDM schedule (``sigma_data``, ``sigma_min``, ``sigma_max``)."""
        return self.model.edm_schedule

    def featurize(self, structure: dict) -> GenerativeModelInput[BioEmu2Conditioning]:
        """Run BioEmu2's trunk and build the static denoiser inputs.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary, optionally annotated with
            :func:`annotate_structure_for_bioemu2`.

        Returns
        -------
        GenerativeModelInput[BioEmu2Conditioning]
            Conditioning for :meth:`step`.
        """
        config = structure.get("_bioemu2_config", BioEmu2Config())
        if "asym_unit" not in structure:
            raise ValueError("structure must contain 'asym_unit' key")

        system = system_from_structure(structure, include_ligands=config.include_ligands)
        msa = resolve_msa(system, config)
        embeddings = self.model.embed(system, msa=msa, mode=config.mode)
        topology, batch = prepare_sampling_inputs(system, device=self.device)
        # Single-device inference always yields a plain StaticContext.
        static_context = cast(
            StaticContext,
            self.model.compute_static_context(batch, embeddings, multiplicity=1),
        )

        num_atoms = topology.n_atoms
        num_padded_atoms = batch.coords.shape[1]
        if not bool(batch.atom_mask[0, :num_atoms].all()) or bool(
            batch.atom_mask[0, num_atoms:].any()
        ):
            raise ValueError("BioEmu2 atom padding is not a contiguous tail; cannot slice atoms")

        asym_unit = structure["asym_unit"]
        true_atom_array = asym_unit[0] if isinstance(asym_unit, AtomArrayStack) else asym_unit

        conditioning = BioEmu2Conditioning(
            s_inputs=embeddings.s_inputs,
            single=embeddings.single,
            pair=embeddings.pair,
            relative_position_encoding=embeddings.relative_position_encoding,
            distogram=embeddings.distogram,
            static_context=static_context,
            static_context_source=(embeddings.pair, embeddings.relative_position_encoding),
            batch=batch,
            num_atoms=num_atoms,
            num_padded_atoms=num_padded_atoms,
            topology=topology,
            model_atom_array=model_atom_array_from_topology(topology, structure),
            true_atom_array=true_atom_array,
            mode=config.mode,
        )
        return GenerativeModelInput(conditioning=conditioning)

    def step(
        self,
        x_t: Float[Tensor, "batch atoms 3"],
        t: Float[Tensor, "*batch"] | float,
        *,
        features: GenerativeModelInput[BioEmu2Conditioning] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        r"""Denoise ``x_t`` at noise level ``t`` and return :math:`\hat{x}_\theta`.

        Keeps the autograd graph: gradients reach ``x_t`` and the trunk embeddings.

        Parameters
        ----------
        x_t : Float[Tensor, "batch atoms 3"]
            Noisy coordinates in Å over the real (unpadded) atoms.
        t : Float[Tensor, "*batch"] | float
            Noise level :math:`\sigma` in Å, scalar or one per sample.
        features : GenerativeModelInput[BioEmu2Conditioning]
            Output of :meth:`featurize`.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Denoised coordinates in Å.
        """
        if features is None or features.conditioning is None:
            raise ValueError("features with conditioning required for step()")
        cond = features.conditioning

        batch_size = x_t.shape[0]
        # (batch, num_atoms, 3) -> (batch, num_padded_atoms, 3); padding rows are zeros.
        x_padded = F.pad(x_t, (0, 0, 0, cond.num_padded_atoms - cond.num_atoms))

        sigmas = torch.as_tensor(t, device=x_t.device, dtype=torch.float32).reshape(-1)
        sigmas = match_batch(sigmas, target_batch_size=batch_size)

        embeddings = cond.trunk_embeddings
        static_context = cond.static_context
        # The pair and relative-position embeddings enter only through the static context.
        # IT-opt swaps in optimized copies, so rebuild it whenever they are not the tensors it
        # was built from (an identity check, so the final no-grad pass sees them too).
        source_pair, source_relpos = cond.static_context_source
        if cond.pair is not source_pair or cond.relative_position_encoding is not source_relpos:
            # load_model's wrapper computes the context under no_grad; its score model does not.
            score_model = cast(BioMoaCModel, getattr(self.model, "score_model"))
            static_context = score_model.compute_static_context(cond.batch, embeddings)

        denoised = self.model.denoise(
            static_context, embeddings, x_padded, sigmas
        ).denoised_coords  # (batch, num_padded_atoms, 3)
        return denoised[:, : cond.num_atoms].float()

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[BioEmu2Conditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Sample unit Gaussian noise; the sampler's first step scales it.

        Parameters
        ----------
        batch_size : int
            Number of samples.
        features : GenerativeModelInput[BioEmu2Conditioning] | None
            Supplies the atom count.
        shape : tuple[int, ...] | None
            ``(num_atoms, 3)``; overrides ``features`` when given.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Standard normal coordinates.
        """
        if shape is not None:
            if len(shape) != 2 or shape[1] != 3:
                raise ValueError("shape must be of the form (num_atoms, 3)")
            return torch.randn((batch_size, *shape), device=self.device)
        if features is None or features.conditioning is None:
            raise ValueError("Either features or shape must be provided to initialize_from_prior()")
        return torch.randn((batch_size, features.conditioning.num_atoms, 3), device=self.device)
