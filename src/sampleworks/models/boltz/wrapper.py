"""
Wrapper classes for Boltz models.
Follows the protocol in model_wrapper_protocol.py
to allow dependency injection/interchangeable use in sampling pipelines.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from atomworks.enums import ChainType
from biotite.structure import AtomArray, AtomArrayStack, infer_elements
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.pad import pad_dim
from boltz.data.types import Manifest
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzDiffusionParams,
    BoltzProcessedInput,
    BoltzSteeringParams,
    check_inputs,
    MSAModuleArgs,
    PairformerArgs,
    PairformerArgsV2,
    process_inputs,
)
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2
from jaxtyping import Float
from loguru import logger
from torch import Tensor

from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.framework_utils import match_batch
from sampleworks.utils.msa import MSAManager
from sampleworks.utils.structure_utils import get_asym_unit_from_structure


def _decode_boltz_atom_name(raw_name: np.ndarray | str | bytes) -> str:
    """Decode a single atom name from a Boltz NPZ file.

    Boltz stores atom names in three formats depending on version and dtype:

    - **Unicode string** (Boltz2, dtype ``'U'``): returned as-is.
    - **Byte string** (dtype ``'S'``): UTF-8 decoded and stripped.
    - **int8 array** (Boltz1): fixed-width array where each byte encodes a
      character via ``chr(byte + 32)``.  Zero bytes are treated as null-
      terminator padding and skipped.  Since ``chr(0 + 32) == ' '``, this
      also strips any space characters embedded in the name.

    Parameters
    ----------
    raw_name : np.ndarray | str | bytes
        A single atom's name entry from ``atoms["name"]``.  For the int8
        case this is a 1-D ``int8`` array (one row of the structured column).

    Returns
    -------
    str
        Decoded atom name (e.g. ``"CA"``, ``"N"``, ``"OXT"``).
    """
    if isinstance(raw_name, (str, np.str_)):
        return str(raw_name)
    if isinstance(raw_name, (bytes, np.bytes_)):
        return bytes(raw_name).decode("utf-8").strip()
    # int8 encoding: 0 is null padding, non-zero bytes decode via chr(c + 32).
    return "".join(chr(int(c) + 32) for c in raw_name if c != 0)


def _decode_npz_names(raw_names: np.ndarray) -> np.ndarray:
    """Vectorized decode of a structured-array name column to unicode.

    Handles the three encodings found in Boltz NPZ files: Unicode
    strings (Boltz 2, dtype kind ``'U'``), byte strings (dtype kind
    ``'S'``), and object arrays.

    Parameters
    ----------
    raw_names : np.ndarray
        1-D array of name values from a Boltz structured array column
        (e.g. ``chains["name"]`` or ``residues["name"]``).

    Returns
    -------
    np.ndarray
        1-D unicode string array with the same length as *raw_names*.
    """
    if raw_names.dtype.kind == "U":
        return raw_names
    if raw_names.dtype.kind == "S":
        return np.char.decode(raw_names, "utf-8")
    return np.array([str(v) for v in raw_names])


def _resolve_boltz_residue_ids(residues: np.ndarray, npz_path: Path) -> np.ndarray:
    """Resolve residue identifiers from a Boltz residue table.

    Tries field names in priority order (``res_id``, ``auth_seq_id``,
    ``seq_id``, ``res_idx``) and returns the first one found, cast to
    ``int32``.

    Parameters
    ----------
    residues : np.ndarray
        Structured array from the ``residues`` key of a Boltz NPZ.
    npz_path : Path
        Source path, used only for error messages.

    Returns
    -------
    np.ndarray
        Integer residue IDs, shape ``(n_residues,)``.

    Raises
    ------
    ValueError
        If none of the expected field names are present.
    """
    fields = residues.dtype.names or ()
    for field_name in ("res_id", "auth_seq_id", "seq_id", "res_idx"):
        if field_name in fields:
            return np.asarray(residues[field_name], dtype=np.int32)

    raise ValueError(
        f"{npz_path}: residues table missing residue id field; available fields={fields}"
    )


def _atom_array_from_boltz_npz(npz_path: Path) -> AtomArray:
    """Build a biotite AtomArray from a Boltz processed ``.npz`` file.

    Handles both Boltz1 (int8-encoded atom names) and Boltz2 (string names).
    Validates required table fields and index monotonicity before mapping.

    Parameters
    ----------
    npz_path
        Path to the ``.npz`` in ``<out_dir>/processed/structures/``.

    Returns
    -------
    AtomArray
        One atom per model atom with chain_id, res_id, atom_name, res_name.
    """
    # Boltz processed NPZ files contain object/structured arrays that require
    # pickle support to deserialize. These files are generated by local Boltz
    # preprocessing artifacts.
    with np.load(npz_path, allow_pickle=True) as data:
        required_keys = {"atoms", "residues", "chains"}
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise ValueError(f"{npz_path}: missing required keys {sorted(missing_keys)}")

        atoms = data["atoms"]
        residues = data["residues"]
        chains = data["chains"]

    atom_fields = atoms.dtype.names or ()
    residue_fields = residues.dtype.names or ()
    chain_fields = chains.dtype.names or ()

    for field_name in ("coords", "name"):
        if field_name not in atom_fields:
            raise ValueError(f"{npz_path}: atoms table missing '{field_name}' field")
    for field_name in ("atom_idx", "name"):
        if field_name not in residue_fields:
            raise ValueError(f"{npz_path}: residues table missing '{field_name}' field")
    for field_name in ("res_idx", "name"):
        if field_name not in chain_fields:
            raise ValueError(f"{npz_path}: chains table missing '{field_name}' field")

    residue_atom_starts = np.asarray(residues["atom_idx"], dtype=np.int64)
    chain_residue_starts = np.asarray(chains["res_idx"], dtype=np.int64)
    if residue_atom_starts.size == 0:
        raise ValueError(f"{npz_path}: residues table is empty")
    if chain_residue_starts.size == 0:
        raise ValueError(f"{npz_path}: chains table is empty")
    if np.any(np.diff(residue_atom_starts) < 0):
        raise ValueError(f"{npz_path}: residues.atom_idx must be sorted ascending")
    if np.any(np.diff(chain_residue_starts) < 0):
        raise ValueError(f"{npz_path}: chains.res_idx must be sorted ascending")

    n_atoms = len(atoms)
    atom_indices = np.arange(n_atoms, dtype=np.int64)

    # Map each atom to its owning residue/chain via searchsorted on start indices.
    # searchsorted(side="right") - 1 finds the last start index <= each atom index.
    # The < 0 fires when the first residue/chain doesn't
    # start at index 0 e.g. a malformed Boltz NPZ.
    atom_res_idx = np.searchsorted(residue_atom_starts, atom_indices, side="right") - 1
    if np.any(atom_res_idx < 0):
        raise ValueError(f"{npz_path}: atom to residue index mapping out of bounds")

    res_chain_idx = np.searchsorted(chain_residue_starts, atom_res_idx, side="right") - 1
    if np.any((res_chain_idx < 0) | (res_chain_idx >= len(chains))):
        raise ValueError(f"{npz_path}: residue to chain index mapping out of bounds")

    residue_ids = _resolve_boltz_residue_ids(residues, npz_path)

    arr = AtomArray(n_atoms)
    arr.coord = np.asarray(atoms["coords"], dtype=np.float32)
    # Broadcast per-chain and per-residue properties to per-atom arrays using
    # the atom to residue index (atom_res_idx) and residue to chain (res_chain_idx) maps.
    arr.chain_id = _decode_npz_names(chains["name"])[res_chain_idx]
    arr.res_id = residue_ids[atom_res_idx]
    arr.res_name = _decode_npz_names(residues["name"])[atom_res_idx]
    # Atom names may be int8-encoded in Boltz1, fall back to element-wise decode.
    atom_names_raw = atoms["name"]
    if atom_names_raw.dtype.kind in ("U", "S"):
        arr.atom_name = _decode_npz_names(atom_names_raw)
    else:
        arr.atom_name = np.array(
            [_decode_boltz_atom_name(atom_names_raw[i]) for i in range(n_atoms)]
        )
    arr.element = infer_elements(arr)
    arr.set_annotation("occupancy", np.ones(n_atoms, dtype=np.float32))
    arr.set_annotation("b_factor", np.full(n_atoms, 20.0, dtype=np.float32))
    return arr


def _load_model_atom_array_from_structures_dir(structures_dir: Path) -> AtomArray | None:
    """Load a model atom array from a Boltz processed ``structures`` directory.

    Expects exactly one ``.npz`` file in *structures_dir*.  If zero or
    multiple files are found, or if parsing fails, a warning is logged and
    ``None`` is returned so callers can fall back gracefully.

    Parameters
    ----------
    structures_dir : Path
        Directory containing Boltz processed structure NPZ file(s).

    Returns
    -------
    AtomArray or None
        Parsed atom array, or ``None`` if loading was not possible.
    """
    npz_files = sorted(structures_dir.glob("*.npz"))
    if len(npz_files) == 0:
        logger.warning(
            f"Found no NPZ files in {structures_dir}; "
            "skipping model_atom_array (mismatch reconciliation disabled)"
        )
        return None

    if len(npz_files) > 1:
        logger.warning(
            f"Found {len(npz_files)} NPZ files in {structures_dir}; "
            "skipping model_atom_array (mismatch reconciliation disabled)"
        )
        return None

    npz_path = npz_files[0]
    try:
        return _atom_array_from_boltz_npz(npz_path)
    except (FileNotFoundError, ValueError, KeyError, TypeError, OSError) as exc:
        logger.warning(
            f"Failed to parse Boltz NPZ at {npz_path}: {exc}; "
            "skipping model_atom_array (mismatch reconciliation disabled)"
        )
        return None
    except Exception:
        raise


@dataclass(frozen=True, slots=True)
class BoltzConditioning:
    """Conditioning tensors from Boltz Pairformer pass.
    Passable to diffusion module forward.

    Attributes
    ----------
    s : Tensor
        Single representation, shape [batch, tokens, dim].
    z : Tensor
        Pair representation, shape [batch, tokens, tokens, dim].
    s_inputs : Tensor
        Input embeddings, shape [batch, tokens, dim].
    relative_position_encoding : Tensor
        Relative position encoding tensor.
    feats : dict[str, Any]
        Raw batch dict from dataloader.
    diffusion_conditioning : dict[str, Tensor] | None
        Additional diffusion conditioning (Boltz2 only).
    true_atom_array : AtomArray | None
        Original structure's atom array (occupancy-filtered) for reward computation.
        Boltz's internal atom representation may differ from this due to padding, etc.
    model_atom_array : AtomArray | None
        The AtomArray of the atoms the model actually operates on for the current
        sequence. May differ from ``true_atom_array`` (padding, added/missing atoms,
        element differences, etc.). Used for atom reconciliation against the true structure.
    """

    s: Tensor
    z: Tensor
    s_inputs: Tensor
    relative_position_encoding: Tensor
    feats: dict[str, Any]
    diffusion_conditioning: dict[str, Tensor] | None = None
    true_atom_array: AtomArray | None = None
    model_atom_array: AtomArray | None = None


@dataclass
class BoltzConfig:
    """Configuration for Boltz featurization.

    Attributes
    ----------
    out_dir : str | Path | None
        Output directory for processed Boltz files.
    num_workers : int
        Number of parallel workers for preprocessing.
    recycling_steps : int
        Number of recycling steps to perform during featurization Pairformer pass.

    """

    out_dir: str | Path | None = None
    num_workers: int = 0
    recycling_steps: int = 3


def process_structure_for_boltz(
    structure: dict,
    *,
    out_dir: str | Path | None = None,
    num_workers: int = 0,
    recycling_steps: int | None = 3,
) -> dict:
    """Annotate an Atomworks structure with Boltz-specific configuration.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary.
    out_dir : str | Path | None
        Output directory for processed Boltz files.
        Defaults to structure metadata ID or "boltz_output".
    num_workers : int
        Number of parallel workers for preprocessing.
    recycling_steps : int | None
        Number of recycling steps to perform during featurization Pairformer pass.
        Will set to 3 if None.

    Returns
    -------
    dict
        Structure dict with "_boltz_config" key added.
    """
    # Other models define a default deeper in their code,
    # but Boltz requires an integer value, so fix it here.
    if recycling_steps is None:
        recycling_steps = 3

    # Keep Boltz dataloading in-process by default. Kubernetes pods usually get
    # a small /dev/shm, and torch DataLoader workers can exhaust it while
    # sharing large featurized batches back to the parent process. This is
    # Boltz-specific because the Protenix/RF3 wrappers do not expose an
    # equivalent preprocessing worker pool here; callers can still pass
    # ``num_workers`` explicitly when profiling shows that multiprocessing is
    # worth the shared-memory tradeoff.
    config = BoltzConfig(
        out_dir=out_dir or structure.get("metadata", {}).get("id", "boltz_output"),
        num_workers=num_workers,
        recycling_steps=recycling_steps,
    )
    return {**structure, "_boltz_config": config}


@dataclass
class PredictArgs:
    """Arguments for model prediction. This is just used for compatibility with Boltz model init,
    not anywhere else"""

    recycling_steps: int = 3  # default in Boltz1
    sampling_steps: int = 200
    diffusion_samples: int = (
        1  # number of samples you want to generate, will be used as multiplicity
    )
    write_confidence_summary: bool = True
    write_full_pae: bool = False
    write_full_pde: bool = False


# TODO move this so that it can be imported without requiring a Boltz installation
def create_boltz_input_from_structure(
    structure: dict, out_dir: str | Path, msa_manager: MSAManager | None, msa_pairing_strategy: str
) -> Path:
    """Creates Boltz YAML file from an Atomworks parsed structure file.

    Parameters
    ----------
    structure: dict
        Atomworks parsed structure.
    out_dir: str | Path
        Path to write the YAML in.

    Returns
    -------
    Path
        Path to the written YAML file.
    """
    out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
    out_dir = out_dir.expanduser().resolve()

    polymer_info = {}
    ligand_info = {}

    chain_info = structure.get("chain_info", {})

    # Process atomworks chain_info into boltz YAML info
    # TODO: Handle non-canonical AAs in a polymer
    # TODO: Handle templates
    # TODO: Handle constraints
    for chain in chain_info:
        # Process protein, DNA, RNA chains
        entity_type: ChainType = chain_info[chain]["chain_type"]
        if entity_type.is_polymer():
            polymer_info[chain] = {}
            polymer_info[chain]["entity_type"] = (
                "protein"
                if entity_type.is_protein()
                else "DNA"
                if entity_type.is_nucleic_acid() and "DEOXY" in entity_type.to_string()
                else "RNA"
            )
            polymer_info[chain]["sequence"] = chain_info[chain][
                "processed_entity_canonical_sequence"
            ]  # Get canonical sequence
            if "CYCLIC" in entity_type.to_string():
                polymer_info[chain]["cyclic"] = True
        else:  # Ligand
            ligand_info[chain] = {}
            ligand_info[chain]["entity_type"] = "ligand"
            ligand_info[chain]["ccd"] = chain_info[chain]["res_name"][0]

    # get all the MSA paths, fetching MSAs as needed.
    # For homodimers (chains with identical sequences), Boltz requires all chains
    # with the same sequence to share the same MSA file. We deduplicate by sequence
    # and map all chains with identical sequences to the representative's MSA.
    if msa_manager:
        sequence_to_chains: dict[str, list[str]] = {}
        for chain_id, info in polymer_info.items():
            seq = info["sequence"]
            sequence_to_chains.setdefault(seq, []).append(chain_id)

        unique_chain_sequences = {chains[0]: seq for seq, chains in sequence_to_chains.items()}
        msa_paths_unique = msa_manager.get_msa(unique_chain_sequences, msa_pairing_strategy)

        msa_paths = {}
        for seq, chains_with_seq in sequence_to_chains.items():
            representative_msa = msa_paths_unique[chains_with_seq[0]]
            for chain_id in chains_with_seq:
                msa_paths[chain_id] = representative_msa
    else:
        msa_paths = None

    chains = {chain_id for chain_id, _ in polymer_info.items()}

    logger.debug(f"Should use msa_paths {msa_paths}")

    boltz_input_path = out_dir / f"{structure.get('metadata', {}).get('id', 'boltz_input')}.yaml"
    boltz_input_path.parent.mkdir(parents=True, exist_ok=True)
    with open(boltz_input_path, "w") as f:
        f.write("sequences:\n")
        for chain_id, info in polymer_info.items():
            f.write(f"    - {info['entity_type']}:\n")
            f.write(f"        id: {chain_id}\n")
            f.write(f"        sequence: {info['sequence']}\n")
            # If we have MSAs for all chains, then send paths to them.
            # If we do not include these lines in the config, Boltz will fetch the MSAs itself
            # so we should only add them if we have all MSAs available.
            if msa_paths and all(c in msa_paths for c in chains):
                f.write(f"        msa: {str(msa_paths[chain_id])}\n")
            elif msa_paths:
                logger.warning("Missing MSA for at least one chain, skipping.")
            if info.get("cyclic", False):
                f.write("        cyclic: true\n")
        if ligand_info:
            for chain_id, info in ligand_info.items():
                f.write(f"    - {info['entity_type']}:\n")
                f.write(f"        id: {chain_id}\n")
                f.write(f"        ccd: {info['ccd']}\n")

    return boltz_input_path


class Boltz2Wrapper:
    """
    Wrapper for Boltz2 model.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        use_msa_manager: bool = True,
        diffusion_args: Boltz2DiffusionParams = Boltz2DiffusionParams(),
        steering_args: BoltzSteeringParams = BoltzSteeringParams(),
        method: str = "MD",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model: Boltz2 | None = None,
    ):
        """
        Parameters
        ----------
        checkpoint_path: str | Path
            Filesystem path to the Boltz2 checkpoint containing trained weights.
        use_msa_manager: bool, optional
            If ``True``, fetch MSA features from the ColabFold server or cached values; otherwise
            rely on precomputed MSAs.  See sampleworks.utils.msa.MSAManager for details.
        diffusion_args: Boltz2DiffusionParams, optional
            Diffusion process parameters passed down to the Boltz2 model.
        steering_args: BoltzSteeringParams, optional
            Steering configuration controlling external potentials applied during
            sampling.
        method: str, optional
            Inference method identifier understood by Boltz2 (e.g. ``"MD"``).
        device: torch.device, optional
            Device to run the model on, by default CUDA if available.

        """
        self.checkpoint_path = checkpoint_path
        self.use_msa_manager = use_msa_manager
        self.diffusion_args = diffusion_args
        self.steering_args = steering_args
        self.method = method
        self.device = torch.device(device)
        # NOTE: assumes checkpoint and ccd dictionary get downloaded to the same place
        self.cache_path = Path(checkpoint_path).parent.expanduser().resolve()
        self.cache_path.mkdir(parents=True, exist_ok=True)

        pairformer_args = PairformerArgsV2()

        msa_args = MSAModuleArgs(
            subsample_msa=True,  # Default from boltz repo
            num_subsampled_msa=1024,  # Default from boltz repo
            use_paired_feature=True,  # Required for Boltz2
        )
        self.msa_manager = MSAManager() if use_msa_manager else None
        self.msa_pairing_strategy = "greedy"

        if not model:
            self.model = (
                Boltz2.load_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    strict=True,
                    predict_args=asdict(PredictArgs()),
                    map_location="cpu",
                    diffusion_process_args=asdict(diffusion_args),
                    ema=False,
                    pairformer_args=asdict(pairformer_args),
                    msa_args=asdict(msa_args),
                    steering_args=asdict(steering_args),
                )
                .to(self.device)
                .eval()
            )
        else:
            self.model = model.to(self.device).eval()

        self.data_module: Boltz2InferenceDataModule
        self.cached_representations: dict[str, Any] = {}

    def _setup_data_module(
        self,
        input_path: str | Path,
        out_dir: str | Path,
        num_workers: int = 0,
    ):
        """Create the Lightning data module used by Boltz to serve data to the model.

        NOTE: This will create processed .npz files, etc. in the out_dir.

        Parameters
        ----------
        input_path: str | Path
            Path to the input Boltz YAML file.
        out_dir: str | Path
            Directory to output processed input.
        num_workers: int, optional
            Number of parallel workers for input data processing, by default 8
        """
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        input_path = input_path.expanduser().resolve()
        ccd_path = self.cache_path / "ccd.pkl"
        mol_dir = self.cache_path / "mols"

        data = check_inputs(input_path)

        if self.msa_manager:
            msa_server_url = self.msa_manager.msa_server_url
        else:
            msa_server_url = "https://api.colabfold.com"

        process_inputs(  # <-- writes processed files to out_dir
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            mol_dir=mol_dir,
            use_msa_server=self.use_msa_manager,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=self.msa_pairing_strategy,
            boltz2=True,
            preprocessing_threads=1,
        )

        processed_dir = out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=Manifest.load(processed_dir / "manifest.json"),
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            constraints_dir=(processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None,
            template_dir=processed_dir / "templates"
            if (processed_dir / "templates").exists()
            else None,
            extra_mols_dir=processed_dir / "mols" if (processed_dir / "mols").exists() else None,
        )

        self._structures_dir = processed.targets_dir

        self.data_module = Boltz2InferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers if num_workers is not None else 0,
            constraints_dir=processed.constraints_dir,
            template_dir=processed_dir / "templates"
            if (processed_dir / "templates").exists()
            else None,
            extra_mols_dir=processed_dir / "mols" if (processed_dir / "mols").exists() else None,
            override_method=self.method,
        )

    def featurize(self, structure: dict) -> GenerativeModelInput[BoltzConditioning]:
        """From an Atomworks structure, calculate Boltz-2 input features.

        Runs the Pairformer pass to produce conditioning features.

        NOTE: Has side effect of creating Boltz input YAML and initial processed files with
        create_boltz_input_from_structure() and the data module setup.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary. Can be annotated with Boltz config
            via `process_structure_for_boltz()`. Config is read from
            `structure["_boltz_config"]` if present, otherwise default BoltzConfig
            values are used.

        Returns
        -------
        GenerativeModelInput[BoltzConditioning]
            Model input with Pairformer conditioning.
        """
        self.cached_representations.clear()

        config = structure.get("_boltz_config", BoltzConfig())
        if isinstance(config, dict):
            config = BoltzConfig(**config)

        # Use model-version-specific directory to avoid conflicts with Boltz1 preprocessed files
        base_dir = config.out_dir or structure.get("metadata", {}).get("id", "boltz2_output")
        out_dir = (
            f"{base_dir}_boltz2"
            if not str(base_dir).endswith(("boltz2", "boltz2_output"))
            else base_dir
        )
        num_workers = config.num_workers
        input_path = create_boltz_input_from_structure(  # side effect of creating Boltz input YAML
            structure,
            out_dir,
            msa_manager=self.msa_manager,
            msa_pairing_strategy=self.msa_pairing_strategy,
        )

        # side effect of creating "processed" files in out_dir
        self._setup_data_module(input_path, out_dir, num_workers=num_workers)

        batch = self.data_module.transfer_batch_to_device(
            next(iter(self.data_module.predict_dataloader())), self.device, 0
        )

        pairformer_out = self._pairformer_pass(batch, recycling_steps=config.recycling_steps)
        self.cached_representations = pairformer_out

        atom_array_or_stack = get_asym_unit_from_structure(structure)
        true_atom_array: AtomArray = (
            cast(AtomArray, atom_array_or_stack[0])
            if isinstance(atom_array_or_stack, AtomArrayStack)
            else atom_array_or_stack
        )

        # Build model atom array from processed NPZ for mismatch reconciliation
        model_atom_array = _load_model_atom_array_from_structures_dir(self._structures_dir)

        conditioning = BoltzConditioning(
            s=pairformer_out["s"],
            z=pairformer_out["z"],
            s_inputs=pairformer_out["s_inputs"],
            relative_position_encoding=pairformer_out["relative_position_encoding"],
            feats=pairformer_out["feats"],
            diffusion_conditioning=pairformer_out.get("diffusion_conditioning"),
            true_atom_array=true_atom_array,
            model_atom_array=model_atom_array,
        )

        return GenerativeModelInput(conditioning=conditioning)

    def _pairformer_pass(
        self, features: dict[str, Any], recycling_steps: int = 3
    ) -> dict[str, Any]:
        """Perform a pass through the Pairformer module. Refer to AlphaFold 3 paper for details.

        Internal method that computes Pairformer representations. Called by
        `featurize()` and cached for reuse across denoising steps.

        Basically copies Boltz code from [here](https://github.com/jwohlwend/boltz/blob/cb04aeccdd480fd4db707f0bbafde538397fa2ac/src/boltz/model/models/boltz2.py#L411-L530)
        without distogram and checkpointing logic.

        Parameters
        ----------
        features : dict[str, Any]
            Raw batch dict from dataloader.
        recycling_steps : int | None, optional
            Number of recycling steps to perform, by default 3.

        Returns
        -------
        dict[str, Any]
            Pairformer outputs (s, z, s_inputs, relative_position_encoding, feats).
            s refers to single representation (shape something like [batch, tokens, dim]) and z
            refers to pair representation (shape something like [batch, tokens, tokens, dim]).
        """
        mask: Tensor = features["token_pad_mask"]
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s_inputs = self.model.input_embedder(features)

        # Initialize single and pair representations
        s_init = self.model.s_init(s_inputs)
        z_init = (
            self.model.z_init_1(s_inputs)[:, :, None] + self.model.z_init_2(s_inputs)[:, None, :]
        )

        relative_position_encoding = self.model.rel_pos(features)

        z_init = z_init + relative_position_encoding
        z_init = z_init + self.model.token_bonds(features["token_bonds"].float())

        if self.model.bond_type_feature:
            z_init = z_init + self.model.token_bonds_type(features["type_bonds"].long())
        z_init = z_init + self.model.contact_conditioning(features)

        s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)

        for _ in range(recycling_steps):  # 3 is Boltz-2 default
            s = s_init + self.model.s_recycle(self.model.s_norm(s))
            z = z_init + self.model.z_recycle(self.model.z_norm(z))

            if self.model.use_templates:
                if self.model.is_template_compiled:
                    template_module = self.model.template_module._orig_mod
                else:
                    template_module = self.model.template_module

                z = z + template_module(z, features, pair_mask, use_kernels=self.model.use_kernels)

            if self.model.is_msa_compiled:
                msa_module = self.model.msa_module._orig_mod
            else:
                msa_module = self.model.msa_module

            z = z + msa_module(z, s_inputs, features, use_kernels=self.model.use_kernels)

            if self.model.is_pairformer_compiled:
                pairformer_module = cast(Any, self.model.pairformer_module)._orig_mod
            else:
                pairformer_module = self.model.pairformer_module

            s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask)

        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            self.model.diffusion_conditioning(
                s_trunk=s,
                z_trunk=z,
                relative_position_encoding=relative_position_encoding,
                feats=features,
            )
        )

        diffusion_conditioning = {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }

        return {
            "s": s,
            "z": z,
            "s_inputs": s_inputs,
            "relative_position_encoding": relative_position_encoding,
            "feats": features,
            "diffusion_conditioning": diffusion_conditioning,
        }

    def step(
        self,
        x_t: Float[Tensor, "batch atoms 3"],
        t: Float[Tensor, "*batch"] | float,
        *,
        features: GenerativeModelInput[BoltzConditioning] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        r"""Perform denoising at given timestep/noise level.

        Returns predicted clean sample :math:`\hat{x}_\theta`.

        Parameters
        ----------
        x_t : Float[Tensor, "batch atoms 3"]
            Noisy structure at timestep :math:`t`.
        t : Float[Tensor, "*batch"] | float
            Current timestep/noise level (:math:`\hat{t}` from EDM schedule).
        features : GenerativeModelInput[BoltzConditioning] | None
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

        # Ensure t has batch dimension and is on correct device
        if isinstance(t, (int, float)):
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.float32)
        else:
            t_tensor = t.to(device=self.device, dtype=torch.float32)
            if isinstance(t_tensor, torch.Tensor) and t_tensor.ndim == 0:
                t_tensor = t_tensor.unsqueeze(0)

        t_tensor = match_batch(t_tensor, target_batch_size=x_t.shape[0])

        feats = cond.feats
        atom_mask = feats.get("atom_pad_mask")  # shape [1, N_padded]
        atom_mask = cast(Tensor, atom_mask)
        atom_mask = atom_mask.repeat_interleave(x_t.shape[0], dim=0)  # shape [batch_size, N_padded]

        pad_len = atom_mask.shape[1] - x_t.shape[1]
        if pad_len >= 0:
            padded_x_t = pad_dim(x_t, dim=1, pad_len=pad_len)
        else:
            raise ValueError("pad_len is negative, cannot pad x_t")

        padded_atom_coords_denoised = self.model.structure_module.preconditioned_network_forward(
            padded_x_t,
            t_tensor,
            network_condition_kwargs=dict(
                multiplicity=padded_x_t.shape[0],
                s_inputs=cond.s_inputs,
                s_trunk=cond.s,
                feats=feats,
                diffusion_conditioning=cond.diffusion_conditioning,
            ),
        )

        atom_coords_denoised = padded_atom_coords_denoised[atom_mask.bool(), :].reshape(
            x_t.shape[0], -1, 3
        )

        return atom_coords_denoised

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[BoltzConditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Create a state drawn from the prior distribution. In the case of Boltz, this is Gaussian.

        Parameters
        ----------
        batch_size : int
            Number of noisy samples to generate.
        features: GenerativeModelInput[BoltzConditioning] | None, optional
            Model features as returned by `featurize`. Useful for determining shape, etc. for
            the state.
        shape: tuple[int, ...] | None, optional
            Explicit shape of the generated state (in the form [num_atoms, 3]), if features is None
            or does not provide shape info. NOTE: shape will override features if both are provided.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Gaussian initialized coordinates.

        Raises
        ------
        ValueError
            If both features and shape are None.
        """
        if shape is not None:
            if len(shape) != 2 or shape[1] != 3:
                raise ValueError("shape must be of the form (num_atoms, 3)")
            return torch.randn((batch_size, *shape), device=self.device)
        if features is None or features.conditioning is None:
            raise ValueError("Either features or shape must be provided to initialize_from_prior()")

        cond = features.conditioning
        feats = cond.feats
        atom_mask = feats.get("atom_pad_mask")
        atom_mask = cast(Tensor, atom_mask)

        num_atoms = int(atom_mask.sum())

        return torch.randn((batch_size, num_atoms, 3), device=self.device)


class Boltz1Wrapper:
    """Wrapper for Boltz1 model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        use_msa_manager: bool = True,
        diffusion_args: BoltzDiffusionParams = BoltzDiffusionParams(),
        steering_args: BoltzSteeringParams = BoltzSteeringParams(),
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model: Boltz1 | None = None,
    ):
        """
        Parameters
        ----------
        checkpoint_path: str | Path
            Filesystem path to the Boltz1 checkpoint containing trained weights.
        use_msa_manager: bool, optional
            If ``True``, fetch MSA features from the ColabFold server or cached values;
            otherwise rely on precomputed MSAs. See sampleworks.utils.msa.MSAManager for details.
        diffusion_args: BoltzDiffusionParams, optional
            Diffusion process parameters passed down to the Boltz1 model.
        steering_args: BoltzSteeringParams, optional
            Steering configuration controlling external potentials applied during
            sampling.
        device: torch.device, optional
            Device to run the model on, by default CUDA if available.

        """
        self.checkpoint_path = checkpoint_path
        self.use_msa_manager = use_msa_manager
        self.diffusion_args = diffusion_args
        self.steering_args = steering_args
        self.device = torch.device(device)
        # NOTE: assumes checkpoint and ccd dictionary get downloaded to the same place
        self.cache_path = Path(checkpoint_path).parent.expanduser().resolve()
        self.cache_path.mkdir(parents=True, exist_ok=True)

        pairformer_args = PairformerArgs()

        msa_args = MSAModuleArgs(
            subsample_msa=True,
            num_subsampled_msa=1024,
            use_paired_feature=False,
        )
        self.msa_manager = MSAManager() if self.use_msa_manager else None
        self.msa_pairing_strategy = "greedy"

        if not model:
            self.model = (
                Boltz1.load_from_checkpoint(
                    checkpoint_path,
                    strict=True,
                    predict_args=asdict(PredictArgs()),
                    map_location="cpu",
                    diffusion_process_args=asdict(diffusion_args),
                    ema=False,
                    use_kernels=True,
                    pairformer_args=asdict(pairformer_args),
                    msa_args=asdict(msa_args),
                    steering_args=asdict(steering_args),
                )
                .to(self.device)
                .eval()
            )
        else:
            self.model = model.to(self.device).eval()

        self.data_module: BoltzInferenceDataModule
        self.cached_representations: dict[str, Any] = {}

    def _setup_data_module(
        self,
        input_path: str | Path,
        out_dir: str | Path,
        num_workers: int = 0,
    ):
        """Create the Lightning data module used by Boltz to serve data to the model.

        Parameters
        ----------
        input_path: str | Path
            Path to the input Boltz YAML file.
        out_dir: str | Path
            Directory to output processed input.
        num_workers: int, optional
            Number of parallel workers for input data processing, by default 2
        """
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        input_path = input_path.expanduser().resolve()
        ccd_path = self.cache_path / "ccd.pkl"
        mol_dir = self.cache_path / "mols"

        data = check_inputs(input_path)

        if self.msa_manager:
            msa_server_url = self.msa_manager.msa_server_url
        else:
            msa_server_url = "https://api.colabfold.com"

        process_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            mol_dir=mol_dir,
            use_msa_server=self.use_msa_manager,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=self.msa_pairing_strategy,
            boltz2=False,
            preprocessing_threads=1,
        )

        processed_dir = out_dir / "processed"
        load_manifest = cast(Any, Manifest.load)
        processed = BoltzProcessedInput(
            manifest=load_manifest(processed_dir / "manifest.json"),
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            constraints_dir=(processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None,
            template_dir=processed_dir / "templates"
            if (processed_dir / "templates").exists()
            else None,
            extra_mols_dir=processed_dir / "mols" if (processed_dir / "mols").exists() else None,
        )

        self._structures_dir = processed.targets_dir

        self.data_module = BoltzInferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            num_workers=num_workers if num_workers is not None else 0,
            constraints_dir=processed.constraints_dir,
        )

    def featurize(self, structure: dict) -> GenerativeModelInput[BoltzConditioning]:
        """From an Atomworks structure, calculate Boltz-1 input features.

        Runs the Pairformer pass to produce conditioning features.

        NOTE: Has side effect of creating Boltz input YAML and initial processed files with
        create_boltz_input_from_structure() and the data module setup.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary. Can be annotated with Boltz config
            via `process_structure_for_boltz()`. Config is read from
            `structure["_boltz_config"]` if present, otherwise default BoltzConfig
            values are used.

        Returns
        -------
        GenerativeModelInput[BoltzConditioning]
            Model input with Pairformer conditioning.
        """
        self.cached_representations.clear()

        config = structure.get("_boltz_config", BoltzConfig())
        if isinstance(config, dict):
            config = BoltzConfig(**config)

        # Use model-version-specific directory to avoid conflicts with Boltz2 preprocessed files
        base_dir = config.out_dir or structure.get("metadata", {}).get("id", "boltz1_output")
        out_dir = (
            f"{base_dir}_boltz1"
            if not str(base_dir).endswith(("boltz1", "boltz1_output"))
            else base_dir
        )
        num_workers = config.num_workers
        input_path = create_boltz_input_from_structure(
            structure,
            out_dir,
            msa_manager=self.msa_manager,
            msa_pairing_strategy=self.msa_pairing_strategy,
        )

        self._setup_data_module(input_path, out_dir, num_workers=num_workers)

        batch = self.data_module.transfer_batch_to_device(
            next(iter(self.data_module.predict_dataloader())), self.device, 0
        )

        pairformer_out = self._pairformer_pass(batch, recycling_steps=config.recycling_steps)
        self.cached_representations = pairformer_out

        atom_array_or_stack = get_asym_unit_from_structure(structure)
        true_atom_array: AtomArray = (
            cast(AtomArray, atom_array_or_stack[0])
            if isinstance(atom_array_or_stack, AtomArrayStack)
            else atom_array_or_stack
        )

        # Build model atom array from processed NPZ for mismatch reconciliation
        model_atom_array = _load_model_atom_array_from_structures_dir(self._structures_dir)

        conditioning = BoltzConditioning(
            s=pairformer_out["s"],
            z=pairformer_out["z"],
            s_inputs=pairformer_out["s_inputs"],
            relative_position_encoding=pairformer_out["relative_position_encoding"],
            feats=pairformer_out["feats"],
            diffusion_conditioning=None,
            true_atom_array=true_atom_array,
            model_atom_array=model_atom_array,
        )

        return GenerativeModelInput(conditioning=conditioning)

    def step(
        self,
        x_t: Float[Tensor, "batch atoms 3"],
        t: Float[Tensor, "*batch"] | float,
        *,
        features: GenerativeModelInput[BoltzConditioning] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        r"""Perform denoising at given timestep/noise level.

        Returns predicted clean sample :math:`\hat{x}_\theta`.

        Parameters
        ----------
        x_t : Float[Tensor, "batch atoms 3"]
            Noisy structure at timestep :math:`t`.
        t : Float[Tensor, "*batch"] | float
            Current timestep/noise level (:math:`\hat{t}` from EDM schedule).
        features : GenerativeModelInput[BoltzConditioning] | None
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

        # Ensure t has batch dimension and is on correct device
        if isinstance(t, (int, float)):
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.float32)
        else:
            t_tensor = t.to(device=self.device, dtype=torch.float32)
            if isinstance(t_tensor, torch.Tensor) and t_tensor.ndim == 0:
                t_tensor = t_tensor.unsqueeze(0)

        t_tensor = match_batch(t_tensor, target_batch_size=x_t.shape[0])

        feats = cond.feats
        atom_mask = feats.get("atom_pad_mask")
        atom_mask = cast(Tensor, atom_mask)
        atom_mask = atom_mask.repeat_interleave(x_t.shape[0], dim=0)

        pad_len = atom_mask.shape[1] - x_t.shape[1]
        if pad_len >= 0:
            padded_x_t = pad_dim(x_t, dim=1, pad_len=pad_len)
        else:
            raise ValueError("pad_len is negative, cannot pad x_t")

        padded_atom_coords_denoised, _ = self.model.structure_module.preconditioned_network_forward(
            padded_x_t,
            t_tensor,
            training=False,
            network_condition_kwargs=dict(
                s_trunk=cond.s,
                z_trunk=cond.z,
                s_inputs=cond.s_inputs,
                feats=feats,
                relative_position_encoding=cond.relative_position_encoding,
                multiplicity=padded_x_t.shape[0],
            ),
        )

        atom_coords_denoised = padded_atom_coords_denoised[atom_mask.bool(), :].reshape(
            x_t.shape[0], -1, 3
        )

        return atom_coords_denoised

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[BoltzConditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Create a state drawn from the prior distribution. In the case of Boltz, this is Gaussian.

        Parameters
        ----------
        batch_size : int
            Number of noisy samples to generate.
        features: GenerativeModelInput[BoltzConditioning] | None, optional
            Model features as returned by `featurize`. Useful for determining shape, etc. for
            the state.
        shape: tuple[int, ...] | None, optional
            Explicit shape of the generated state (in the form [num_atoms, 3]), if features is None
            or does not provide shape info. NOTE: shape will override features if both are provided.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Gaussian initialized coordinates.

        Raises
        ------
        ValueError
            If both features and shape are None.
        """
        if shape is not None:
            if len(shape) != 2 or shape[1] != 3:
                raise ValueError("shape must be of the form (num_atoms, 3)")
            return torch.randn((batch_size, *shape), device=self.device)
        if features is None or features.conditioning is None:
            raise ValueError("Either features or shape must be provided to initialize_from_prior()")

        cond = features.conditioning
        feats = cond.feats
        atom_mask = feats.get("atom_pad_mask")
        atom_mask = cast(Tensor, atom_mask)

        num_atoms = int(atom_mask.sum())

        return torch.randn((batch_size, num_atoms, 3), device=self.device)

    def _pairformer_pass(
        self, features: dict[str, Any], recycling_steps: int = 3
    ) -> dict[str, Any]:
        """Perform a pass through the Pairformer module.

        Internal method that computes Pairformer representations. Called by
        `featurize()` and cached for reuse across denoising steps.

        Basically copies Boltz code from [here](https://github.com/jwohlwend/boltz/blob/cb04aeccdd480fd4db707f0bbafde538397fa2ac/src/boltz/model/models/boltz1.py#L285-340).

        Parameters
        ----------
        features : dict[str, Any]
            Raw batch dict from dataloader.
        recycling_steps : int | None, optional
            Number of recycling steps to perform, by default 3.

        Returns
        -------
        dict[str, Any]
            Pairformer outputs (s, z, s_inputs, relative_position_encoding, feats).
        """
        mask: Tensor = features["token_pad_mask"]
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s_inputs = self.model.input_embedder(features)

        s_init = self.model.s_init(s_inputs)
        z_init = (
            self.model.z_init_1(s_inputs)[:, :, None] + self.model.z_init_2(s_inputs)[:, None, :]
        )

        relative_position_encoding = self.model.rel_pos(features)
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.model.token_bonds(features["token_bonds"].float())

        s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)

        for _ in range(recycling_steps):  # 3 is Boltz-1 default
            s = s_init + self.model.s_recycle(self.model.s_norm(s))
            z = z_init + self.model.z_recycle(self.model.z_norm(z))

            if not self.model.no_msa:
                z = z + self.model.msa_module(
                    z, s_inputs, features, use_kernels=self.model.use_kernels
                )

            if self.model.is_pairformer_compiled:
                pairformer_module = cast(Any, self.model.pairformer_module)._orig_mod
            else:
                pairformer_module = self.model.pairformer_module

            s, z = pairformer_module(
                s,
                z,
                mask=mask,
                pair_mask=pair_mask,
                use_kernels=self.model.use_kernels,
            )

        return {
            "s": s,
            "z": z,
            "s_inputs": s_inputs,
            "relative_position_encoding": relative_position_encoding,
            "feats": features,
        }
