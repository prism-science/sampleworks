import os
from collections.abc import Mapping
from hashlib import sha3_256
from pathlib import Path

from loguru import logger

from sampleworks.utils.guidance_constants import StructurePredictor
from sampleworks.utils.imports import PROTENIX_AVAILABLE
from sampleworks.utils.mmseqs2 import run_mmseqs2


if PROTENIX_AVAILABLE:
    from runner.msa_search import msa_search as protenix_msa_search

    logger.debug("Protenix MSA tools are available at top of msa.py")
else:
    logger.error("Protenix is not installed, cannot use Protenix MSA tools.")


MAX_PAIRED_SEQS = 8192
MAX_MSA_SEQS = 16384
MSA_CACHE_DIR_ENV_VAR = "MSA_CACHE_DIR"
MSAData = Mapping[str, str] | Mapping[int, str] | Mapping[str | int, str]


def _resolve_msa_cache_dir(msa_cache_dir: Path | str | None) -> Path:
    """Choose the MSA cache directory.

    Parameters
    ----------
    msa_cache_dir : Path | str | None
        Directory requested by the caller. When None, the ``MSA_CACHE_DIR``
        environment variable is used if set and non-empty, else ``~/.sampleworks/msa``.

    Returns
    -------
    Path
        The cache directory, with ``~`` expanded.
    """
    if msa_cache_dir is None:
        msa_cache_dir = os.environ.get(MSA_CACHE_DIR_ENV_VAR, "").strip() or None
    if msa_cache_dir is None:
        return Path.home() / ".sampleworks" / "msa"
    return Path(msa_cache_dir).expanduser()


def _msa_data_key_sort_key(key: str | int) -> tuple[int, int | str]:
    """Return a deterministic sort key for supported MSA sequence keys."""
    if isinstance(key, int):
        return (0, key)
    return (1, key)


def _validate_msa_cache_contents(msa_hash: str, msa_dir: Path) -> None:
    """Validate the contents of the MSA cache.
    For each hash, there should be a set of files `"{hash}_{n}.csv", "{hash}_{n}.a3m"`
    where `n` is the index of the sequence in the input data, 0, 1, ..., len(sequences) - 1.
    In most cases there is only one sequence, so `n = 0` only. The .a3m files should have a sequence
    on every other line. The .csv files have two columns. The sequences in the second column of the
    .csv file should be the same as the sequences in the a3m file.

    Parameters
    ----------
    msa_hash : str
        The hash key for the MSA files.
    msa_dir : Path
        The directory containing the MSA cache files.

    Raises
    ------
    FileNotFoundError
        If expected files are missing.
    ValueError
        If file contents don't match or are invalid.
    """
    # Find all files matching the hash pattern
    csv_files = sorted(msa_dir.glob(f"{msa_hash}_*.csv"))
    a3m_files = sorted(msa_dir.glob(f"{msa_hash}_*.a3m"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for hash {msa_hash} in {msa_dir}")
    if not a3m_files:
        raise FileNotFoundError(f"No A3M files found for hash {msa_hash} in {msa_dir}")

    # Validate that we have matching pairs
    csv_indices = {int(f.stem.split("_")[-1]) for f in csv_files}
    a3m_indices = {int(f.stem.split("_")[-1]) for f in a3m_files}

    if csv_indices != a3m_indices:
        raise ValueError(
            f"Mismatch between CSV and A3M file indices. "
            f"CSV: {sorted(csv_indices)}, A3M: {sorted(a3m_indices)}"
        )

    # Validate each pair of files
    for idx in sorted(csv_indices):
        csv_path = msa_dir / f"{msa_hash}_{idx}.csv"
        a3m_path = msa_dir / f"{msa_hash}_{idx}.a3m"

        # Read CSV sequences (skip header, take second column)
        with csv_path.open("r") as f:
            csv_lines = f.readlines()

        if not csv_lines or csv_lines[0].strip() != "key,sequence":
            raise ValueError(f"Invalid CSV header in {csv_path}")

        csv_sequences = [line.strip().split(",", 1)[1] for line in csv_lines[1:] if line.strip()]

        # Read A3M sequences (every other line, skipping headers)
        with a3m_path.open("r") as f:
            a3m_lines = f.readlines()

        # A3M format: header lines start with '>', sequences on alternating lines
        a3m_sequences = [line.strip() for i, line in enumerate(a3m_lines) if i % 2 == 1]

        # Validate that sequences match
        if len(csv_sequences) != len(a3m_sequences):
            raise ValueError(
                f"Sequence count mismatch for index {idx}: "
                f"CSV has {len(csv_sequences)}, A3M has {len(a3m_sequences)}"
            )

        for seq_idx, (csv_seq, a3m_seq) in enumerate(zip(csv_sequences, a3m_sequences)):
            if csv_seq != a3m_seq:
                raise ValueError(
                    f"Sequence mismatch at index {idx}, sequence {seq_idx}: "
                    f"CSV and A3M sequences differ"
                )

        logger.debug(f"Validated {len(csv_sequences)} sequences for hash {msa_hash}, index {idx}")


# From https://github.com/jwohlwend/boltz/blob/main/src/boltz/main.py#L415
# FIXME: this function is a big, long mess. We should clean it up.
#   For the love of decent code, don't copy this and use it somewhere else and respect the
#   leading underscore!
def _compute_msa(
    data: MSAData,
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    api_key_header: str | None = None,
    api_key_value: str | None = None,
) -> dict[str | int, Path]:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : MSAData
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.
    msa_server_username : str, optional
        Username for basic authentication with MSA server.
    msa_server_password : str, optional
        Password for basic authentication with MSA server.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).

    Returns
    -------
    dict[str | int, Path]
        A dictionary mapping target names (keys of input data dict) to MSA file paths.

    """
    logger.info(f"Calling MSA server for target {target_id} with {len(data)} sequences")
    logger.info(f"MSA server URL: {msa_server_url}")
    logger.info(f"MSA pairing strategy: {msa_pairing_strategy}")

    # Construct auth headers if API key header/value is provided
    auth_headers = None
    if api_key_value:
        key = api_key_header if api_key_header else "X-API-Key"
        value = api_key_value
        auth_headers = {"Content-Type": "application/json", key: value}
        logger.info(f"Using API key authentication for MSA server (header: {key})")
    elif msa_server_username and msa_server_password:
        logger.info("Using basic authentication for MSA server")
    else:
        logger.info("No authentication provided for MSA server")

    # NB: this code is borrowed from Boltz, and it appears to ignore the
    # pairing argument used elsewhere. It also relies on dicts being ordered now
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            str(msa_dir / f"{target_id}_paired_tmp"),
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            auth_headers=auth_headers,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        str(msa_dir / f"{target_id}_unpaired_tmp"),
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        auth_headers=auth_headers,
    )

    # FIXME: this code relies on the fact that run_mmseqs2 returns sequence alignments in the same
    #  order as they are in `data`, and furthermore just returns a list of strings, the content
    #  of each string being a single sequence alignment. It's some weird file parsing that we
    #  should clean up so users don't break it or have to worry about it.
    outputs: dict[str | int, Path] = {}
    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[:MAX_PAIRED_SEQS]

        # Set key per row and remove empty sequences
        keys = [pair_idx for pair_idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (MAX_MSA_SEQS - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{target_id}_{idx}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))

        """ Write out a3m format for RF3 """
        # in addition to CSV write a3m (FASTA) formats--but omit the alignment stats
        with msa_path.with_suffix(".a3m").open("w") as f:
            for seq_idx, seq in enumerate(seqs):
                f.write(f">{target_id}_{idx}_{seq_idx}\n{seq}\n")

        outputs[name] = msa_path

    return outputs


class MSAManager:
    """
    Manages multiple sequence alignment (MSA) operations, including caching,
    API requests, and file management.

    This class facilitates interacting with an external MSA server for sequence
    alignments while leveraging local caching for efficiency. It computes MSA
    results based on input data and pairing strategy, while transparently
    retrieving cached results when available.

    Parameters
    ----------
    msa_cache_dir : Path | str | None
        Where to cache MSAs; created if missing. When None, the ``MSA_CACHE_DIR``
        environment variable is used if set, else ``~/.sampleworks/msa``.
    msa_server_url : str
        Base URL of the MMseqs2 API server.
    msa_server_username : str | None
        Username for basic authentication with the MSA server.
    msa_server_password : str | None
        Password for basic authentication with the MSA server.
    api_key_header : str | None
        Header name for API key authentication (default ``X-API-Key``).
    api_key_value : str | None
        API key for header authentication.
    """

    def __init__(
        self,
        msa_cache_dir: Path | str | None = None,
        msa_server_url: str = "https://api.colabfold.com",
        msa_server_username: str | None = None,
        msa_server_password: str | None = None,
        api_key_header: str | None = None,
        api_key_value: str | None = None,
    ):
        self.msa_dir = _resolve_msa_cache_dir(msa_cache_dir)
        self.msa_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using MSA cache directory {self.msa_dir}")
        self.msa_server_url = msa_server_url
        self.msa_server_username = msa_server_username
        self.msa_server_password = msa_server_password
        self.api_key_header = api_key_header
        self.api_key_value = api_key_value

        self._api_calls = 0
        self._cache_hits = 0

    @staticmethod
    def _hash_arguments(data: MSAData, msa_pairing_strategy: str) -> str:
        encoded_sequence_tuple = str.encode(str(tuple(data.values())) + msa_pairing_strategy)
        hexdigest = sha3_256(encoded_sequence_tuple).hexdigest()
        return hexdigest

    def get_msa(
        self,
        data: MSAData,
        msa_pairing_strategy: str,
        structure_predictor: str | StructurePredictor = StructurePredictor.BOLTZ_2,
    ) -> dict[str | int, Path]:
        """Fetches existing MSA files from disk or computes new ones if necessary.

        Parameters
        ----------
        data : MSAData
            A dictionary mapping target (usu. chain or index) names to protein sequences.
        msa_pairing_strategy : str
            The MSA pairing strategy to use (usually "greedy").
        structure_predictor : str | StructurePredictor
            The name of the model that will use the MSA, to make sure the format is correct.

        Returns
        -------
        dict[str | int, Path]
            A dictionary mapping target names to MSA file paths.
        """
        hash_key = self._hash_arguments(data, msa_pairing_strategy)

        if structure_predictor in [
            StructurePredictor.BOLTZ_1,
            StructurePredictor.BOLTZ_2,
            StructurePredictor.RF3,
        ]:
            # get standard MSAs
            suffix = "a3m" if structure_predictor == StructurePredictor.RF3 else "csv"
            msa_path_dict = {
                key: self.msa_dir / f"{hash_key}_{idx}.{suffix}" for idx, key in enumerate(data)
            }

            if not all([m.exists() for m in msa_path_dict.values()]):
                # this will generate both a3m and csv files for us.
                # do NOT capture this output, it will return paths to .csv files,
                # which we don't necessarily want.
                _ = _compute_msa(
                    data,
                    hash_key,  # this is the "target_id" argument to compute_msa
                    self.msa_dir,
                    self.msa_server_url,
                    msa_pairing_strategy,
                    msa_server_username=self.msa_server_username,
                    msa_server_password=self.msa_server_password,
                    api_key_header=self.api_key_header,
                    api_key_value=self.api_key_value,
                )
                self._api_calls += 1
            else:
                self._cache_hits += 1

            _validate_msa_cache_contents(hash_key, self.msa_dir)

        # Protenix needs special MSAs
        # This is a kind of hacky way to handle Protenix; I'm not sure what to do easily but
        # use their pipeline and just have it put everything in our cache directory.
        elif structure_predictor == StructurePredictor.PROTENIX:
            if not PROTENIX_AVAILABLE:
                raise RuntimeError("Protenix is not installed, cannot use Protenix MSA tools.")

            logger.info("Running Protenix MSA tools.")
            # Make sure we have a protenix subdirectory
            protenix_dir = self.msa_dir / "protenix"
            protenix_dir.mkdir(parents=True, exist_ok=True)
            # Protenix adds extra information, easiest just to use their pipeline.
            # make sure sort order stays the same:
            data_items = sorted(data.items(), key=lambda item: _msa_data_key_sort_key(item[0]))
            data_keys = [key for key, _ in data_items]
            sequences = [sequence for _, sequence in data_items]
            out_dir = self.msa_dir / "protenix" / hash_key

            msa_directories = [out_dir / str(idx) for idx in data_keys]
            reqd_files = ["non_pairing.a3m", "pairing.a3m"]
            need_msas = not all(
                (out_dir / str(idx) / fn).exists() for idx in data_keys for fn in reqd_files
            )
            if need_msas:
                msa_directories = protenix_msa_search(sequences, str(out_dir), mode="protenix")
                self._api_calls += 1
            else:
                self._cache_hits += 1

            # protenix_msa_search returns string-style paths; normalize to Path so the
            # return type matches the declared dict[str | int, Path] (idempotent on Paths).
            msa_path_dict = {key: Path(path) for key, path in zip(data_keys, msa_directories)}

        else:
            raise ValueError(f"Unknown structure predictor: {structure_predictor}")
        return msa_path_dict

    def report_on_usage(self):
        """
        Report on the usage of the MSA manager, including API calls and cache hits.
        """
        logger.info(
            f"MSA Manager Usage Report: API Calls={self._api_calls}, Cache Hits={self._cache_hits}"
        )
