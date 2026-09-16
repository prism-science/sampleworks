# utility script to put all header information from original PDB entry into our CIF files
import fnmatch
import json
import re
from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.io.utils.io_utils import load_any
from biotite.database.rcsb import fetch
from biotite.structure.io.pdbx import CIFColumn, CIFFile, set_structure
from loguru import logger
from sampleworks.utils.atom_array_utils import remove_atoms_with_any_nan_coords
from sampleworks.utils.cif_utils import add_category_to_cif, resolve_mixed_hetatm_atom_altlocs


SAMPLEWORKS_CACHE = Path("~/.sampleworks/rcsb").expanduser()


# A valid PDB ID is either the extended 12-char form `pdb_` + 8 alphanumerics
# (e.g. `pdb_00004hhb`) or the legacy 4-char form, whose first character is always a digit (0-9).
# The leading-digit rule is what distinguishes a real legacy ID from a stray folder name like
# `TEST`/`logs`.
_VALID_RCSB_ID = re.compile(r"pdb_[A-Za-z0-9]{8}|[0-9][A-Za-z0-9]{3}")

# Default --rcsb-pattern: locate the id (one capturing group) right after the
# grid_search_results/ folder. Its group IS _VALID_RCSB_ID, so the default pattern and the
# validator can never drift apart.
DEFAULT_RCSB_PATTERN = rf"grid_search_results/({_VALID_RCSB_ID.pattern})"


def crawl_dir_by_depth(
    root_dir: str | Path,
    target_pattern: str,
    n_levels: int,
) -> list[Path]:
    """
    Recursively crawl `root_dir` up to `n_levels` directory levels deep and return
    all files whose *name* matches `target_pattern` (fnmatch-style, e.g. "*.cif").

    Depth meaning:
      - n_levels = 0: only files directly in root_dir
      - n_levels = 1: root_dir + its immediate subdirectories
      - etc.
    """
    root = Path(root_dir)
    if n_levels < 0:
        return []

    results: list[Path] = []

    def _crawl(current: Path, levels_left: int) -> None:
        try:
            for entry in current.iterdir():
                if entry.is_file():
                    if fnmatch.fnmatch(entry.name, target_pattern):
                        logger.info(f"Found matching file: {entry}")
                        results.append(entry)
                elif entry.is_dir() and levels_left > 0:
                    _crawl(entry, levels_left - 1)
        except (PermissionError, FileNotFoundError):
            # Skip unreadable or transient directories
            return

    _crawl(root, n_levels)
    return results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--cif-pattern",
        default="refined.cif",
        help="Pattern used by fnmatch/glob for cif files to patch, default: 'refined.cif'",
    )
    parser.add_argument(
        "--rcsb-pattern",
        default=DEFAULT_RCSB_PATTERN,
        help="Regex pattern for rcsb ids in file paths. "
        "Must have only one group, surrounding the id",
    )
    parser.add_argument(
        "--depth", type=int, default=4, help="Depth to search the directory tree below input-dir"
    )
    parser.add_argument("--grid-search-input-dir", required=True)
    parser.add_argument(
        "--input-pdb-pattern",
        default="{pdb_id}/{pdb_id}_single_001_density_input.cif",
        help="Pattern used by fnmatch/glob for input pdb files. The complete path of the input "
        "pdb must match f'{grid-search-input-dir}/{input-pdb-pattern}'. Defaults to "
        "'{pdb_id}/{pdb_id}_single_001_density_input.cif'",
    )
    args = parser.parse_args()
    return args


def main(
    input_dir: str | Path,
    grid_search_input_dir: str | Path,
    target_pattern: str,
    rcsb_regex: str = DEFAULT_RCSB_PATTERN,
    depth: int = 4,
    input_pdb_pattern: str = "{pdb_id}/{pdb_id}_single_001_density_input.cif",
) -> int:
    """Patch Sampleworks output CIF files for downstream evaluation tools.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing generated Sampleworks CIF files.
    grid_search_input_dir : str or Path
        Root directory containing original input CIF files.
    target_pattern : str
        Filename pattern for generated CIFs to patch.
    rcsb_regex : str, optional
        Regular expression with one capture group for the RCSB ID.
    depth : int, optional
        Directory recursion depth below ``input_dir``.
    input_pdb_pattern : str, optional
        Format string for locating original input CIFs from ``pdb_id``.

    Returns
    -------
    int
        ``0`` when all matched files patch successfully, otherwise ``1``.
    """
    # make sure the cache exists
    SAMPLEWORKS_CACHE.mkdir(parents=True, exist_ok=True)

    cif_files_to_patch = crawl_dir_by_depth(input_dir, target_pattern, n_levels=depth)
    if not cif_files_to_patch:
        logger.error(f"No CIF files matching {target_pattern!r} found under {input_dir}")
        return 1
    results = joblib.Parallel()(
        joblib.delayed(patch_individual_cif_file)(
            f, rcsb_regex, Path(grid_search_input_dir), input_pdb_pattern
        )
        for f in cif_files_to_patch
    )
    results = [r for r in results if r]
    if results:
        logger.error("The following errors occurred:")
        for r in results:
            print(r)
        return 1
    return 0


class InvalidRcsbIdError(ValueError):
    """Raised when ``--rcsb-pattern`` matches a path but its capturing group captured a
    string that is not a valid PDB id -- usually a sign the pattern targets the wrong part
    of the path. Distinct from a plain no-match so the caller can warn specifically."""

    def __init__(self, token: str, rcsb_regex: str) -> None:
        self.token = token
        self.rcsb_regex = rcsb_regex
        super().__init__(f"--rcsb-pattern captured {token!r}, which is not a valid PDB id")


def extract_rcsb_id(cif_path: Path, rcsb_regex: str) -> str | None:
    """Extract and validate the RCSB id from a cif path.

    ``rcsb_regex`` locates the candidate (it must contain exactly one capturing group around
    the id; see ``--rcsb-pattern``). The id must **start a folder component**, and must
    either fill that component or be followed by a delimiter the pattern itself matches --
    so a stray prefix can't silently resolve to the wrong entry. The token is then checked
    against the PDB-id grammar and returned *verbatim* (legacy ``4hhb`` or extended
    ``pdb_00004hhb`` -- no normalization).

    Examples with the default pattern (folder -> result)::

        4hhb                  -> "4hhb"          (works)
        pdb_00004hhb          -> "pdb_00004hhb"  (works)
        1VME                  -> "1VME"          (works)
        4hhb_final            -> None  (id is only a prefix of the folder)
        protease_pdb_1000abcd -> None  (id is not at the component start)
        1abc_pdb_1000abcd     -> None  (two id-like parts -- ambiguous)
        logs                  -> None  (not a PDB id)

    Possible problem this guards against: without the rule, ``4hhb_final`` would yield
    ``4hhb`` and ``1abc_pdb_1000abcd`` would yield ``1abc`` -- both silently patching the
    wrong entry. Such folders are skipped with a warning instead.

    When the id is genuinely embedded in a longer folder name, say so in the pattern by
    matching the delimiter after the capturing group. Occupancy-sweep runs produce folders
    like ``1VME_0.25occA_0.75occB`` whose reference entry really is ``1VME``::

        --rcsb-pattern '<results-dir>/([0-9][A-Za-z0-9]{3}|pdb_[A-Za-z0-9]{8})_[0-9.]+occ'

    Matching past the group is what distinguishes "I know what follows the id" from the
    default pattern's "the id is the whole folder", so relaxing it here does not weaken the
    default. Beware names containing more than one id-like substring.

    Returns ``None`` when no complete PDB-id folder is found. Raises ``InvalidRcsbIdError``
    when a whole component is captured but is not a valid PDB id (a likely sign the pattern
    targets the wrong part of the path), and ``ValueError`` when the pattern does not have
    exactly one capturing group. This is the main checkpoint: a wrong id here would patch
    coordinates into the wrong template.
    """
    rcsb_re = re.compile(rcsb_regex)
    if rcsb_re.groups != 1:
        raise ValueError(
            f"--rcsb-pattern must have exactly one capturing group, "
            f"got {rcsb_re.groups}: {rcsb_regex!r}"
        )
    path_str = cif_path.as_posix()
    m = rcsb_re.search(path_str)
    if not m:
        return None
    # The id must start a folder component: reject a capture that begins mid-name
    # (e.g. "4hhb" out of "foo4hhb"), which would silently resolve to the wrong entry.
    start_ok = m.start(1) == 0 or path_str[m.start(1) - 1] == "/"
    # For the right-hand edge the id must either fill the component, or the pattern itself
    # must say what follows it. A pattern that matches past its capturing group has stated
    # the delimiter explicitly (e.g. `([0-9][A-Za-z0-9]{3})_[0-9.]+occ` for occupancy-sweep
    # folders like `1VME_0.25occA_0.75occB`), which is the documented way to handle ids
    # embedded in longer names. With the default pattern nothing follows the group, so the
    # strict rule still applies and `4hhb_final` is still skipped rather than silently
    # patched as `4hhb`.
    pattern_states_delimiter = m.end(0) > m.end(1)
    end_ok = pattern_states_delimiter or m.end(1) == len(path_str) or path_str[m.end(1)] == "/"
    if not (start_ok and end_ok):
        return None
    token = m.group(1)
    if not _VALID_RCSB_ID.fullmatch(token):
        raise InvalidRcsbIdError(token, rcsb_regex)
    return token


def patch_individual_cif_file(
    cif_file: Path, rcsb_regex: str, reference_dir: Path, input_pdb_pattern: str
) -> str | None:  # returns an error message if there was one
    cif_path = Path(cif_file)
    try:
        rcsb_id = extract_rcsb_id(cif_path, rcsb_regex)
    except InvalidRcsbIdError as exc:
        msg = (
            f"--rcsb-pattern {rcsb_regex!r} matched {cif_file} but captured {exc.token!r}, "
            f"which is not a valid PDB id (expected e.g. '4hhb' or 'pdb_00004hhb'; "
            f"regex {_VALID_RCSB_ID.pattern!r}). Check that the capturing group targets the id."
        )
        logger.warning(msg)
        return msg
    if rcsb_id is None:
        msg = (
            f"--rcsb-pattern {rcsb_regex!r} did not find a complete PDB-id folder in "
            f"{cif_file}. Expected a folder named exactly a PDB id (e.g. '4hhb' or "
            f"'pdb_00004hhb'; regex {_VALID_RCSB_ID.pattern!r}). "
            f"Check the directory layout or the pattern."
        )
        logger.warning(msg)
        return msg
    # Get the offset for residue numbering in the reference structure
    try:
        reference_path = reference_dir / input_pdb_pattern.format(pdb_id=rcsb_id)
        # fetch only downloads the file if it isn't already present.
        rcsb_path = fetch(rcsb_id, format="cif", target_path=str(SAMPLEWORKS_CACHE))

        # Mirror the fix from guidance_script_utils - strip mixed ATOM/HETATM altlocs
        # at the same residue position (e.g. 6NI5/6 CYS/CSO) so the reference matches
        # what is generated by guidance. Returns the original path if nothing to fix.
        safe_reference_path = resolve_mixed_hetatm_atom_altlocs(reference_path)
        reference = load_any(safe_reference_path)
        if safe_reference_path != reference_path:
            safe_reference_path.unlink()

        asym_unit = load_any(cif_file)
        asym_unit = ensure_atom_array_stack(asym_unit)
    except Exception:
        msg = f"Unable to read and parse either/both of {reference_path}, {cif_file}. "
        logger.warning(msg)
        return msg

    # get the unique residue numbers for each file
    if reference.res_id is None or asym_unit.res_id is None:
        msg = f"Residue numbers for {cif_path} and/or {reference_path} are missing."
        logger.error(msg)
        return msg

    # CodeRabbit improved this: we are now not chain agnostic, so we can handle multiple chains
    ref_keys = list(dict.fromkeys(zip(reference.chain_id.tolist(), reference.res_id.tolist())))
    cif_keys = list(dict.fromkeys(zip(asym_unit.chain_id.tolist(), asym_unit.res_id.tolist())))

    # There should be a single, unique mapping between them. If not, something is wrong.
    if len(ref_keys) != len(cif_keys):
        msg = f"Residue numbers in {cif_path} cannot be mapped to those in {reference_path}"
        logger.error(msg)
        return msg

    # TODO: uncomment below region after fixing chain mismatches upstream
    # (protenix json creation needs update)
    # which breaks the commented multi-chain handling below

    # patch the residue numbers to match the original pdb
    # mapping = {}
    # for cif_key, ref_key in zip(cif_keys, ref_keys, strict=True):
    #     if cif_key[0] != ref_key[0]:
    #         msg = f"Chain mismatch while remapping residues for {cif_path} vs {reference_path}"
    #         logger.error(msg)
    #         # return msg
    #         # TODO: fix chain mismatches upstream (protenix json creation needs update)
    #         # this breaks multi-chain stuff for now

    #     mapping[cif_key] = ref_key[1]

    # TODO: delete the line below after uncommenting the mapping above
    # Some models currently relabel chains and/or renumber residues, so we
    # remap both per atom. ``mapping`` is keyed by the predicted (chain, res) and
    # returns the reference (chain, res) at the same position
    mapping = dict(zip(cif_keys, ref_keys, strict=True))

    atom_keys = list(zip(asym_unit.chain_id.tolist(), asym_unit.res_id.tolist()))
    # TODO: uncomment after fixing chain mismatches upstream
    # asym_unit.res_id = np.array([mapping[k] for k in atom_keys], dtype=asym_unit.res_id.dtype)
    # TODO: delete these two lines after fixing chain mismatches upstream
    asym_unit.chain_id = np.array([mapping[k][0] for k in atom_keys])
    asym_unit.res_id = np.array([mapping[k][1] for k in atom_keys], dtype=asym_unit.res_id.dtype)

    # load the actual PDB, we'll copy the new coordinates and metadata into it.
    template = CIFFile.read(rcsb_path)

    # Write sampleworks trial metadata to the CIF file, if we can find it
    cif_data = CIFFile.read(cif_path)
    if "sampleworks" in cif_data.block:
        template.block["sampleworks"] = cif_data.block["sampleworks"]
    elif (metadata_path := cif_path.parent / "job_metadata.json").exists():
        with open(metadata_path, "r") as fp:
            metadata = json.load(fp)
        if metadata is not None:
            add_category_to_cif(template, metadata, "sampleworks")
        else:
            logger.warning(f"Sampleworks metadata file at {metadata_path} is empty")
    else:
        logger.warning(f"No sampleworks metadata found for {cif_path}")

    # remove any atoms with nan coordinates--these seem to come in because we sometimes use parse
    # (from AtomWorks) which creates them. Still, we'll do this here just in case.
    asym_unit = remove_atoms_with_any_nan_coords(asym_unit)

    # make sure entity ids match in atom_site and entity_poly
    if "entity_poly" in template.block:
        ep = template.block["entity_poly"]
        # fixme for now I'm using a hack--if there's one polymer entity, just assert that
        #   polymers in atom_site have to be that one. Otherwise do nothing.
        if len(ep["entity_id"]) == 1:
            entity_id = ep["entity_id"].as_item()
            if "label_entity_id" not in asym_unit.get_annotation_categories():
                asym_unit.add_annotation("label_entity_id", int)
            asym_unit.label_entity_id = np.ones_like(asym_unit.label_entity_id) * int(entity_id)
    else:
        logger.warning("No entity_poly block found in template CIF file. Cannot patch entity ids")

    # now set the structure with correct entity ids
    set_structure(template, asym_unit)

    # If there's a pdbx_poly_seq_scheme, make sure the seq nums all agree, as
    # the numbers in our outputs will all agree. We appear to use the one called ndb_seq_num
    nsm = template.block["pdbx_poly_seq_scheme"]["ndb_seq_num"]
    template.block["pdbx_poly_seq_scheme"]["pdb_seq_num"] = nsm
    template.block["pdbx_poly_seq_scheme"]["auth_seq_num"] = nsm

    # Make sure the id field is unique to each atom
    template.block["atom_site"]["id"] = CIFColumn(np.arange(np.prod(asym_unit.shape)))

    # make sure there are "occupancy" and "B_iso_or_equiv" annotations
    if "occupancy" not in template.block["atom_site"].keys():
        template.block["atom_site"]["occupancy"] = CIFColumn(
            [1.0] * len(template.block["atom_site"]["id"])
        )
    if "B_iso_or_equiv" not in template.block["atom_site"].keys():
        template.block["atom_site"]["B_iso_or_equiv"] = CIFColumn(
            [20.0] * len(template.block["atom_site"]["id"])
        )

    template.block.name = cif_path.stem
    patched_cif_name = cif_path.parent / (cif_path.stem + "-patched.cif")
    template.write(patched_cif_name)
    logger.info(f"Wrote {patched_cif_name}")
    return None


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        main(
            args.input_dir,
            args.grid_search_input_dir,
            args.cif_pattern,
            args.rcsb_pattern,
            args.depth,
            args.input_pdb_pattern,
        )
    )
