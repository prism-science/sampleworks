"""Handle modified and nonstandard residues for ESMFold2.

The current approach simply replaces select modified and nonstandard residues with their
canonical equivalent. ESMFold2's API also supports modification record objects, but we
have not yet explored using them here.
"""

from atomworks.constants import STANDARD_AA, STANDARD_AA_ONE_LETTER
from biotite.structure import AtomArray, get_residue_starts


CANONICAL_RESIDUE_REPLACEMENTS: dict[str, str] = {
    "MSE": "MET",
}


def canonicalize_residue_sequence(atoms: AtomArray) -> str:
    """Build a canonical sequence using supported residue replacements.

    Parameters
    ----------
    atoms
        Nonempty protein atom array for one chain, in residue order. The caller
        validates the chain type. The reference array is not modified.

    Returns
    -------
    str
        Canonical one-letter sequence in the original residue order.

    Raises
    ------
    NotImplementedError
        A nonstandard residue has no supported replacement.
    """
    standard_residue_codes = dict(zip(STANDARD_AA, STANDARD_AA_ONE_LETTER, strict=True))
    unsupported_residues = sorted(
        set(atoms.res_name) - standard_residue_codes.keys() - CANONICAL_RESIDUE_REPLACEMENTS.keys()
    )

    if unsupported_residues:
        raise NotImplementedError(
            f"{unsupported_residues} modified/nonstandard residues are unsupported."
        )

    residue_names = atoms.res_name[get_residue_starts(atoms)]
    return "".join(
        standard_residue_codes[CANONICAL_RESIDUE_REPLACEMENTS.get(name, name)]
        for name in residue_names
    )
