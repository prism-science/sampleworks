"""Handle modified and nonstandard residues for ESMFold2.

The current approach simply replaces select modified and nonstandard residues with their
canonical equivalent. ESMFold2's API also supports modification record objects, but we
have not yet explored using them here.
"""

from atomworks.constants import STANDARD_AA, STANDARD_AA_ONE_LETTER
from biotite.structure import AtomArray, get_residue_starts
from esm.models.esmfold2.types import ProteinInput


CANONICAL_RESIDUE_REPLACEMENTS: dict[str, str] = {
    "MSE": "MET",
}


def handle_modified_and_nonstandard_residues(atoms: AtomArray) -> ProteinInput:
    """Build a protein input using supported canonical residue replacements.

    Parameters
    ----------
    atoms
        Nonempty protein atom array for one chain, in residue order. The caller
        validates the chain type. The reference array is not modified.

    Returns
    -------
    ProteinInput
        Chain identity and canonicalized sequence.

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
    sequence = "".join(
        standard_residue_codes[CANONICAL_RESIDUE_REPLACEMENTS.get(name, name)]
        for name in residue_names
    )
    return ProteinInput(id=str(atoms.chain_id[0]), sequence=sequence)
