import re

from sampleworks.utils.cif_utils import RCSB_ID_PATTERN


# `{value}occ{label}` altloc occupancy token, e.g. `0.25occA`, as written by occupancy_to_str.
OCCUPANCY_TOKEN_PATTERN = re.compile(r"(\d+\.?\d*)(?i:occ)([A-Za-z])")
# Grid-search protein name: an RCSB id followed only by occupancy tokens, e.g. `1VME_0.5occA`.
PROTEIN_NAME_PATTERN = re.compile(
    rf"(?P<rcsb_id>{RCSB_ID_PATTERN.pattern})(?:_{OCCUPANCY_TOKEN_PATTERN.pattern})*"
)


def extract_protein_and_occupancy(dir_name: str) -> tuple[str, dict[str, float]]:
    """Extract protein name and altloc occupancies from a directory name.

    Parses all ``{value}occ{label}`` tokens found in *dir_name*.  The protein
    name is taken as the first underscore-delimited token that does not match
    the occupancy pattern.

    Parameters
    ----------
    dir_name : str
        Directory name to parse, e.g. ``"1vme_0.5occA_0.5occB"``.

    Returns
    -------
    tuple[str, dict[str, float]]
        ``(protein, altloc_occupancies)`` where *altloc_occupancies* maps
        uppercase altloc labels to their occupancy values.  The dict is empty
        when no occupancy tokens are found.

    Examples
    -------
    >>> extract_protein_and_occupancy('1vme_0.5occA_0.5occB')
    ('1vme', {'A': 0.5, 'B': 0.5})
    >>> extract_protein_and_occupancy('6b8x_1.0occA')
    ('6b8x', {'A': 1.0})
    >>> extract_protein_and_occupancy('5sop_1.0occB')
    ('5sop', {'B': 1.0})
    >>> extract_protein_and_occupancy('1abc_0.5occA_0.3occB_0.2occC')
    ('1abc', {'A': 0.5, 'B': 0.3, 'C': 0.2})
    """
    protein = dir_name.split("_")[0].lower()

    altloc_occupancies: dict[str, float] = {}
    for match in OCCUPANCY_TOKEN_PATTERN.finditer(dir_name):
        label = match.group(2).upper()
        altloc_occupancies[label] = float(match.group(1))

    return protein, altloc_occupancies


def rcsb_id_from_protein_name(protein_name: str) -> str | None:
    """Extract the RCSB id from a grid-search protein name.

    The whole name must be an RCSB id followed only by occupancy tokens, so names such as
    ``4hhb_final`` or ``1abc_pdb_1000abcd`` return None rather than a wrong entry. Unlike
    ``extract_protein_and_occupancy``, the letter case of the id is kept.

    Parameters
    ----------
    protein_name : str
        Protein name from a ``--proteins`` CSV or a trial directory, e.g.
        ``"1VME_0.25occA_0.75occB"``.

    Returns
    -------
    str | None
        The legacy (``"1VME"``) or extended (``"pdb_00001vme"``) RCSB id, or None when the
        name is not of that form.

    Examples
    --------
    >>> rcsb_id_from_protein_name('3T94_1.0occA')
    '3T94'
    >>> rcsb_id_from_protein_name('pdb_00003t94_0.5occA_0.5occB')
    'pdb_00003t94'
    >>> rcsb_id_from_protein_name('lysozyme_1.0occA') is None
    True
    """
    match = PROTEIN_NAME_PATTERN.fullmatch(protein_name)
    return match["rcsb_id"] if match else None


def occupancy_to_str(**altloc_occupancies: float) -> str:
    """Convert altloc occupancies to the string format used in filenames.

    Zero-occupancy altlocs are omitted. Values are rounded to two decimal
    places to avoid floating-point artifacts in filenames.

    Parameters
    ----------
    **altloc_occupancies : float
        Keyword arguments mapping altloc labels to their occupancies.

    Returns
    -------
    str
        Underscore-joined occupancy string, e.g. ``"0.5occA_0.5occB"``.

    Raises
    ------
    ValueError
        If no altlocs have non-zero occupancy, if occupancies sum to more than 1 when rounded to 2
         decimal places, or if any occupancy is outside the range [0, 1].

    Examples
    -------
    >>> occupancy_to_str(A=1.0, B=0.0)
    '1.0occA'
    >>> occupancy_to_str(A=0.0, B=1.0)
    '1.0occB'
    >>> occupancy_to_str(A=0.5, B=0.5)
    '0.5occA_0.5occB'
    >>> occupancy_to_str(A=0.25, B=0.75)
    '0.25occA_0.75occB'
    >>> occupancy_to_str(A=0.5, B=0.3, C=0.2)
    '0.5occA_0.3occB_0.2occC'
    """
    # Canonicalize occupancies by rounding before validation and output
    canonical_occ = {label: round(float(val), 2) for label, val in altloc_occupancies.items()}

    if sum(canonical_occ.values()) > 1:
        raise ValueError(
            "Altloc occupancies cannot sum to more than 1, currently "
            f"they sum to {sum(canonical_occ.values())}: {canonical_occ}"
        )
    if any(occ < 0 or occ > 1 for occ in canonical_occ.values()):
        raise ValueError(
            f"Altloc occupancies must be between 0 and 1, currently they don't: {canonical_occ}"
        )
    parts = []
    for label in sorted(canonical_occ, key=lambda label_name: str(label_name).upper()):
        occ = canonical_occ[label]
        if abs(occ) > 1e-6:
            label_str = str(label).upper()
            parts.append(f"{occ}occ{label_str}")
    if not parts:
        raise ValueError("At least one altloc must have non-zero occupancy")
    return "_".join(parts)
