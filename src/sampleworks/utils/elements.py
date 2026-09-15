from collections.abc import Iterable

import gemmi
from loguru import logger

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ATOM_STRUCTURE_FACTORS,
    ATOMIC_NUM_TO_ELEMENT,
    ELEMENT_TO_SCATTERING_INDEX,
)


VALID_ELEMENTS = set(ATOMIC_NUM_TO_ELEMENT) | set(ATOM_STRUCTURE_FACTORS.keys())


def normalize_element(elem: str) -> str:
    """Normalize element symbol to title case (e.g., 'CA' -> 'Ca', 'c' -> 'C')."""
    normalized = elem.strip().title()
    if normalized not in VALID_ELEMENTS:
        logger.warning(f"Unrecognized element symbol: '{elem}' (normalized: '{normalized}')")
    return normalized


def element_to_scattering_idx(raw: str) -> int:
    """Normalize an element symbol and return its scattering tensor index.

    Unknown elements are logged and mapped to index 0 (the ``'?'``
    placeholder row, which carries zero scattering factors).

    Parameters
    ----------
    raw
        Raw element symbol as found on an ``AtomArray`` (e.g. ``'CA'``,
        ``'Fe2+'``, ``'SE'``).

    Returns
    -------
    int
        Index into the scattering parameter tensor built by
        ``setup_scattering_params``.
    """
    normalized = normalize_element(raw)
    idx = ELEMENT_TO_SCATTERING_INDEX.get(normalized)
    if idx is None:
        logger.warning(
            f"Element '{raw}' (normalized: '{normalized}') is not in the scattering "
            "index table and will contribute zero density."
        )
        return 0
    return idx


def elements_to_scattering_indices(elements: Iterable[str]) -> list[int]:
    """Map a sequence of raw element symbols to scattering tensor indices.

    Convenience wrapper around :func:`element_to_scattering_idx` for the
    common case of converting an entire ``AtomArray.element`` array.

    Parameters
    ----------
    elements
        Iterable of raw element symbols.

    Returns
    -------
    list[int]
        Scattering tensor indices, one per input element.
    """
    return [element_to_scattering_idx(e) for e in elements]


def it92_coefficients(symbols: Iterable[str] | None = None) -> dict[str, tuple[float, ...]]:
    """Fetch IT92 X-ray scattering coefficients from gemmi.

    Returns the coefficients of ``f(s) = sum_i a_i exp(-b_i s^2) + c`` as
    ``(a1..a4, b1..b4, c)``, the layout ``lunus.sf``'s kernel builders take.
    Formal charge is ignored, the neutral-atom convention of the IT92 table.

    Parameters
    ----------
    symbols
        Element symbols to fetch. Defaults to every symbol of
        :data:`ELEMENT_TO_SCATTERING_INDEX` that gemmi recognizes, which
        excludes the ``'?'`` placeholder and qfit's valence-state
        pseudo-elements.

    Returns
    -------
    dict[str, tuple[float, ...]]
        Nine coefficients per symbol, keyed as passed in.

    Raises
    ------
    KeyError
        If an explicitly requested symbol is not a gemmi element.
    """
    default_set = symbols is None
    coefficients: dict[str, tuple[float, ...]] = {}
    for symbol in ELEMENT_TO_SCATTERING_INDEX if symbols is None else symbols:
        element = gemmi.Element(symbol.rstrip("+-0123456789"))
        # gemmi yields the unknown element, atomic number 0, rather than raising.
        if element.atomic_number == 0:
            if default_set:
                continue
            raise KeyError(f"gemmi does not recognize element symbol {symbol!r}")
        it92 = element.it92
        coefficients[symbol] = (*it92.a, *it92.b, it92.c)
    return coefficients
