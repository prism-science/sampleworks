"""Tests for the per-element lookups in ``sampleworks.utils.elements``."""

import logging

import pytest
from sampleworks.utils.elements import covalent_radius


@pytest.mark.parametrize(
    ("symbol", "expected_radius"),
    [("C", 0.73), ("N", 0.71), ("O", 0.66), ("SE", 1.20), ("CA", 1.76)],
)
def test_covalent_radius_of_known_element(symbol, expected_radius, caplog):
    """A known symbol, title-case or PDB all-caps, returns its covalent radius without warning.

    ``'CA'`` must resolve to calcium (1.76 Å), not be mistaken for carbon.

    Parameters
    ----------
    symbol : str
        Element symbol as it may appear on an ``AtomArray``.
    expected_radius : float
        Tabulated covalent radius of that element, in Å.
    """
    with caplog.at_level(logging.WARNING):
        radius = covalent_radius(symbol)
    assert radius == pytest.approx(expected_radius)
    assert caplog.text == ""


@pytest.mark.parametrize("symbol", ["?", "Xx"])
def test_covalent_radius_warns_on_unrecognized_symbol(symbol, caplog):
    """An unrecognized symbol falls back to gemmi's unknown-element radius and says so.

    Parameters
    ----------
    symbol : str
        A symbol gemmi cannot resolve: the ``'?'`` placeholder some density inputs carry,
        and a symbol that is not an element at all.
    """
    with caplog.at_level(logging.WARNING):
        radius = covalent_radius(symbol)
    assert radius == pytest.approx(0.50)
    assert f"'{symbol}'" in caplog.text
