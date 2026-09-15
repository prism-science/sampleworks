"""Tests for elements module."""

import pytest
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ELEMENT_TO_SCATTERING_INDEX,
)
from sampleworks.utils.elements import it92_coefficients


class TestIT92Coefficients:
    """IT92 scattering coefficients read from gemmi."""

    def test_carbon_matches_published_table(self):
        """Coefficients must be the published IT92 values (Int. Tables Vol. C,
        Table 6.1.1.4), ordered ``(a1..a4, b1..b4, c)``."""
        assert it92_coefficients(["C"])["C"] == pytest.approx(
            (2.31, 1.02, 1.5886, 0.865, 20.8439, 10.2075, 0.5687, 51.6512, 0.2156)
        )

    def test_formal_charge_is_ignored(self):
        """IT92 tabulates neutral atoms, so ionic symbols must resolve to their
        neutral parent rather than failing or scattering off nothing."""
        coefficients = it92_coefficients(["Fe", "Fe2+", "O", "O1-"])
        assert coefficients["Fe2+"] == coefficients["Fe"]
        assert coefficients["O1-"] == coefficients["O"]

    def test_unknown_symbol_raises(self):
        """An unrecognized symbol must fail loudly; gemmi's own default is to
        return the unknown element with meaningless coefficients."""
        with pytest.raises(KeyError, match="Xx"):
            it92_coefficients(["Xx"])

    def test_default_set_covers_the_scattering_index_table(self):
        """The default set must cover every real element of the density path's
        index table, omitting only the placeholder and qfit's valence-state
        pseudo-elements, which IT92 does not tabulate."""
        omitted = set(ELEMENT_TO_SCATTERING_INDEX) - set(it92_coefficients())
        assert omitted == {"?", "Cval"}
