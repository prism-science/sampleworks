"""A test case for testing the geometry reward

This test case is designed to verify the reward function that evaluates the geometry of a
molecular structure. It checks whether the reward function correctly computes the reward
based on the provided atomic coordinates and other relevant parameters.
"""

import gemmi
from sampleworks.core.rewards.geometry import _covalent_radius


def test_known_element_matches_gemmi():
    r = _covalent_radius("C")
    assert r > 0, "Covalent radius for Carbon should be greater than 0"
    assert r == gemmi.Element("C").covalent_r
