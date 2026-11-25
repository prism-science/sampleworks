"""Tests for the CCD distance-bound violation penalty."""

import biotite.structure as struc
import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, AtomArrayStack
from sampleworks.core.losses.ccd_bounds import CCDBoundsLossFunction
from sampleworks.core.losses.protocol import LossInputs, PreparableLossFunctionProtocol
from sampleworks.core.losses.registry import build_single_loss, LossBuildContext
from sampleworks.utils.guidance_constants import Losses


@pytest.fixture(scope="module")
def protein_1vme(structure_1vme) -> AtomArray:
    atom_array = structure_1vme["asym_unit"]
    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    resolved = ~np.isnan(atom_array.coord).any(axis=-1)
    return atom_array[struc.filter_amino_acids(atom_array) & resolved]


def prepared_loss(atom_array: AtomArray, **kwargs) -> CCDBoundsLossFunction:
    loss = CCDBoundsLossFunction(**kwargs)
    loss.prepare(LossInputs.from_atom_array(atom_array, ensemble_size=1))
    return loss


def penalty(loss: CCDBoundsLossFunction, coordinates: torch.Tensor) -> torch.Tensor:
    """Call ``loss`` with placeholder per-atom inputs, which it ignores."""
    unused = torch.zeros(coordinates.shape[:-1])
    return loss(coordinates, unused.long(), unused, unused)


def score(loss: CCDBoundsLossFunction, coords) -> float:
    coordinates = torch.as_tensor(np.asarray(coords), dtype=torch.float32).reshape(1, -1, 3)
    return penalty(loss, coordinates).item()


def test_is_preparable_and_registered():
    loss = build_single_loss(Losses.CCD_BOUNDS, {}, LossBuildContext(structure={}))

    assert isinstance(loss, CCDBoundsLossFunction)
    assert isinstance(loss, PreparableLossFunctionProtocol)


def test_a_deposited_structure_barely_violates_its_bounds_and_noise_raises_it(protein_1vme):
    """Refined geometry sits inside the bounds; only isolated pairs graze them."""
    loss = prepared_loss(protein_1vme)
    rng = np.random.default_rng(0)

    clean = score(loss, protein_1vme.coord)
    noisy = score(loss, protein_1vme.coord + rng.normal(scale=0.3, size=protein_1vme.coord.shape))

    assert clean / protein_1vme.array_length() < 1e-3
    assert noisy > 100 * max(clean, 1e-3)


def test_a_descent_step_reduces_the_violation(protein_1vme):
    loss = prepared_loss(protein_1vme)
    rng = np.random.default_rng(1)
    noisy = torch.tensor(
        protein_1vme.coord + rng.normal(scale=0.3, size=protein_1vme.coord.shape),
        dtype=torch.float32,
    ).unsqueeze(0)
    noisy.requires_grad_(True)

    before = penalty(loss, noisy)
    before.backward()
    gradient = noisy.grad
    assert gradient is not None
    stepped = (noisy - 1e-3 * gradient).detach()

    assert torch.isfinite(gradient).all()
    assert score(loss, stepped) < before.item()


def test_a_broken_peptide_bond_is_penalized_by_its_excess_length(protein_1vme):
    """Pulling a residue away along its incoming C-N bond is charged past the bond's window."""
    starts = struc.get_residue_starts(protein_1vme, add_exclusive_stop=True)
    first, second = protein_1vme[starts[0] : starts[1]], protein_1vme[starts[1] : starts[2]]
    dipeptide = first + second
    loss = prepared_loss(dipeptide, bond_buffer=0.0, angle_buffer=0.0)

    carbon = dipeptide.coord[dipeptide.atom_name == "C"][0]
    nitrogen = dipeptide.coord[dipeptide.atom_name == "N"][1]
    direction = (nitrogen - carbon) / np.linalg.norm(nitrogen - carbon)
    pulled = dipeptide.coord.copy()
    pulled[first.array_length() :] += 2.0 * direction

    assert score(loss, pulled) - score(loss, dipeptide.coord) > 1.5
