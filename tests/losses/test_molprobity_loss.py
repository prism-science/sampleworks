"""Tests for the soft heavy-atom MolProbity score."""

import biotite.structure as struc
import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, AtomArrayStack
from sampleworks.core.losses.molprobity import (
    _contour_grids,
    MolProbityLossFunction,
    RAMACHANDRAN_FAVORED,
)
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


@pytest.fixture(scope="module")
def loss_1vme(protein_1vme) -> MolProbityLossFunction:
    loss = MolProbityLossFunction()
    loss.prepare(LossInputs.from_atom_array(protein_1vme, ensemble_size=1))
    return loss


def penalty(loss: MolProbityLossFunction, coordinates: torch.Tensor) -> torch.Tensor:
    """Call ``loss`` with placeholder per-atom inputs, which it ignores."""
    unused = torch.zeros(coordinates.shape[:-1])
    return loss(coordinates, unused.long(), unused, unused)


def as_batch(coords) -> torch.Tensor:
    return torch.as_tensor(np.asarray(coords), dtype=torch.float32).reshape(1, -1, 3)


def test_is_preparable_and_registered():
    loss = build_single_loss(Losses.MOLPROBITY, {}, LossBuildContext(structure={}))

    assert isinstance(loss, MolProbityLossFunction)
    assert isinstance(loss, PreparableLossFunctionProtocol)


@pytest.mark.parametrize(
    ("table", "phi_psi", "favored"),
    [
        # phi/psi and verdicts from wwPDB validation reports, which run MolProbity.
        ("general", (-113.6, 49.0), False),  # 2YL0 A59
        ("general", (-71.1, -60.9), False),  # 2YL0 A76, at the contour edge
        ("general", (-147.9, 10.2), False),  # 5SOP A129
        ("isoleucine_valine", (-50.2, -16.7), False),  # 5SOP A137
        ("general", (-70.1, -32.5), True),  # 1VME A319
    ],
)
def test_ramachandran_grids_reproduce_molprobity_verdicts(table, phi_psi, favored):
    value = _contour_grids()[("rama", table)](torch.tensor(phi_psi)).item()

    assert (value >= RAMACHANDRAN_FAVORED) is favored


def test_a_deposited_structure_scores_like_a_well_refined_one_and_noise_raises_it(
    loss_1vme, protein_1vme
):
    coordinates = as_batch(protein_1vme.coord)
    rng = np.random.default_rng(0)
    noisy = as_batch(protein_1vme.coord + rng.normal(scale=0.5, size=protein_1vme.coord.shape))

    assert loss_1vme.rotamer_outliers(coordinates).item() < 2.0
    assert loss_1vme.ramachandran_outside_favored(coordinates).item() < 6.0
    assert penalty(loss_1vme, noisy) > penalty(loss_1vme, coordinates)


def test_gradients_are_finite_and_a_descent_step_lowers_the_score(loss_1vme, protein_1vme):
    rng = np.random.default_rng(1)
    noisy = as_batch(protein_1vme.coord + rng.normal(scale=0.3, size=protein_1vme.coord.shape))
    noisy.requires_grad_(True)

    before = penalty(loss_1vme, noisy)
    before.backward()
    gradient = noisy.grad
    assert gradient is not None
    stepped = (noisy - 1e-2 * gradient / gradient.norm(dim=-1, keepdim=True).max()).detach()

    assert torch.isfinite(gradient).all()
    assert penalty(loss_1vme, stepped) < before
