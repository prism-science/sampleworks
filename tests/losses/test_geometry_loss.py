"""Tests for ``BondGeometryLoss``, the bond-length and steric-clash penalty.

The toy systems are a water molecule (two O-H bonds, one non-bonded H-H pair) and
pairs of single-atom residues (no bonds, one non-bonded pair), placed by hand so the
expected hinge values follow from the covalent radii alone. The last test scores the
deposited 1VME protein against a noised copy of it.
"""

import numpy as np
import pytest
import torch
from biotite.structure import array, Atom, AtomArray, AtomArrayStack, filter_amino_acids
from biotite.structure.info import residue
from sampleworks.core.losses.geometry import BondGeometryLoss
from sampleworks.utils.elements import covalent_radius


BOND_TOLERANCE = 0.2
CLASH_PADDING = 0.4
OH_IDEAL_LENGTH = covalent_radius("O") + covalent_radius("H")


def make_loss(atom_array: AtomArray, weight: float = 1.0, **kwargs) -> BondGeometryLoss:
    """A ``BondGeometryLoss`` on CPU with the default tolerance and padding.

    Parameters
    ----------
    atom_array
        Atoms whose order matches the coordinates the loss will score.
    weight
        Scale on the summed penalty.
    **kwargs
        Further keyword arguments for ``BondGeometryLoss``.

    Returns
    -------
    BondGeometryLoss
        The constructed loss.
    """
    return BondGeometryLoss(
        atom_array,
        weight,
        "cpu",
        bond_tolerance=BOND_TOLERANCE,
        clash_padding=CLASH_PADDING,
        **kwargs,
    )


def water_coords(bond_length: float) -> torch.Tensor:
    """Water with both O-H bonds of ``bond_length`` at a right angle, in O, H1, H2 order.

    Parameters
    ----------
    bond_length
        Length of each O-H bond, in Å.

    Returns
    -------
    torch.Tensor
        Coordinates, shape [1, 3, 3] (ensemble member, atom, xyz).
    """
    return torch.tensor([[[0.0, 0.0, 0.0], [bond_length, 0.0, 0.0], [0.0, bond_length, 0.0]]])


def single_atom_residues(element: str) -> AtomArray:
    """Two unbonded single-atom residues of ``element``.

    ``UNX`` (unknown atom or ion) has no bonds in the component dictionary, so the
    pair is scored for clashes and has no bond term.

    Parameters
    ----------
    element
        Element symbol given to both atoms.

    Returns
    -------
    AtomArray
        Two atoms in residues 1 and 2.
    """
    return array(
        [
            Atom(
                np.zeros(3),
                chain_id="A",
                res_id=res_id,
                res_name="UNX",
                atom_name="UNK",
                element=element,
            )
            for res_id in (1, 2)
        ]
    )


def pair_coords(*distances: float) -> torch.Tensor:
    """Two atoms ``distance`` apart along x, one ensemble member per distance.

    Parameters
    ----------
    *distances
        Separation for each ensemble member, in Å.

    Returns
    -------
    torch.Tensor
        Coordinates, shape [len(distances), 2, 3].
    """
    return torch.tensor([[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]] for distance in distances])


@pytest.fixture
def water() -> AtomArray:
    """Water from the component dictionary, with its O, H1, H2 atom names and elements."""
    return residue("HOH")


@pytest.mark.parametrize(
    ["deviation", "expected"],
    [[0.0, 0.0], [0.9 * BOND_TOLERANCE, 0.0], [-0.9 * BOND_TOLERANCE, 0.0]],
)
def test_bond_deviation_values(water, deviation, expected):
    """Bonds stretched or compressed by less than ``bond_tolerance`` cost nothing.

    Parameters
    ----------
    deviation
        Signed change of each O-H bond from its ideal length, in Å.
    """
    loss = make_loss(water)
    assert loss.bond_length_loss(water_coords(OH_IDEAL_LENGTH + deviation)) == expected


@pytest.mark.parametrize("bond_power", [1, 2])
@pytest.mark.parametrize("direction", [1.0, -1.0], ids=["stretched", "compressed"])
def test_bond_excess_beyond_tolerance_is_raised_to_bond_power(water, bond_power, direction):
    """Past the tolerance, each bond costs its excess deviation to the power ``bond_power``.

    Stretching and compressing by the same amount cost the same; water has two bonds.

    Parameters
    ----------
    bond_power
        Exponent on the bond-length hinge.
    direction
        +1 stretches both bonds, -1 compresses them.
    """
    excess = 0.3
    loss = make_loss(water, bond_power=bond_power)
    coords = water_coords(OH_IDEAL_LENGTH + direction * (BOND_TOLERANCE + excess))
    assert loss.bond_length_loss(coords).item() == pytest.approx(2 * excess**bond_power, rel=1e-4)


def test_bonded_atoms_are_not_scored_as_clashes(water):
    """An ideal O-H bond is shorter than the O-H clash distance, yet is not a clash.

    The H-H pair is non-bonded but far enough apart at a right angle to be clear.
    """
    loss = make_loss(water)
    assert loss.collision_loss(water_coords(OH_IDEAL_LENGTH)) == 0.0


def test_clash_penalty_grows_as_atoms_overlap():
    """Non-bonded atoms are free outside their clash distance and cost more the closer they get."""
    clash_distance = 2 * covalent_radius("C") + CLASH_PADDING
    loss = make_loss(single_atom_residues("C"))

    separations = [clash_distance + 0.1, clash_distance - 0.1, clash_distance - 0.3]
    penalties = [loss.collision_loss(pair_coords(d)).item() for d in separations]

    assert penalties[0] == 0.0
    assert 0.0 < penalties[1] < penalties[2]


def test_clash_gradient_pushes_atoms_apart():
    """A gradient-descent step on a clashing pair increases their separation."""
    loss = make_loss(single_atom_residues("C"))
    distance = 1.2
    coords = pair_coords(distance).requires_grad_(True)
    loss.collision_loss(coords).backward()

    assert coords.grad is not None
    stepped = coords.detach() - 0.01 * coords.grad
    assert (stepped[0, 1] - stepped[0, 0]).norm() > distance


def test_every_clashing_ensemble_member_receives_a_gradient():
    """A clash in a member that is not the worst one is still penalized."""
    loss = make_loss(single_atom_residues("C"))
    coords = pair_coords(1.0, 1.5).requires_grad_(True)
    loss.collision_loss(coords).backward()

    assert coords.grad is not None
    assert (coords.grad.norm(dim=(-2, -1)) > 0).all()


def test_separated_ensemble_member_does_not_offset_a_clash():
    """A member with no clash neither cancels another member's clash nor receives a gradient."""
    loss = make_loss(single_atom_residues("C"))
    coords = pair_coords(1.2, 5.0).requires_grad_(True)
    penalty = loss.collision_loss(coords)
    penalty.backward()

    assert penalty > 0
    assert coords.grad is not None
    assert coords.grad[0].norm() > 0
    assert coords.grad[1].norm() == 0


def test_unrecognized_element_uses_the_fallback_radius():
    """An unknown element symbol builds a loss, with gemmi's 0.50 Å unknown-element radius.

    Two such atoms clash below 0.50 + 0.50 + padding = 1.40 Å and not above it, where
    two carbons at the same separation would.
    """
    unknown = make_loss(single_atom_residues("?"))
    carbon = make_loss(single_atom_residues("C"))
    distance = 1.0 + CLASH_PADDING  # 2 * gemmi's unknown element radius + padding.

    assert unknown.collision_loss(pair_coords(distance - 0.05)) > 0
    assert unknown.collision_loss(pair_coords(distance)) == 0.0
    assert carbon.collision_loss(pair_coords(distance + 0.05)) > 0


@pytest.mark.parametrize("weight", [0.0, 0.5, 3.0])
def test_call_is_the_weighted_sum_of_both_penalties(water, weight):
    """``__call__`` scales the bond and clash penalties together by ``weight``.

    Parameters
    ----------
    weight
        Scale on the summed penalty.
    """
    loss = make_loss(water, weight=weight)
    coords = water_coords(OH_IDEAL_LENGTH - 0.5)  # bond and H-H clash penalties both nonzero
    bond = loss.bond_length_loss(coords)
    clash = loss.collision_loss(coords)

    assert bond > 0 and clash > 0
    assert loss(coords).item() == pytest.approx(weight * (bond + clash).item())


def test_deposited_protein_scores_far_below_a_noised_copy(structure_1vme):
    """Deposited 1VME has near-ideal bonds and few clashes; noise breaks both.

    A few gradient-descent steps on the noised coordinates then lower the penalty.
    """
    atom_array = structure_1vme["asym_unit"]
    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    modelled = np.isfinite(atom_array.coord).all(axis=-1)
    protein = atom_array[filter_amino_acids(atom_array) & modelled & (atom_array.res_id <= 40)]
    loss = make_loss(protein)

    deposited = torch.tensor(protein.coord[None], dtype=torch.float32)  # [1, n_atoms, 3]
    generator = torch.Generator().manual_seed(0)
    noised = deposited + 0.3 * torch.randn(deposited.shape, generator=generator)

    assert loss.bond_length_loss(deposited) < 0.01 * loss.bond_length_loss(noised)
    assert loss.collision_loss(deposited) < 0.1 * loss.collision_loss(noised)

    coords = noised.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([coords], lr=1e-3)
    initial = loss(coords).item()
    for _ in range(20):
        optimizer.zero_grad()
        loss(coords).backward()
        optimizer.step()
    assert loss(coords).item() < initial
