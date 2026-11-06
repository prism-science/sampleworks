"""Tests for the bond-geometry penalty as a composable, two-phase loss."""

from pathlib import Path

import numpy as np
import pytest
import torch
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, connect_via_residue_names
from sampleworks.core.losses.composite import CompositeLoss
from sampleworks.core.losses.config import build_loss, LossConfig
from sampleworks.core.losses.geometry import BondGeometryLoss, BondGeometryLossFunction
from sampleworks.core.losses.protocol import LossInputs, PreparableLossFunctionProtocol
from sampleworks.core.losses.registry import build_single_loss, LossBuildContext
from sampleworks.utils.guidance_constants import Losses


@pytest.fixture(scope="module")
def atom_array_1vme(resources_dir: Path) -> AtomArray:
    """Single-conformer 1vme protein atoms."""
    atom_array = load_any(
        resources_dir / "1vme" / "1VME_single_001_density_input.cif",
        extra_fields=["occupancy", "b_factor"],
    )
    if hasattr(atom_array, "stack_depth"):
        atom_array = atom_array[0]
    return atom_array


@pytest.fixture
def prepared_loss(atom_array_1vme: AtomArray) -> tuple[BondGeometryLossFunction, LossInputs]:
    """Bond-geometry loss prepared on 1vme, with its single-member loss inputs."""
    loss_inputs = LossInputs.from_atom_array(atom_array_1vme, ensemble_size=1)
    loss = build_single_loss(Losses.BOND_GEOMETRY, {}, LossBuildContext(structure={}))
    assert isinstance(loss, BondGeometryLossFunction)
    loss.prepare(loss_inputs, device="cpu")
    return loss, loss_inputs


def _score(
    loss: BondGeometryLossFunction, inputs: LossInputs, coords: torch.Tensor
) -> torch.Tensor:
    return loss(coords, inputs.elements, inputs.b_factors, inputs.occupancies)


def test_registered_loss_is_two_phase():
    """The registry builds an unprepared loss that the trajectory scalers will prepare."""
    loss = build_single_loss(Losses.BOND_GEOMETRY, {}, LossBuildContext(structure={}))

    assert isinstance(loss, PreparableLossFunctionProtocol)
    with pytest.raises(RuntimeError, match="prepare"):
        loss(torch.zeros(1, 3, 3), None, None, None)  # ty: ignore[invalid-argument-type]


def test_stretched_bond_raises_penalty(prepared_loss, atom_array_1vme):
    """Pulling one bonded atom 1 A away from its partner increases the penalty."""
    loss, inputs = prepared_loss
    coords = inputs.input_coords.clone()  # [1, n_atoms, 3]
    i, j = connect_via_residue_names(atom_array_1vme).as_array()[0, :2].astype(np.int64)

    direction = coords[0, j] - coords[0, i]
    stretched = coords.clone()
    stretched[0, j] += direction / direction.norm()

    assert _score(loss, inputs, stretched) > _score(loss, inputs, coords)


def test_clash_raises_collision_term(atom_array_1vme):
    """Placing two distant atoms 0.5 A apart increases the clash term."""
    penalty = BondGeometryLoss(atom_array_1vme, weight=1.0, device="cpu")
    coords = torch.from_numpy(atom_array_1vme.coord).float().unsqueeze(0)  # [1, n_atoms, 3]
    clashed = coords.clone()
    clashed[0, -1] = coords[0, 0] + torch.tensor([0.5, 0.0, 0.0])

    assert penalty.collision_loss(clashed) > penalty.collision_loss(coords)


def test_gradient_step_repairs_distortion(prepared_loss):
    """A gradient step on randomly distorted coordinates lowers the penalty."""
    loss, inputs = prepared_loss
    torch.manual_seed(0)
    distorted = (inputs.input_coords + 0.5 * torch.randn_like(inputs.input_coords)).requires_grad_()

    before = _score(loss, inputs, distorted)
    before.backward()
    grad = distorted.grad
    assert grad is not None and torch.isfinite(grad).all()

    with torch.no_grad():
        after = _score(loss, inputs, distorted - 1e-3 * grad)
    assert after < before


@pytest.mark.gpu
def test_composes_with_density_loss(
    structure_1vme_density, density_path_1vme, atom_array_1vme, device
):
    """A density + bond-geometry configuration builds, prepares and scores as one loss."""
    config = LossConfig.from_mapping(
        {
            "real_space_density": {
                "weight": 0.5,
                "loss_options": {"density": str(density_path_1vme), "resolution": 1.8},
            },
            "bond_geometry": {"weight": 0.5},
        }
    )
    loss = build_loss(config, LossBuildContext(structure=structure_1vme_density, device=device))
    assert isinstance(loss, CompositeLoss)

    inputs = LossInputs.from_atom_array(atom_array_1vme, ensemble_size=2, device=device)
    loss.prepare(inputs, device=device)
    coords = inputs.input_coords.clone().requires_grad_()
    value = loss(coords, inputs.elements, inputs.b_factors, inputs.occupancies)
    value.backward()

    assert torch.isfinite(value)
    assert coords.grad is not None and torch.isfinite(coords.grad).all()
