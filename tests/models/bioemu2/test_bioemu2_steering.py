"""Tests for the BioEmu2 native-steering baseline on the random-weight model."""

import pytest
import torch
from sampleworks.models.bioemu2.steering import BioEmu2NativeSteering
from sampleworks.models.bioemu2.wrapper import annotate_structure_for_bioemu2, BioEmu2Wrapper


class DistanceToOriginLoss:
    """Mean squared distance of the atoms from the origin; records each call's batch size."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def __call__(self, coordinates, elements, b_factors, occupancies):
        self.batch_sizes.append(coordinates.shape[0])
        return coordinates.square().sum(-1).mean()


@pytest.fixture
def structure(structure_2yl0_density: dict) -> dict:
    return annotate_structure_for_bioemu2(structure_2yl0_density, use_msa_server=False)


@pytest.mark.parametrize("weight_update", ["smc", "guidance"])
def test_returns_one_best_particle_per_member(
    tiny_bioemu2_wrapper: BioEmu2Wrapper, structure: dict, weight_update
):
    loss = DistanceToOriginLoss()
    steering = BioEmu2NativeSteering(ensemble_size=2, num_steps=3, weight_update=weight_update)
    out = steering.sample(structure, tiny_bioemu2_wrapper, loss_function=loss, num_particles=3)

    features = tiny_bioemu2_wrapper.featurize(structure)
    assert features.conditioning is not None
    final = torch.as_tensor(out.final_state)
    assert final.shape == (2, features.conditioning.num_atoms, 3)
    assert torch.isfinite(final).all()
    assert out.losses is not None and len(out.losses) == 2
    # The potential scores one structure per loss call.
    assert set(loss.batch_sizes) == {1}


def test_needs_particles_to_resample(tiny_bioemu2_wrapper: BioEmu2Wrapper, structure: dict):
    with pytest.raises(ValueError, match="num_particles"):
        BioEmu2NativeSteering().sample(
            structure, tiny_bioemu2_wrapper, loss_function=DistanceToOriginLoss(), num_particles=1
        )
