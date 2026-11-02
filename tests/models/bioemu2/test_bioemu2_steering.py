"""Tests for FK steering configured like BioEmu2, on the random-weight model."""

import torch
from sampleworks.core.scalers.step_scalers import DataSpaceDPSScaler
from sampleworks.models.bioemu2.sampler import build_bioemu2_edm_sampler
from sampleworks.models.bioemu2.steering import build_bioemu_fksteering
from sampleworks.models.bioemu2.wrapper import annotate_structure_for_bioemu2, BioEmu2Wrapper

from tests.mocks.losses import MockGradientLossFunction


def test_returns_every_particle(tiny_bioemu2_wrapper: BioEmu2Wrapper, structure_2yl0_density: dict):
    structure = annotate_structure_for_bioemu2(structure_2yl0_density, use_msa_server=False)
    steering = build_bioemu_fksteering(num_steps=4, resampling_interval=2)
    out = steering.sample(
        structure=structure,
        model=tiny_bioemu2_wrapper,
        sampler=build_bioemu2_edm_sampler(tiny_bioemu2_wrapper),
        step_scaler=DataSpaceDPSScaler(step_size=0.0),
        loss_function=MockGradientLossFunction(),
        num_particles=3,
    )

    features = tiny_bioemu2_wrapper.featurize(structure)
    assert features.conditioning is not None
    final = torch.as_tensor(out.final_state)
    assert final.shape == (3, features.conditioning.num_atoms, 3)
    assert torch.isfinite(final).all()
