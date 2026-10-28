"""Tests for scaling an experimental map to a model's computed density."""

import pytest
import torch
from sampleworks.core.forward_models.xray.real_space_density import scale_map


@pytest.fixture
def model_map_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    """A positive model density on a 16^3 grid and a mask over its central 8^3 box."""
    generator = torch.Generator().manual_seed(0)
    model_map = torch.rand(16, 16, 16, generator=generator) * 3.0 + 1.0
    mask = torch.zeros(16, 16, 16, dtype=torch.bool)
    mask[4:12, 4:12, 4:12] = True
    return model_map, mask


def correlated_map(model_map: torch.Tensor, mask: torch.Tensor, correlation: float) -> torch.Tensor:
    """A map with the model's masked mean and std and the given masked correlation."""
    generator = torch.Generator().manual_seed(1)
    noise = torch.randn(model_map.shape, generator=generator)
    model_masked, noise_masked = model_map[mask], noise[mask]
    noise_masked = noise_masked - noise_masked.mean()
    model_centered = model_masked - model_masked.mean()
    # Remove the model component so noise is exactly uncorrelated with the model in the mask.
    noise_masked = noise_masked - model_centered * (
        torch.dot(noise_masked, model_centered) / torch.dot(model_centered, model_centered)
    )
    xmap = torch.zeros_like(model_map)
    xmap[mask] = model_masked.mean() + model_masked.std() * (
        correlation * model_centered / model_centered.std()
        + (1 - correlation**2) ** 0.5 * noise_masked / noise_masked.std()
    )
    return xmap


def test_target_does_not_depend_on_map_units(model_map_and_mask):
    """A map that already matches the model's mean and std, but correlates with it only at 0.5,
    must scale the same however it is stored."""
    model_map, mask = model_map_and_mask
    xmap = correlated_map(model_map, mask, correlation=0.5)

    reference = scale_map(xmap, model_map, mask)
    for factor in (1 / 30, 30):
        torch.testing.assert_close(scale_map(xmap * factor, model_map, mask), reference)


def test_anticorrelated_map_keeps_its_contrast(model_map_and_mask):
    """A poor fit must not flatten or invert the target: the scaled map stays a positive
    multiple of the input, with the model's masked spread."""
    model_map, mask = model_map_and_mask
    xmap = correlated_map(model_map, mask, correlation=-0.5)

    with pytest.warns(UserWarning, match="not positively correlated"):
        scaled = scale_map(xmap, model_map, mask)

    correlation_with_input = torch.corrcoef(torch.stack([scaled[mask], xmap[mask]]))[0, 1]
    torch.testing.assert_close(correlation_with_input, torch.tensor(1.0))
    torch.testing.assert_close(scaled[mask].std(), model_map[mask].std())
    torch.testing.assert_close(scaled[mask].mean(), model_map[mask].mean())


def test_constant_map_cannot_be_scaled(model_map_and_mask):
    model_map, mask = model_map_and_mask
    with pytest.raises(ValueError, match="constant within the mask"):
        scale_map(torch.full_like(model_map, 2.0), model_map, mask)
