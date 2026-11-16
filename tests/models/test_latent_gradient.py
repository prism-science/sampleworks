"""GPU check that each model's denoiser really responds to the latents IT-opt optimizes.

IT-opt writes copies of the trunk latents s and z with requires_grad onto the conditioning,
backpropagates a loss on the denoised structure to them, and finally samples with the optimized
latents frozen. The wrappers were written for coordinate guidance, where those latents are
cached and detached, so two things can go wrong without any error: the gradient never reaches
s or z, and the optimizer just spins; or the denoiser decodes conditioning cached from the
original latents, so the optimized ones never reach the output (Boltz2 did this before #442).
Both only show up as bad numbers after a long GPU run. So for every model in
DEFAULT_SINGLE_REP_ATTR this test runs one denoiser step and makes the two checks described
below. It needs a checkpoint and a GPU; the wrapper fixtures skip otherwise (#422).
"""

import pytest
import torch
from sampleworks.models.latent_adapter import (
    AttrLatentIO,
    DEFAULT_PAIR_REP_ATTR,
    DEFAULT_SINGLE_REP_ATTR,
)
from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.guidance_constants import StructurePredictor

from tests.conftest import annotate_structure_for_wrapper_type, get_fixture_name_for_wrapper_type


# The attribute tables are the list of models IT-opt can address; a new model joins by being
# added there.
LATENT_MODELS = [StructurePredictor(name) for name in DEFAULT_SINGLE_REP_ATTR]
T = 1.0  # every wrapper takes a float t; unit prior noise at sigma 1 is a coherent input

# Protenix caches z-derived diffusion inputs at featurize time, and with that cache on the denoiser
# never reads z_trunk, so its gradient is None. _run_guidance turns the cache off for latent
# optimization; the test runs that same configuration.
ANNOTATE_KWARGS = {StructurePredictor.PROTENIX: {"enable_diffusion_shared_vars_cache": False}}


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", LATENT_MODELS, ids=lambda w: w.value)
def test_latents_reach_the_denoiser(wrapper_type, structure_2yl0, temp_output_dir, request):
    """A gradient reaches s and z, and a changed z changes the output with gradients off."""
    wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
    structure = annotate_structure_for_wrapper_type(
        wrapper_type, structure_2yl0, temp_output_dir, **ANNOTATE_KWARGS.get(wrapper_type, {})
    )
    with torch.no_grad():
        features = wrapper.featurize(structure)
    cond = features.conditioning
    io = AttrLatentIO(DEFAULT_SINGLE_REP_ATTR[wrapper_type], DEFAULT_PAIR_REP_ATTR[wrapper_type])
    z_trunk = io.read_pair(cond)
    assert z_trunk is not None  # every model in the table has a pair latent
    x_t = wrapper.initialize_from_prior(1, features)

    # Check 1, the Level-0 check in docs/IT_OPT_TESTING.md section 2: a gradient reaches s and z.
    # IT-opt optimizes s and z through copies with requires_grad on: autograd stores a gradient
    # on the copies and the optimizer updates them, while the trunk's originals stay untouched.
    # The test writes the same copies onto the conditioning.
    s_copy = io.read_single(cond).detach().clone().requires_grad_(True)
    z_copy = z_trunk.detach().clone().requires_grad_(True)
    live = GenerativeModelInput(conditioning=io.write_pair(io.write_single(cond, s_copy), z_copy))
    wrapper.step(x_t, T, features=live).square().sum().backward()
    for name, grad in (("single", s_copy.grad), ("pair", z_copy.grad)):
        assert grad is not None, f"{wrapper_type}: nothing reached the {name} latent"
        assert torch.isfinite(grad).all(), f"{wrapper_type}: {name} gradient is not finite"
        assert grad.abs().sum() > 0, f"{wrapper_type}: {name} gradient is zero"

    # Check 2: with gradients off, as in IT-opt's final clean sampling pass, a changed z changes
    # the output. A wrapper that decodes conditioning cached from the original latents passes
    # check 1 and fails here.
    with torch.no_grad():
        reference = wrapper.step(x_t, T, features=features)
        perturbed = io.write_pair(cond, z_trunk + 3 * z_trunk.std() * torch.randn_like(z_trunk))
        moved = wrapper.step(x_t, T, features=GenerativeModelInput(conditioning=perturbed))
    assert not torch.allclose(moved, reference), (
        f"{wrapper_type}: a changed z left the output unchanged"
    )
