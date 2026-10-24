"""GPU check that IT-opt's latents reach each model's denoiser (IT_OPT_TESTING.md, Level 0).

Two halves per model, one denoiser step each, no reward. Backward: leaves written in as the
single and pair latents receive a finite non-zero gradient. Forward with grad off: a perturbed
pair latent changes the output. The second half is what catches a cached latent-derived
conditioning being reused, as Boltz2 did before #442; the first half passed on that code
whenever grad was on. Needs a checkpoint and a GPU; the wrapper fixtures skip otherwise (#422).
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
    """Injected s/z leaves get a gradient, and a changed z changes the output with grad off."""
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

    # Backward half: Level 0 as the doc describes it.
    s_leaf = io.read_single(cond).detach().clone().requires_grad_(True)
    z_leaf = z_trunk.detach().clone().requires_grad_(True)
    live = GenerativeModelInput(conditioning=io.write_pair(io.write_single(cond, s_leaf), z_leaf))
    # .backward(), not autograd.grad(): Boltz1's trunk runs under fairscale activation
    # checkpointing, which refuses autograd.grad(). An unused leaf keeps grad None either way.
    wrapper.step(x_t, T, features=live).square().sum().backward()
    for name, grad in (("single", s_leaf.grad), ("pair", z_leaf.grad)):
        assert grad is not None, f"{wrapper_type}: nothing reached the {name} latent"
        assert torch.isfinite(grad).all(), f"{wrapper_type}: {name} gradient is not finite"
        assert grad.abs().sum() > 0, f"{wrapper_type}: {name} gradient is zero"

    # Forward half: grad off, as the final frozen-latent pass runs.
    with torch.no_grad():
        reference = wrapper.step(x_t, T, features=features)
        perturbed = io.write_pair(cond, z_trunk + 3 * z_trunk.std() * torch.randn_like(z_trunk))
        moved = wrapper.step(x_t, T, features=GenerativeModelInput(conditioning=perturbed))
    assert not torch.allclose(moved, reference), (
        f"{wrapper_type}: a changed z left the output unchanged"
    )
