"""CPU check that the IT-opt latent seam resolves against the real RF3 conditioning.

``AttrLatentIO`` reaches the trunk latents by attribute name, and the guidance wiring looks
those names up in ``DEFAULT_SINGLE_REP_ATTR`` / ``DEFAULT_PAIR_REP_ATTR``. Nothing otherwise
ties either map to ``RF3Conditioning``, so renaming a field there would leave the maps stale
and surface only as a runtime error part-way into a GPU run. The sibling tests in
``test_latent_adapter.py`` cover the same seam against mock dataclasses; this one pins it to
the conditioning RF3 actually produces. No checkpoint and no GPU -- it skips unless RF3 is
installed. See docs/IT_OPT_DESIGN.md.
"""

import torch
from sampleworks.models.latent_adapter import (
    AttrLatentIO,
    DEFAULT_PAIR_REP_ATTR,
    DEFAULT_SINGLE_REP_ATTR,
)
from sampleworks.utils.imports import require_rf3, RF3_AVAILABLE


if RF3_AVAILABLE:
    from sampleworks.models.rf3.wrapper import RF3Conditioning


@require_rf3()
def test_attr_latent_io_resolves_rf3_latents():
    """AttrLatentIO built from the registered RF3 names reaches the real conditioning."""
    io = AttrLatentIO(
        single_attr=DEFAULT_SINGLE_REP_ATTR["rf3"],
        pair_attr=DEFAULT_PAIR_REP_ATTR["rf3"],
    )
    torch.manual_seed(0)
    cond = RF3Conditioning(
        s_inputs=torch.randn(1, 8, 4),
        s_trunk=torch.randn(1, 8, 4),  # [batch, tokens, d_s]
        z_trunk=torch.randn(1, 8, 8, 2),  # [batch, tokens, tokens, d_z]
        features={},
    )

    assert io.read_single(cond) is cond.s_trunk
    assert io.read_pair(cond) is cond.z_trunk
