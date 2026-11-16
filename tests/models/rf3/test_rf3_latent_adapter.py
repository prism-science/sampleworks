"""CPU check that the attribute names IT-opt uses for RF3's latents exist on RF3Conditioning.

IT-opt reaches a model's latents through the names in DEFAULT_SINGLE_REP_ATTR and
DEFAULT_PAIR_REP_ATTR (models/latent_adapter.py); for RF3 they are "s_trunk" and "z_trunk".
These are plain strings, and nothing in the code ties them to the fields of RF3Conditioning. If
a field were renamed, the table would still look fine and the mismatch would only show up on a
GPU, part-way into a latent optimization run. So this test builds a small RF3Conditioning from
random tensors and checks that AttrLatentIO can read and write it under those names, which turns
a rename into a fast-test failure instead. test_latent_adapter.py covers AttrLatentIO itself
against mock dataclasses; this is the only CPU test that uses the real class. Needs RF3
installed, but no checkpoint and no GPU; skips otherwise.
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
