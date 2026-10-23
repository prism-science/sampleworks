"""Tests for the two-phase loss preparation hook."""

import torch
from sampleworks.core.losses.protocol import (
    LossFunctionProtocol,
    LossInputs,
    PreparableLossFunctionProtocol,
    prepare_loss_if_needed,
)

from tests.mocks import MockGradientLossFunction, MockPreparableLossFunction
from tests.utils.atom_array_builders import build_test_atom_array


def _inputs(n_atoms: int) -> LossInputs:
    """Single-conformer loss inputs over ``n_atoms`` test atoms."""
    return LossInputs.from_atom_array(build_test_atom_array(n_atoms=n_atoms), ensemble_size=1)


def test_preparable_loss_satisfies_both_protocols():
    """A loss with prepare() is still an ordinary loss function."""
    loss = MockPreparableLossFunction()

    assert isinstance(loss, LossFunctionProtocol)
    assert isinstance(loss, PreparableLossFunctionProtocol)


def test_plain_loss_is_not_preparable():
    """Structural typing must not classify every loss as two-phase."""
    assert not isinstance(MockGradientLossFunction(), PreparableLossFunctionProtocol)


def test_prepare_hook_forwards_loss_inputs_and_device():
    """The loss is bound to the very loss inputs and device the caller passed."""
    loss = MockPreparableLossFunction()
    inputs = _inputs(7)

    prepare_loss_if_needed(loss, inputs, device=torch.device("cpu"))

    assert loss.prepared_with == [(7, "cpu")]
    # The same object `__call__` will be fed, not a copy or a derived atom array.
    assert loss.prepared_inputs == [inputs]
    assert loss.prepared_inputs[0] is inputs


def test_prepare_hook_is_a_no_op_for_losses_that_do_not_need_it():
    """Callers can prepare unconditionally, so one-phase losses must be untouched."""
    loss = MockGradientLossFunction()

    prepare_loss_if_needed(loss, _inputs(4), device="cpu")

    # 0.5 * ||ones(1, 4, 3)||^2: the loss scores exactly as if the hook had never run.
    assert loss(torch.ones(1, 4, 3)) == 6.0


def test_prepare_is_rerunnable_for_a_new_topology():
    """Preparing again rebinds, so a loss can move between topologies or devices."""
    loss = MockPreparableLossFunction()

    prepare_loss_if_needed(loss, _inputs(3), device="cpu")
    prepare_loss_if_needed(loss, _inputs(5), device="cpu")

    assert loss.prepared_with == [(3, "cpu"), (5, "cpu")]
