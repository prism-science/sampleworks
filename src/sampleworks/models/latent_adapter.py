"""Read and write a model's post-trunk latents (IT-opt plumbing).

The latents are the *single* representation ``s`` and the *pair* representation ``z``. They live on
the model's *conditioning* -- the bundle of inputs ``featurize`` produces and hands to the ``step``
(denoise) call, a (frozen) dataclass on which ``s`` and ``z`` are named attributes. This module lets
inference-time latent optimization (IT-opt) read those attributes and swap them out, so the scaler
in ``core/scalers/latent_optimization.py`` can optimize them against an experimental reward. See
``docs/IT_OPT_DESIGN.md``.

What each piece is for
----------------------
- :class:`AttrLatentIO` -- reads/writes ``s`` and ``z`` by attribute name. This is the piece the
  IT-opt scaler actually uses: it reads the baseline latents once, holds its own optimizable copies,
  and writes them back each step.
- :class:`LatentIO` -- the protocol (read/write contract) that :class:`AttrLatentIO` satisfies.

Design goal
-----------
- **Minimal, model-agnostic.** The only model-specific knowledge is *which attribute of the
  conditioning holds each representation* -- ``"s"``/``"z"`` for Boltz, ``"s_trunk"``/``"z_trunk"``
  for Protenix/RF3. No model package is imported here, so a 4th or 5th model is one more pair of
  strings, not new code.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

from torch import Tensor


# ---- Reading the type hints in this file --------------------------------------
# A ": type" after a name (or "-> type" after a function) is only a HINT -- it CLAIMS what a
# value should be, but nothing enforces it: pass the wrong type and Python still runs the code,
# and deleting every hint changes nothing. Hints are for humans (and optional checkers like ty).
# (In the table, "|" means "or".)
#
#   with the hint                    plain Python               what it claims (useless in runtime)
#   attr: str                        attr                       should be a string
#   single: Tensor                   single                     should be a Tensor (~ a numpy array)
#   pair_attr: str | None = None     pair_attr = None           should be a str, or None
#   d: dict[str, str] = {}           d = {}                     should be a dict, string -> string
#   f(...) -> Tensor | None          f(...)                     f should return a Tensor or None
# -------------------------------------------------------------------------------

# Convenience maps of the representation attribute name per model. Documentation/
# config only -- the adapter never imports these models.
DEFAULT_SINGLE_REP_ATTR: dict[str, str] = {
    "boltz1": "s",
    "boltz2": "s",
    "protenix": "s_trunk",
    "rf3": "s_trunk",
}
DEFAULT_PAIR_REP_ATTR: dict[str, str] = {
    "boltz1": "z",
    "boltz2": "z",
    "protenix": "z_trunk",
    "rf3": "z_trunk",
}


# "conditioning" throughout this file is the model's conditioning object: a (frozen) dataclass
# carrying the latents ``s``/``z`` (and other cached state) as named attributes.
class AttrLatentIO:
    """A general-purpose :class:`LatentIO` addressing each representation by attribute name.

    Works for any (dataclass) conditioning whose representations are stored in
    named attributes. Writes use :func:`dataclasses.replace`, so non-tensor
    sidecar state on the conditioning (e.g. RF3's chiral tracking arrays) is
    preserved untouched. All three model wrappers use
    ``@dataclass(frozen=True, slots=True)`` conditioning, which is exactly this
    case.

    Parameters
    ----------
    single_attr
        Attribute holding the single representation, e.g. ``"s"`` (Boltz) or
        ``"s_trunk"`` (Protenix/RF3).
    pair_attr
        Attribute holding the pair representation, e.g. ``"z"`` / ``"z_trunk"``.
        When ``None`` (default), the pair accessors are no-ops -- single-rep-only
        behaviour, matching the original scaffold.
    """

    def __init__(self, single_attr: str, pair_attr: str | None = None):
        self.single_attr = single_attr
        self.pair_attr = pair_attr

    def read_single(self, conditioning) -> Tensor:
        # No default: a name that does not match this model's conditioning is a configuration
        # error, and AttributeError says so more usefully than a silent None would.
        return getattr(conditioning, self.single_attr)

    def write_single(self, conditioning, single: Tensor):
        return dataclasses.replace(conditioning, **{self.single_attr: single})  # ty: ignore

    def read_pair(self, conditioning) -> Tensor | None:
        # None here means "this io addresses no pair rep", which is a supported configuration --
        # unlike a missing attribute above.
        if self.pair_attr is None:
            return None
        return getattr(conditioning, self.pair_attr)

    def write_pair(self, conditioning, pair: Tensor):
        if self.pair_attr is None:
            return conditioning
        return dataclasses.replace(conditioning, **{self.pair_attr: pair})  # ty: ignore


# ============================ protocol (template / interface) ============================
# A Protocol is just a checklist of methods a class must have -- think a C header, or an abstract
# base class, but *structural*: any class that already has these methods counts as a LatentIO
# without inheriting from anything. This definition does not run; it only documents the contract and
# lets the ty checker flag a mismatch. @runtime_checkable also lets isinstance(obj, LatentIO) work,
# which the unit tests use. The real implementation is above: AttrLatentIO is the LatentIO.


@runtime_checkable
class LatentIO(Protocol):
    """Reads/writes the single and pair representations on a conditioning object.

    Implementations encapsulate the *only* model-specific knowledge in this
    module: where each representation lives on the conditioning object. Pair-rep
    methods return ``None`` / pass through unchanged when the implementation is
    single-rep only. Each write returns a copy of the conditioning with one
    representation replaced.
    """

    def read_single(self, conditioning) -> Tensor | None:
        """Return the single representation tensor, or ``None`` if unavailable."""
        ...

    def write_single(self, conditioning, single: Tensor):
        """Return a copy of ``conditioning`` with the single representation replaced."""
        ...

    def read_pair(self, conditioning) -> Tensor | None:
        """Return the pair representation tensor, or ``None`` if unavailable."""
        ...

    def write_pair(self, conditioning, pair: Tensor):
        """Return a copy of ``conditioning`` with the pair representation replaced."""
        ...
