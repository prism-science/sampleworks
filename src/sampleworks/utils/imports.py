# ruff: noqa: UP047

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, overload, TypeVar


BOLTZ_AVAILABLE = False
PROTENIX_AVAILABLE = False
RF3_AVAILABLE = False
PROTPARDELLE_AVAILABLE = False

try:
    from sampleworks.models.boltz.wrapper import Boltz1Wrapper, Boltz2Wrapper

    BOLTZ_AVAILABLE = True
    del Boltz1Wrapper, Boltz2Wrapper
except (ImportError, ModuleNotFoundError):
    pass

try:
    # we were testing whether we could load our own modules, but
    # that increases the likelihood of a circular import, and this
    # try/except construction makes those hard to debug, so just test
    # that the actual requirements are available.
    from protenix.model.protenix import Protenix
    from runner.msa_search import msa_search

    PROTENIX_AVAILABLE = True
    del Protenix, msa_search
except (ImportError, ModuleNotFoundError):
    pass

try:
    from sampleworks.models.rf3.wrapper import RF3Wrapper

    RF3_AVAILABLE = True
    del RF3Wrapper
except (ImportError, ModuleNotFoundError):
    pass

try:
    # Protpardelle's package import (via protpardelle.env) raises
    # NotADirectoryError (an OSError) when the model_params directory is not
    # set up, so catch OSError in addition to import errors.
    from sampleworks.models.protpardelle.wrapper import ProtpardelleWrapper

    PROTPARDELLE_AVAILABLE = True
    del ProtpardelleWrapper
except (ImportError, ModuleNotFoundError, OSError):
    pass

F = TypeVar("F", bound=Callable[..., Any])


def _skip_or_raise(error_message: str) -> None:
    """Skip the current test under pytest, otherwise raise ``ImportError``.

    Parameters
    ----------
    error_message: str
        Reason reported by the skip or the exception.

    Raises
    ------
    ImportError
        If pytest is not installed.
    """
    try:
        import pytest

        pytest.skip(error_message)
    except ImportError:
        raise ImportError(error_message) from None


@dataclass(frozen=True)
class ModelRequirement:
    """An optional model dependency, with a decorator and a check that enforce it.

    Parameters
    ----------
    available: bool
        Whether the dependency imported successfully.
    default_message: str
        Error message used when the caller supplies none.

    Examples
    --------
    >>> require_boltz = ModelRequirement(BOLTZ_AVAILABLE, "Boltz is missing").require
    >>> @require_boltz
    ... def train_boltz_model():
    ...     pass

    >>> @require_boltz("Custom error message")
    ... def custom_function():
    ...     pass
    """

    available: bool
    default_message: str

    @overload
    def require(self, message: F) -> F: ...
    @overload
    def require(self, message: str | None = None) -> Callable[[F], F]: ...
    def require(self, message: F | str | None = None) -> F | Callable[[F], F]:
        """Decorate a function so it skips (under pytest) or raises when unavailable.

        Parameters
        ----------
        message: str or Callable, optional
            Custom error message. If None, uses ``default_message``. A callable here
            is the decorated function itself, from bare ``@require`` usage.

        Returns
        -------
        Callable
            The decorated function for bare usage, otherwise the decorator.
        """
        if callable(message):
            # Re-dispatch so both ``@require`` and ``@require("msg")`` work.
            return self.require()(cast("F", message))

        error_message = message or self.default_message

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self.available:
                    _skip_or_raise(error_message)
                return func(*args, **kwargs)

            return cast("F", wrapper)

        return decorator

    def check(self, message: str | None = None) -> None:
        """Raise ``ImportError`` if the dependency is unavailable.

        Parameters
        ----------
        message: str, optional
            Custom error message. If None, uses ``default_message``.

        Raises
        ------
        ImportError
            If the dependency is unavailable.
        """
        if not self.available:
            raise ImportError(message or self.default_message)


def _missing_wrapper_message(display_name: str, pixi_environment: str) -> str:
    """Build the default message for a model wrapper that failed to import.

    Parameters
    ----------
    display_name: str
        Human-readable model name.
    pixi_environment: str
        Pixi environment that provides the model's dependencies.

    Returns
    -------
    str
        Error message with install instructions.
    """
    return (
        f"{display_name} model wrapper is not available. "
        f"Install with: pixi install -e {pixi_environment}"
    )


_BOLTZ = ModelRequirement(BOLTZ_AVAILABLE, _missing_wrapper_message("Boltz", "boltz"))
_PROTENIX = ModelRequirement(PROTENIX_AVAILABLE, _missing_wrapper_message("Protenix", "protenix"))
_RF3 = ModelRequirement(RF3_AVAILABLE, _missing_wrapper_message("RF3", "rf3"))
_PROTPARDELLE = ModelRequirement(
    PROTPARDELLE_AVAILABLE, _missing_wrapper_message("Protpardelle", "protpardelle")
)
_ANY_MODEL = ModelRequirement(
    BOLTZ_AVAILABLE or PROTENIX_AVAILABLE or RF3_AVAILABLE or PROTPARDELLE_AVAILABLE,
    "No model wrappers are available. "
    "Please install at least one model wrapper with the appropriate "
    "feature group: 'pixi install -e boltz', 'pixi install -e protenix', "
    "'pixi install -e rf3', or 'pixi install -e protpardelle'",
)

require_boltz = _BOLTZ.require
require_protenix = _PROTENIX.require
require_rf3 = _RF3.require
require_protpardelle = _PROTPARDELLE.require
require_any_model = _ANY_MODEL.require

check_boltz_available = _BOLTZ.check
check_protenix_available = _PROTENIX.check
check_rf3_available = _RF3.check
check_protpardelle_available = _PROTPARDELLE.check
check_any_model_available = _ANY_MODEL.check
