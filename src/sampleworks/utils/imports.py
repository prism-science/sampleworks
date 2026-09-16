# ruff: noqa: UP047

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, cast, overload, TypeVar


BOLTZ_AVAILABLE = False
PROTENIX_AVAILABLE = False
RF3_AVAILABLE = False
PROTPARDELLE_AVAILABLE = False
TORCHREF_AVAILABLE = False

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

try:
    # torchref backs the reciprocal-space reward in
    # sampleworks.core.rewards.torchref_rewards, which imports it lazily so this
    # flag is the only thing that has to know whether it is installed.
    from torchref import ModelFT, read_mtz

    TORCHREF_AVAILABLE = True
    del ModelFT, read_mtz
except (ImportError, ModuleNotFoundError):
    pass

F = TypeVar("F", bound=Callable[..., Any])


@overload
def require_boltz(message: F) -> F: ...
@overload
def require_boltz(message: str | None = None) -> Callable[[F], F]: ...
def require_boltz(message: F | str | None = None) -> F | Callable[[F], F]:
    """Decorator to require Boltz model availability.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @require_boltz
    ... def train_boltz_model():
    ...     pass

    >>> @require_boltz("Custom error message")
    ... def custom_function():
    ...     pass
    """
    if callable(message):
        # Bare ``@require_boltz`` usage: the decorated function arrives as ``message``.
        # Re-dispatch so both ``@require_boltz`` and ``@require_boltz("msg")`` work.
        return require_boltz()(cast("F", message))

    default_message = "Boltz model wrapper is not available. Install with: pixi install -e boltz"

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not BOLTZ_AVAILABLE:
                error_msg = message or default_message
                try:
                    import pytest

                    pytest.skip(error_msg)
                except ImportError:
                    raise ImportError(error_msg) from None
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


@overload
def require_protenix(message: F) -> F: ...
@overload
def require_protenix(message: str | None = None) -> Callable[[F], F]: ...
def require_protenix(message: F | str | None = None) -> F | Callable[[F], F]:
    """Decorator to require Protenix model availability.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @require_protenix
    ... def train_protenix_model():
    ...     pass

    >>> @require_protenix("Custom error message")
    ... def custom_function():
    ...     pass
    """
    if callable(message):
        # Bare ``@require_protenix`` usage: the decorated function arrives as ``message``.
        # Re-dispatch so both ``@require_protenix`` and ``@require_protenix("msg")`` work.
        return require_protenix()(cast("F", message))

    default_message = (
        "Protenix model wrapper is not available. Install with: pixi install -e protenix"
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not PROTENIX_AVAILABLE:
                error_msg = message or default_message
                try:
                    import pytest

                    pytest.skip(error_msg)
                except ImportError:
                    raise ImportError(error_msg) from None
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


@overload
def require_rf3(message: F) -> F: ...
@overload
def require_rf3(message: str | None = None) -> Callable[[F], F]: ...
def require_rf3(message: F | str | None = None) -> F | Callable[[F], F]:
    """Decorator to require RF3 model availability.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @require_rf3
    ... def train_rf3_model():
    ...     pass

    >>> @require_rf3("Custom error message")
    ... def custom_function():
    ...     pass
    """
    if callable(message):
        # Bare ``@require_rf3`` usage: the decorated function arrives as ``message``.
        # Re-dispatch so both ``@require_rf3`` and ``@require_rf3("msg")`` work.
        return require_rf3()(cast("F", message))

    default_message = "RF3 model wrapper is not available. Install with: pixi install -e rf3"

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not RF3_AVAILABLE:
                error_msg = message or default_message
                try:
                    import pytest

                    pytest.skip(error_msg)
                except ImportError:
                    raise ImportError(error_msg) from None
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


@overload
def require_protpardelle(message: F) -> F: ...
@overload
def require_protpardelle(message: str | None = None) -> Callable[[F], F]: ...
def require_protpardelle(message: F | str | None = None) -> F | Callable[[F], F]:
    """Decorator to require Protpardelle model availability.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @require_protpardelle
    ... def sample_protpardelle():
    ...     pass

    >>> @require_protpardelle("Custom error message")
    ... def custom_function():
    ...     pass
    """
    if callable(message):
        # Bare ``@require_protpardelle`` usage: the decorated function arrives as ``message``.
        # Re-dispatch so both ``@require_protpardelle`` and ``@require_protpardelle("msg")`` work.
        return require_protpardelle()(cast("F", message))

    default_message = (
        "Protpardelle model wrapper is not available. Install with: pixi install -e protpardelle"
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not PROTPARDELLE_AVAILABLE:
                error_msg = message or default_message
                try:
                    import pytest

                    pytest.skip(error_msg)
                except ImportError:
                    raise ImportError(error_msg) from None
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


@overload
def require_any_model(message: F) -> F: ...
@overload
def require_any_model(message: str | None = None) -> Callable[[F], F]: ...
def require_any_model(message: F | str | None = None) -> F | Callable[[F], F]:
    """Decorator to require at least one model wrapper availability.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @require_any_model
    ... def train_any_model():
    ...     pass

    >>> @require_any_model("Need at least one model")
    ... def custom_function():
    ...     pass
    """
    if callable(message):
        # Bare ``@require_any_model`` usage: the decorated function arrives as ``message``.
        # Re-dispatch so both ``@require_any_model`` and ``@require_any_model("msg")`` work.
        return require_any_model()(cast("F", message))

    default_message = (
        "No model wrappers are available. "
        "Please install at least one model wrapper with the appropriate feature group: "
        "'pixi install -e boltz', 'pixi install -e protpardelle', or "
        "'pixi install -e protenix', or 'pixi install -e rf3'"
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if (
                not BOLTZ_AVAILABLE
                and not PROTENIX_AVAILABLE
                and not RF3_AVAILABLE
                and not PROTPARDELLE_AVAILABLE
            ):
                error_msg = message or default_message
                try:
                    import pytest

                    pytest.skip(error_msg)
                except ImportError:
                    raise ImportError(error_msg) from None
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def check_boltz_available(message: str | None = None) -> None:
    """Check if Boltz is available, raise ImportError if not.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Raises
    ------
    ImportError
        If Boltz model wrapper is not available.
    """
    if not BOLTZ_AVAILABLE:
        default_message = (
            "Boltz model wrapper is not available. Install with: pixi install -e boltz"
        )
        raise ImportError(message or default_message)


def check_protenix_available(message: str | None = None) -> None:
    """Check if Protenix is available, raise ImportError if not.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Raises
    ------
    ImportError
        If Protenix model wrapper is not available.
    """
    if not PROTENIX_AVAILABLE:
        default_message = (
            "Protenix model wrapper is not available. Install with: pixi install -e protenix"
        )
        raise ImportError(message or default_message)


def check_rf3_available(message: str | None = None) -> None:
    """Check if RF3 is available, raise ImportError if not.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Raises
    ------
    ImportError
        If RF3 model wrapper is not available.
    """
    if not RF3_AVAILABLE:
        default_message = "RF3 model wrapper is not available. Install with: pixi install -e rf3"
        raise ImportError(message or default_message)


def check_protpardelle_available(message: str | None = None) -> None:
    """Check if Protpardelle is available, raise ImportError if not.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Raises
    ------
    ImportError
        If Protpardelle model wrapper is not available.
    """
    if not PROTPARDELLE_AVAILABLE:
        default_message = (
            "Protpardelle model wrapper is not available. Install with: "
            "pixi install -e protpardelle"
        )
        raise ImportError(message or default_message)


def check_any_model_available(message: str | None = None) -> None:
    """Check if at least one model is available, raise ImportError if not.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Raises
    ------
    ImportError
        If no model wrapper is available.
    """
    if (
        not BOLTZ_AVAILABLE
        and not PROTENIX_AVAILABLE
        and not RF3_AVAILABLE
        and not PROTPARDELLE_AVAILABLE
    ):
        default_message = (
            "No model wrappers are available. "
            "Please install at least one model wrapper with the appropriate "
            "feature group: 'pixi install -e boltz', 'pixi install -e protenix', "
            "'pixi install -e rf3', or 'pixi install -e protpardelle'"
        )
        raise ImportError(message or default_message)
