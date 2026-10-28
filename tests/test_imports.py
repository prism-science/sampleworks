"""Tests for the centralized imports module."""

import pytest
from sampleworks.utils.imports import (
    BIOEMU2_AVAILABLE,
    BOLTZ_AVAILABLE,
    check_any_model_available,
    check_bioemu2_available,
    check_boltz_available,
    check_protenix_available,
    check_rf3_available,
    PROTENIX_AVAILABLE,
    PROTPARDELLE_AVAILABLE,
    require_any_model,
    require_boltz,
    require_protenix,
    require_rf3,
    RF3_AVAILABLE,
)


ANY_MODEL_AVAILABLE = (
    BOLTZ_AVAILABLE
    or PROTENIX_AVAILABLE
    or RF3_AVAILABLE
    or PROTPARDELLE_AVAILABLE
    or BIOEMU2_AVAILABLE
)


class TestAvailabilityFlags:
    """Test that availability flags are properly set."""

    def test_boltz_available_is_bool(self):
        """BOLTZ_AVAILABLE should be a boolean."""
        assert isinstance(BOLTZ_AVAILABLE, bool)

    def test_protenix_available_is_bool(self):
        """PROTENIX_AVAILABLE should be a boolean."""
        assert isinstance(PROTENIX_AVAILABLE, bool)

    def test_rf3_available_is_bool(self):
        """RF3_AVAILABLE should be a boolean."""
        assert isinstance(RF3_AVAILABLE, bool)

    def test_bioemu2_available_is_bool(self):
        """BIOEMU2_AVAILABLE should be a boolean."""
        assert isinstance(BIOEMU2_AVAILABLE, bool)


class TestCheckFunctions:
    """Test the check_*_available helper functions."""

    def test_check_any_model_available_no_error(self):
        """check_any_model_available should not raise if any model available."""
        if ANY_MODEL_AVAILABLE:
            check_any_model_available()

    def test_check_bioemu2_available(self):
        """check_bioemu2_available should match BIOEMU2_AVAILABLE flag."""
        if BIOEMU2_AVAILABLE:
            check_bioemu2_available()
        else:
            with pytest.raises(ImportError, match="BioEmu2 model wrapper"):
                check_bioemu2_available()

    def test_check_boltz_available(self):
        """check_boltz_available should match BOLTZ_AVAILABLE flag."""
        if BOLTZ_AVAILABLE:
            check_boltz_available()
        else:
            with pytest.raises(ImportError, match="Boltz model wrapper"):
                check_boltz_available()

    def test_check_protenix_available(self):
        """check_protenix_available should match PROTENIX_AVAILABLE flag."""
        if PROTENIX_AVAILABLE:
            check_protenix_available()
        else:
            with pytest.raises(ImportError, match="Protenix model wrapper"):
                check_protenix_available()

    def test_check_rf3_available(self):
        """check_rf3_available should match RF3_AVAILABLE flag."""
        if RF3_AVAILABLE:
            check_rf3_available()
        else:
            with pytest.raises(ImportError, match="RF3 model wrapper"):
                check_rf3_available()

    def test_check_functions_custom_message(self):
        """Check functions should support custom error messages."""
        custom_msg = "Custom error message for testing"

        if not BOLTZ_AVAILABLE:
            with pytest.raises(ImportError, match=custom_msg):
                check_boltz_available(custom_msg)

        if not PROTENIX_AVAILABLE:
            with pytest.raises(ImportError, match=custom_msg):
                check_protenix_available(custom_msg)

        if not RF3_AVAILABLE:
            with pytest.raises(ImportError, match=custom_msg):
                check_rf3_available(custom_msg)

        if not ANY_MODEL_AVAILABLE:
            with pytest.raises(ImportError, match=custom_msg):
                check_any_model_available(custom_msg)


class TestRequireDecorators:
    """Test the require_* decorators."""

    def test_require_boltz_decorator(self):
        """@require_boltz should skip if Boltz unavailable."""

        @require_boltz()
        def boltz_function():
            return "executed"

        if BOLTZ_AVAILABLE:
            assert boltz_function() == "executed"
        else:
            with pytest.raises(pytest.skip.Exception):
                boltz_function()

    def test_require_protenix_decorator(self):
        """@require_protenix should skip if Protenix unavailable."""

        @require_protenix()
        def protenix_function():
            return "executed"

        if PROTENIX_AVAILABLE:
            assert protenix_function() == "executed"
        else:
            with pytest.raises(pytest.skip.Exception):
                protenix_function()

    def test_require_rf3_decorator(self):
        """@require_rf3 should skip if RF3 unavailable."""

        @require_rf3()
        def rf3_function():
            return "executed"

        if RF3_AVAILABLE:
            assert rf3_function() == "executed"
        else:
            with pytest.raises(pytest.skip.Exception):
                rf3_function()

    def test_require_any_model_decorator(self):
        """@require_any_model should skip if no models available."""

        @require_any_model()
        def any_model_function():
            return "executed"

        if ANY_MODEL_AVAILABLE:
            assert any_model_function() == "executed"
        else:
            with pytest.raises(pytest.skip.Exception):
                any_model_function()

    def test_decorator_preserves_function_metadata(self):
        """Decorators should preserve function name and docstring."""

        @require_any_model()
        def example_function():
            """Example docstring."""
            pass

        assert example_function.__name__ == "example_function"
        assert example_function.__doc__ == "Example docstring."

    def test_decorator_with_arguments(self):
        """Decorators should work with functions that have arguments."""

        @require_any_model()
        def function_with_args(x: int, y: int) -> int:
            return x + y

        if ANY_MODEL_AVAILABLE:
            assert function_with_args(2, 3) == 5

    def test_decorator_custom_message(self):
        """Decorators should support custom error messages."""
        custom_msg = "Custom decorator error message"

        @require_boltz(custom_msg)
        def custom_boltz_function():
            return "executed"

        if not BOLTZ_AVAILABLE:
            with pytest.raises(pytest.skip.Exception, match=custom_msg):
                custom_boltz_function()

        @require_protenix(custom_msg)
        def custom_protenix_function():
            return "executed"

        if not PROTENIX_AVAILABLE:
            with pytest.raises(pytest.skip.Exception, match=custom_msg):
                custom_protenix_function()

        @require_rf3(custom_msg)
        def custom_rf3_function():
            return "executed"

        if not RF3_AVAILABLE:
            with pytest.raises(pytest.skip.Exception, match=custom_msg):
                custom_rf3_function()

        @require_any_model(custom_msg)
        def custom_any_model_function():
            return "executed"

        if not ANY_MODEL_AVAILABLE:
            with pytest.raises(pytest.skip.Exception, match=custom_msg):
                custom_any_model_function()


class TestReexports:
    """Test that re-exports from utils.__init__ work."""

    def test_reexports_from_utils_package(self):
        """All imports should be available from sampleworks.utils."""
        from sampleworks.utils import (
            BOLTZ_AVAILABLE as BOLTZ_REEXPORT,
            check_any_model_available as check_any_reexport,
            check_boltz_available as check_boltz_reexport,
            check_protenix_available as check_protenix_reexport,
            check_rf3_available as check_rf3_reexport,
            PROTENIX_AVAILABLE as PROTENIX_REEXPORT,
            require_any_model as require_any_reexport,
            require_boltz as require_boltz_reexport,
            require_protenix as require_protenix_reexport,
            require_rf3 as require_rf3_reexport,
            RF3_AVAILABLE as RF3_REEXPORT,
        )

        assert BOLTZ_REEXPORT == BOLTZ_AVAILABLE
        assert PROTENIX_REEXPORT == PROTENIX_AVAILABLE
        assert RF3_REEXPORT == RF3_AVAILABLE
        assert check_any_reexport is check_any_model_available
        assert check_boltz_reexport is check_boltz_available
        assert check_protenix_reexport is check_protenix_available
        assert check_rf3_reexport is check_rf3_available
        assert require_any_reexport is require_any_model
        assert require_boltz_reexport is require_boltz
        assert require_protenix_reexport is require_protenix
        assert require_rf3_reexport is require_rf3


class TestDecoratorIntegrationWithPytest:
    """Test decorator integration with pytest test functions."""

    @require_boltz()
    def test_boltz_specific_feature(self):
        """This test should be skipped if Boltz is not available."""
        assert BOLTZ_AVAILABLE

    @require_protenix()
    def test_protenix_specific_feature(self):
        """This test should be skipped if Protenix is not available."""
        assert PROTENIX_AVAILABLE

    @require_rf3()
    def test_rf3_specific_feature(self):
        """This test should be skipped if RF3 is not available."""
        assert RF3_AVAILABLE

    @require_any_model()
    def test_any_model_feature(self):
        """This test should be skipped if no models are available."""
        assert ANY_MODEL_AVAILABLE
