"""Tests for the loss registry and its option schemas."""

import dataclasses

import pytest
from sampleworks.core.losses.options import (
    option_type,
    path_option_names,
    RealSpaceDensityOptions,
)
from sampleworks.core.losses.registry import (
    build_single_loss,
    coerce_options,
    get_loss_spec,
    LOSS_SPECS,
    loss_type_names,
    LossBuildContext,
)
from sampleworks.utils.guidance_constants import Losses


@pytest.mark.parametrize("loss", list(LOSS_SPECS), ids=lambda r: r.value)
class TestRegistryEntries:
    """Contract every registered loss must satisfy."""

    def test_builder_is_importable(self, loss: Losses):
        """A registry entry that cannot resolve its builder is a broken entry."""
        assert callable(LOSS_SPECS[loss].builder())

    def test_spec_is_keyed_by_its_own_name(self, loss: Losses):
        assert LOSS_SPECS[loss].name is loss

    def test_declared_data_and_resolution_options_exist(self, loss: Losses):
        """The grid-search injection points must name real options."""
        spec = LOSS_SPECS[loss]
        option_names = {f.name for f in dataclasses.fields(spec.options_cls)}

        for declared in (spec.data_path_option, spec.resolution_option):
            if declared is not None:
                assert declared in option_names

    def test_file_valued_options_are_declared_as_paths(self, loss: Losses):
        """A path option that forgets ``path=True`` silently escapes path remapping."""
        spec = LOSS_SPECS[loss]
        looks_like_a_path = {
            f.name
            for f in dataclasses.fields(spec.options_cls)
            if f.name.endswith(("file", "path", "density"))
        }

        assert looks_like_a_path <= set(path_option_names(spec.options_cls))

    def test_options_are_constructible_with_defaults(self, loss: Losses):
        assert coerce_options(LOSS_SPECS[loss], {}) == LOSS_SPECS[loss].options_cls()


def test_every_loss_enum_member_is_registered():
    """The enum and the registry must not drift apart."""
    assert set(LOSS_SPECS) == set(Losses)
    assert loss_type_names() == [r.value for r in LOSS_SPECS]


def test_get_loss_spec_accepts_the_string_form():
    assert get_loss_spec("structure_factor") is LOSS_SPECS[Losses.STRUCTURE_FACTOR]


def test_get_loss_spec_names_the_alternatives_for_an_unknown_loss():
    with pytest.raises(ValueError, match="Unknown loss type 'diffuse_scattering'"):
        get_loss_spec("diffuse_scattering")


def test_coerce_options_rejects_unknown_options_and_lists_the_valid_ones():
    with pytest.raises(ValueError, match=r"Unknown option\(s\) \['looss_order'\]"):
        coerce_options(get_loss_spec(Losses.REAL_SPACE_DENSITY), {"looss_order": 1})


def test_coerce_options_materializes_defaults_for_absent_options():
    options = coerce_options(get_loss_spec(Losses.REAL_SPACE_DENSITY), {"resolution": 1.8})

    assert options == RealSpaceDensityOptions(resolution=1.8, loss_order=2, em=False)


def test_option_type_strips_optionality():
    assert option_type(RealSpaceDensityOptions, "resolution") is float
    assert option_type(RealSpaceDensityOptions, "loss_order") is int
    assert option_type(RealSpaceDensityOptions, "em") is bool
    assert option_type(RealSpaceDensityOptions, "scattering_table") is str


def test_option_type_keeps_a_generic_whole():
    """Collapsing list[str] to str would silently drop nargs from its CLI flag."""

    @dataclasses.dataclass(frozen=True)
    class Options:
        required_columns: list[str] = dataclasses.field(default_factory=list)
        optional_columns: list[str] | None = None

    assert option_type(Options, "required_columns") == list[str]
    assert option_type(Options, "optional_columns") == list[str]


class TestBuilderValidation:
    """Missing required inputs are the loss's own error to raise."""

    def test_density_loss_requires_a_map(self):
        with pytest.raises(ValueError, match="needs a density map"):
            build_single_loss(
                Losses.REAL_SPACE_DENSITY,
                {"resolution": 1.8},
                LossBuildContext(structure={}),
            )

    def test_density_loss_requires_a_resolution(self):
        with pytest.raises(ValueError, match="needs a map resolution"):
            build_single_loss(
                Losses.REAL_SPACE_DENSITY,
                {"density": "map.ccp4"},
                LossBuildContext(structure={}),
            )

    def test_structure_factor_loss_requires_an_mtz(self):
        with pytest.raises(ValueError, match="needs a target MTZ"):
            build_single_loss(Losses.STRUCTURE_FACTOR, {}, LossBuildContext(structure={}))
