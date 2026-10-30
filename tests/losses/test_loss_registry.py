"""Tests for the loss registry and its option schemas."""

import dataclasses

import pytest
from sampleworks.core.losses.options import (
    option_type,
    path_option_names,
    RealSpaceDensityOptions,
)
from sampleworks.core.losses.registry import (
    bool_from_text,
    build_single_loss,
    coerce_options,
    get_loss_spec,
    is_path_option,
    loss_option_names,
    LOSS_SPECS,
    loss_type_names,
    LossBuildContext,
    option_from_text,
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

    def test_shared_option_names_share_a_declaration(self, loss: Losses):
        """A proteins-CSV column is parsed by option name alone, for every loss that has it."""
        for option in dataclasses.fields(LOSS_SPECS[loss].options_cls):
            for other in LOSS_SPECS.values():
                for other_option in dataclasses.fields(other.options_cls):
                    if other_option.name != option.name:
                        continue
                    assert option_type(other.options_cls, option.name) == option_type(
                        LOSS_SPECS[loss].options_cls, option.name
                    )
                    assert other_option.metadata["path"] == option.metadata["path"]
                    assert other_option.metadata["json_arg"] == option.metadata["json_arg"]

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


class TestOptionsFromText:
    """Proteins-CSV cells are parsed to each option's declared type."""

    @pytest.mark.parametrize(
        ("name", "text", "expected"),
        [
            ("resolution", "1.8", 1.8),
            ("loss_order", "1", 1),
            ("em", "true", True),
            ("normalize_amplitude", "No", False),
            ("expcolumns", "FP SIGFP", ["FP", "SIGFP"]),
            ("sfcalculator_kwargs", '{"n_bins": 15}', {"n_bins": 15}),
            ("bulk_solvent", "combined", "combined"),
        ],
    )
    def test_parses_to_the_declared_type(self, name, text, expected):
        assert option_from_text(name, text) == expected

    @pytest.mark.parametrize(
        "text", ["true", "True", "TRUE", "t", "T", "yes", "Yes", "y", "on", "ON", "1", " true "]
    )
    def test_truthy_text_is_true(self, text):
        assert bool_from_text(text) is True

    @pytest.mark.parametrize(
        "text", ["false", "False", "FALSE", "f", "F", "no", "No", "n", "off", "OFF", "0", " no "]
    )
    def test_falsy_text_is_false(self, text):
        assert bool_from_text(text) is False

    @pytest.mark.parametrize("text", ["maybe", "", "2", "-1", "truee", "none"])
    def test_other_text_is_not_a_boolean(self, text):
        with pytest.raises(ValueError, match="Expected true or false"):
            bool_from_text(text)

    def test_rejects_a_value_that_is_not_a_boolean(self):
        with pytest.raises(ValueError, match="Expected true or false"):
            option_from_text("em", "maybe")

    def test_rejects_an_option_no_loss_declares(self):
        with pytest.raises(ValueError, match="No registered loss has an option named 'mtz_file'"):
            option_from_text("mtz_file", "x.mtz")

    def test_knows_which_options_are_paths(self):
        assert is_path_option("mtzfile")
        assert is_path_option("diffuse_target")
        assert not is_path_option("resolution")

    def test_option_names_span_every_loss(self):
        names = loss_option_names()

        assert {"density", "mtzfile", "bragg_target", "diffuse_target", "resolution"} <= set(names)
        assert len(names) == len(set(names))
