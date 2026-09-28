"""Tests for the reward registry and its option schemas."""

import dataclasses

import pytest
from sampleworks.core.rewards.options import (
    option_type,
    path_option_names,
    RealSpaceDensityOptions,
)
from sampleworks.core.rewards.registry import (
    build_single_reward,
    coerce_options,
    get_reward_spec,
    REWARD_SPECS,
    reward_type_names,
    RewardBuildContext,
)
from sampleworks.utils.guidance_constants import Rewards


@pytest.mark.parametrize("reward", list(REWARD_SPECS), ids=lambda r: r.value)
class TestRegistryEntries:
    """Contract every registered reward must satisfy."""

    def test_builder_is_importable(self, reward: Rewards):
        """A registry entry that cannot resolve its builder is a broken entry."""
        assert callable(REWARD_SPECS[reward].builder())

    def test_spec_is_keyed_by_its_own_name(self, reward: Rewards):
        assert REWARD_SPECS[reward].name is reward

    def test_declared_data_and_resolution_options_exist(self, reward: Rewards):
        """The grid-search injection points must name real options."""
        spec = REWARD_SPECS[reward]
        option_names = {f.name for f in dataclasses.fields(spec.options_cls)}

        for declared in (spec.data_path_option, spec.resolution_option):
            if declared is not None:
                assert declared in option_names

    def test_file_valued_options_are_declared_as_paths(self, reward: Rewards):
        """A path option that forgets ``path=True`` silently escapes path remapping."""
        spec = REWARD_SPECS[reward]
        looks_like_a_path = {
            f.name
            for f in dataclasses.fields(spec.options_cls)
            if f.name.endswith(("file", "path", "density"))
        }

        assert looks_like_a_path <= set(path_option_names(spec.options_cls))

    def test_options_are_constructible_with_defaults(self, reward: Rewards):
        assert coerce_options(REWARD_SPECS[reward], {}) == REWARD_SPECS[reward].options_cls()


def test_every_reward_enum_member_is_registered():
    """The enum and the registry must not drift apart."""
    assert set(REWARD_SPECS) == set(Rewards)
    assert reward_type_names() == [r.value for r in REWARD_SPECS]


def test_get_reward_spec_accepts_the_string_form():
    assert get_reward_spec("structure_factor") is REWARD_SPECS[Rewards.STRUCTURE_FACTOR]


def test_get_reward_spec_names_the_alternatives_for_an_unknown_reward():
    with pytest.raises(ValueError, match="Unknown reward type 'diffuse_scattering'"):
        get_reward_spec("diffuse_scattering")


def test_coerce_options_rejects_unknown_options_and_lists_the_valid_ones():
    with pytest.raises(ValueError, match=r"Unknown option\(s\) \['looss_order'\]"):
        coerce_options(get_reward_spec(Rewards.REAL_SPACE_DENSITY), {"looss_order": 1})


def test_coerce_options_materializes_defaults_for_absent_options():
    options = coerce_options(get_reward_spec(Rewards.REAL_SPACE_DENSITY), {"resolution": 1.8})

    assert options == RealSpaceDensityOptions(resolution=1.8, loss_order=2, em=False)


def test_option_type_strips_optionality():
    assert option_type(RealSpaceDensityOptions, "resolution") is float
    assert option_type(RealSpaceDensityOptions, "loss_order") is int
    assert option_type(RealSpaceDensityOptions, "em") is bool


def test_option_type_keeps_a_generic_whole():
    """Collapsing list[str] to str would silently drop nargs from its CLI flag."""

    @dataclasses.dataclass(frozen=True)
    class Options:
        required_columns: list[str] = dataclasses.field(default_factory=list)
        optional_columns: list[str] | None = None

    assert option_type(Options, "required_columns") == list[str]
    assert option_type(Options, "optional_columns") == list[str]


class TestBuilderValidation:
    """Missing required inputs are the reward's own error to raise."""

    def test_density_reward_requires_a_map(self):
        with pytest.raises(ValueError, match="needs a density map"):
            build_single_reward(
                Rewards.REAL_SPACE_DENSITY,
                {"resolution": 1.8},
                RewardBuildContext(structure={}),
            )

    def test_density_reward_requires_a_resolution(self):
        with pytest.raises(ValueError, match="needs a map resolution"):
            build_single_reward(
                Rewards.REAL_SPACE_DENSITY,
                {"density": "map.ccp4"},
                RewardBuildContext(structure={}),
            )

    def test_structure_factor_reward_requires_an_mtz(self):
        with pytest.raises(ValueError, match="needs a target MTZ"):
            build_single_reward(Rewards.STRUCTURE_FACTOR, {}, RewardBuildContext(structure={}))
