"""Tests for loss configuration parsing (issue #358)."""

import json
import pickle

import pytest
from sampleworks.core.losses.config import LossConfig, LossEntry
from sampleworks.utils.guidance_constants import Losses


ISSUE_358_YAML = """
real_space_density:
  weight: 0.4
  loss_options:
    loss_order: 1
structure_factor:
  weight: 0.6
  loss_options:
    mtzfile: /data/1vme.mtz
    bulk_solvent: combined
"""


class TestParsing:
    """The configuration shape from issue #358, in each supported format."""

    def test_parses_the_documented_yaml_shape(self, tmp_path):
        config_file = tmp_path / "losses.yaml"
        config_file.write_text(ISSUE_358_YAML)

        config = LossConfig.from_file(config_file)

        assert config.entries == (
            LossEntry(Losses.REAL_SPACE_DENSITY, 0.4, {"loss_order": 1}),
            LossEntry(
                Losses.STRUCTURE_FACTOR,
                0.6,
                {"mtzfile": "/data/1vme.mtz", "bulk_solvent": "combined"},
            ),
        )

    @pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml", ".toml"])
    def test_formats_agree(self, tmp_path, suffix):
        """The same configuration parses identically from JSON, YAML, and TOML."""
        mapping = {
            "structure_factor": {
                "weight": 1.0,
                "loss_options": {"mtzfile": "/data/x.mtz", "batch_partition": 4},
            }
        }
        config_file = tmp_path / f"losses{suffix}"
        if suffix == ".toml":
            config_file.write_text(
                "[structure_factor]\nweight = 1.0\n"
                '[structure_factor.loss_options]\nmtzfile = "/data/x.mtz"\nbatch_partition = 4\n'
            )
        else:
            config_file.write_text(json.dumps(mapping))  # valid YAML too

        assert LossConfig.from_file(config_file) == LossConfig.from_mapping(mapping)

    def test_yaml_resolves_environment_interpolation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SW_TEST_DATA_DIR", "/mnt/data")
        config_file = tmp_path / "losses.yaml"
        config_file.write_text(
            "structure_factor:\n  loss_options:\n    mtzfile: ${oc.env:SW_TEST_DATA_DIR}/1vme.mtz\n"
        )

        config = LossConfig.from_file(config_file)

        assert config.entries[0].options["mtzfile"] == "/mnt/data/1vme.mtz"

    def test_an_entry_may_be_empty_meaning_all_defaults(self):
        config = LossConfig.from_mapping({"real_space_density": None})

        assert config.entries == (LossEntry(Losses.REAL_SPACE_DENSITY),)

    def test_cli_and_file_forms_produce_the_same_configuration(self):
        from_flags = LossConfig.single(Losses.REAL_SPACE_DENSITY, density="m.ccp4", loss_order=1)
        from_file = LossConfig.from_mapping(
            {"real_space_density": {"loss_options": {"density": "m.ccp4", "loss_order": 1}}}
        )

        assert from_flags == from_file

    def test_mapping_round_trips(self):
        mapping = {
            "real_space_density": {"weight": 0.4, "loss_options": {"loss_order": 1}},
            "structure_factor": {"weight": 0.6, "loss_options": {"mtzfile": "/data/x.mtz"}},
        }

        assert LossConfig.from_mapping(mapping).to_mapping() == mapping

    def test_mapping_survives_json_and_pickle(self):
        """Run configurations are JSON-serialized into metadata and pickled into job queues."""
        mapping = LossConfig.single(Losses.STRUCTURE_FACTOR, mtzfile="/data/x.mtz").to_mapping()

        assert json.loads(json.dumps(mapping)) == mapping
        assert pickle.loads(pickle.dumps(mapping)) == mapping


class TestValidation:
    """Bad configurations fail at parse time, naming what to fix."""

    def test_unknown_loss_name_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="Unknown loss type 'densty'"):
            LossConfig.from_mapping({"densty": {}})

    def test_unknown_option_is_rejected(self):
        with pytest.raises(ValueError, match=r"Unknown option\(s\) \['mtz_file'\]"):
            LossConfig.from_mapping(
                {"structure_factor": {"loss_options": {"mtz_file": "/data/x.mtz"}}}
            )

    def test_loss_options_must_be_a_mapping(self):
        with pytest.raises(ValueError, match="must be a mapping of option name to value"):
            LossConfig.from_mapping({"real_space_density": {"loss_options": ["density", "x.ccp4"]}})

    def test_options_outside_loss_options_are_rejected(self):
        """A flat entry is the most likely mistake; say where the options go."""
        with pytest.raises(ValueError, match="Loss options belong under 'loss_options'"):
            LossConfig.from_mapping({"real_space_density": {"loss_order": 1}})

    def test_empty_configuration_is_rejected(self):
        with pytest.raises(ValueError, match="needs at least one loss"):
            LossConfig(())

    @pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
    def test_negative_or_non_finite_weight_is_rejected(self, weight):
        with pytest.raises(ValueError, match="must be non-negative and finite"):
            LossConfig((LossEntry(Losses.REAL_SPACE_DENSITY, weight),))

    @pytest.mark.parametrize("bragg_weight", [-0.1, 1.5, float("nan"), "0.5"])
    def test_bragg_weight_outside_the_unit_interval_is_rejected(self, bragg_weight):
        with pytest.raises(ValueError, match=r"bragg_weight must lie in \[0, 1\]"):
            LossConfig.single(Losses.DIFFUSE_BRAGG, bragg_weight=bragg_weight)

    @pytest.mark.parametrize("batch_partition", [0, -2, 2.5, True])
    def test_non_positive_integer_batch_partition_is_rejected(self, batch_partition):
        with pytest.raises(ValueError, match="batch_partition must be a positive integer"):
            LossConfig.single(Losses.STRUCTURE_FACTOR, batch_partition=batch_partition)

    def test_unsupported_file_format_is_rejected(self, tmp_path):
        config_file = tmp_path / "losses.ini"
        config_file.write_text("[real_space_density]\n")

        with pytest.raises(ValueError, match="Unsupported loss configuration format"):
            LossConfig.from_file(config_file)

    def test_missing_file_is_reported_as_such(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Loss configuration file not found"):
            LossConfig.from_file(tmp_path / "absent.yaml")


class TestWeights:
    """Weight resolution is where composition semantics live."""

    def test_omitted_weights_are_uniform(self):
        config = LossConfig.from_mapping({"real_space_density": {}, "structure_factor": {}})

        assert config.resolved_weights() == (0.5, 0.5)

    def test_a_single_loss_is_unweighted(self):
        assert LossConfig.single(Losses.REAL_SPACE_DENSITY).resolved_weights() == (1.0,)

    def test_given_weights_are_used_as_given(self):
        config = LossConfig.from_mapping(
            {"real_space_density": {"weight": 2.0}, "structure_factor": {"weight": 3.0}}
        )

        assert config.resolved_weights() == (2.0, 3.0)

    def test_partially_specified_weights_are_rejected(self):
        config = LossConfig.from_mapping(
            {"real_space_density": {"weight": 0.4}, "structure_factor": {}}
        )

        with pytest.raises(ValueError, match=r"\['structure_factor'\] have no weight"):
            config.resolved_weights()

    def test_the_same_loss_cannot_be_configured_twice(self):
        with pytest.raises(ValueError, match="configured more than once"):
            LossConfig(
                (
                    LossEntry(Losses.REAL_SPACE_DENSITY),
                    LossEntry(Losses.REAL_SPACE_DENSITY, options={"loss_order": 1}),
                )
            )


class TestMissingInputFiles:
    """Path options naming files that do not exist are reported before any work."""

    def test_reports_each_path_that_does_not_exist(self, tmp_path):
        present = tmp_path / "bragg.mtz"
        present.touch()
        absent = tmp_path / "diffuse.mtz"
        config = LossConfig.from_mapping(
            {
                "diffuse_bragg": {
                    "loss_options": {"bragg_target": str(present), "diffuse_target": str(absent)}
                },
                "real_space_density": {"loss_options": {"resolution": 1.8}},
            }
        )

        assert config.missing_input_files() == {"diffuse_bragg": {"diffuse_target": str(absent)}}

    def test_existing_and_unset_paths_are_not_reported(self, tmp_path):
        density = tmp_path / "map.ccp4"
        density.touch()
        config = LossConfig.from_mapping(
            {
                "real_space_density": {"loss_options": {"density": str(density)}},
                "structure_factor": {},
            }
        )

        assert config.missing_input_files() == {}


class TestPerProteinOptions:
    """Grid search fills each protein's CSV columns in after the loss is chosen."""

    PROTEIN_ROW = {"density": "/data/1vme.ccp4", "mtzfile": "/data/1vme.mtz", "resolution": 1.8}

    def test_each_value_lands_only_in_losses_that_declare_it(self):
        config = LossConfig.from_mapping(
            {"real_space_density": {}, "structure_factor": {}}
        ).with_unset_options_from(self.PROTEIN_ROW)

        by_loss = {entry.loss: entry.options for entry in config.entries}
        assert by_loss[Losses.REAL_SPACE_DENSITY] == {
            "density": "/data/1vme.ccp4",
            "resolution": 1.8,
        }
        assert by_loss[Losses.STRUCTURE_FACTOR] == {
            "mtzfile": "/data/1vme.mtz",
            "resolution": 1.8,
        }

    def test_explicitly_configured_values_win(self):
        config = LossConfig.single(
            Losses.STRUCTURE_FACTOR, mtzfile="/explicit.mtz"
        ).with_unset_options_from(self.PROTEIN_ROW)

        assert config.entries[0].options == {"mtzfile": "/explicit.mtz", "resolution": 1.8}

    def test_filled_options_satisfy_required_option_checks(self):
        config = LossConfig.single(Losses.STRUCTURE_FACTOR)

        assert config.missing_required_options() == {"structure_factor": ("mtzfile",)}
        assert config.with_unset_options_from(self.PROTEIN_ROW).missing_required_options() == {}

    @pytest.mark.parametrize(
        ("bragg_weight", "missing"),
        [
            (0.5, ("bragg_target", "diffuse_target")),
            (1.0, ("bragg_target",)),
            (0.0, ("diffuse_target",)),
        ],
    )
    def test_diffuse_bragg_requires_the_targets_its_weight_scores(self, bragg_weight, missing):
        config = LossConfig.single(Losses.DIFFUSE_BRAGG, bragg_weight=bragg_weight)

        assert config.missing_required_options() == {"diffuse_bragg": missing}

    def test_diffuse_bragg_with_its_weighted_target_is_complete(self):
        config = LossConfig.single(
            Losses.DIFFUSE_BRAGG, bragg_weight=1.0, bragg_target="/data/bragg.mtz"
        )

        assert config.missing_required_options() == {}

    @pytest.mark.parametrize("loss", list(Losses))
    @pytest.mark.parametrize("resolution", [0.0, -1.0, float("nan"), float("inf"), "1.8", True])
    def test_each_loss_rejects_an_invalid_resolution(self, loss, resolution):
        with pytest.raises(ValueError, match="resolution must be a positive, finite number"):
            LossConfig.single(loss).with_unset_options_from({"resolution": resolution})

    def test_paths_are_remapped_for_run_metadata(self):
        config = LossConfig.single(Losses.STRUCTURE_FACTOR, mtzfile="/data/x.mtz", resolution=2.0)

        mapping = config.remapped_paths(lambda p: p.replace("/data", "/host"))

        options = mapping["structure_factor"]["loss_options"]
        assert options["mtzfile"] == "/host/x.mtz"
        assert options["resolution"] == 2.0
