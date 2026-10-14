"""Tests for the unified guidance CLI and GuidanceConfig.from_cli()."""

from __future__ import annotations

import subprocess
import sys

import pytest
from sampleworks.utils.guidance_script_arguments import GuidanceConfig


COMMON_ARGS = [
    "--protein",
    "1VME",
    "--structure",
    "test.cif",
    "--density",
    "test.ccp4",
    "--resolution",
    "1.8",
    "--output-dir",
    "output",
]


class TestFromCliUnified:
    """Test from_cli() when model and guidance_type come from CLI args."""

    def test_boltz2_pure_guidance(self):
        argv = ["--model", "boltz2", "--guidance-type", "pure_guidance"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.model_name == "boltz2"
        assert config.guidance_type == "pure_guidance"
        assert config.protein == "1VME"
        assert config.structure == "test.cif"
        assert config.density == "test.ccp4"
        assert config.resolution == 1.8

    def test_rf3_fk_steering(self):
        argv = ["--model", "rf3", "--guidance-type", "fk_steering"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.model_name == "rf3"
        assert config.guidance_type == "fk_steering"
        assert hasattr(config, "num_particles")
        assert hasattr(config, "fk_lambda")

    def test_model_specific_args_boltz2_method(self):
        argv = [
            "--model",
            "boltz2",
            "--guidance-type",
            "pure_guidance",
            "--method",
            "MD",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.method == "MD"

    def test_model_specific_args_rf3_msa(self):
        argv = [
            "--model",
            "rf3",
            "--guidance-type",
            "pure_guidance",
            "--msa-path",
            "/data/msa.a3m",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert getattr(config, "msa_path") == "/data/msa.a3m"

    def test_model_specific_args_protpardelle_sampling(self):
        argv = [
            "--model",
            "protpardelle",
            "--guidance-type",
            "pure_guidance",
            "--model-checkpoint",
            "/data/cc89.pth",
            "--protpardelle-config-path",
            "/data/cc89.yaml",
            "--num-diffusion-steps",
            "64",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert getattr(config, "model_checkpoint") == "/data/cc89.pth"
        assert getattr(config, "protpardelle_config_path") == "/data/cc89.yaml"
        assert config.num_diffusion_steps == 64

    def test_guidance_specific_args_fk(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "fk_steering",
            "--num-particles",
            "5",
            "--fk-lambda",
            "2.0",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.num_particles == 5
        assert config.fk_lambda == 2.0

    def test_guidance_specific_args_pure(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--step-size",
            "0.5",
            "--step-scaler-type",
            "dataspace",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.step_size == 0.5
        assert config.step_scaler_type == "dataspace"


class TestFromCliLegacyScripts:
    """Test from_cli() when model/guidance_type are pre-set (legacy script pattern)."""

    def test_preset_model_and_guidance_type(self):
        config = GuidanceConfig.from_cli(
            COMMON_ARGS,
            model_name="boltz2",
            guidance_type="pure_guidance",
        )
        assert config.model_name == "boltz2"
        assert config.guidance_type == "pure_guidance"
        assert config.protein == "1VME"

    def test_no_model_guidance_type_on_cli_required(self):
        """Legacy scripts should not require --model/--guidance-type on CLI."""
        config = GuidanceConfig.from_cli(
            COMMON_ARGS,
            model_name="protenix",
            guidance_type="fk_steering",
        )
        assert config.model_name == "protenix"
        assert config.guidance_type == "fk_steering"

    @pytest.mark.parametrize("model", ["boltz1", "boltz2", "protenix", "rf3"])
    @pytest.mark.parametrize("guidance_type", ["pure_guidance", "fk_steering"])
    def test_all_model_guidance_combos(self, model, guidance_type):
        config = GuidanceConfig.from_cli(
            COMMON_ARGS,
            model_name=model,
            guidance_type=guidance_type,
        )
        assert config.model_name == model
        assert config.guidance_type == guidance_type


class TestFromCliValidation:
    """Test error handling for invalid inputs."""

    def test_missing_protein_errors(self):
        argv = [
            "--model",
            "boltz2",
            "--guidance-type",
            "pure_guidance",
            "--structure",
            "test.cif",
            "--density",
            "test.ccp4",
            "--resolution",
            "1.8",
        ]
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_invalid_model_errors(self):
        argv = ["--model", "invalid_model", "--guidance-type", "pure_guidance"] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_invalid_guidance_type_errors(self):
        argv = ["--model", "boltz2", "--guidance-type", "invalid_type"] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_missing_required_structure_errors(self):
        argv = [
            "--model",
            "boltz2",
            "--guidance-type",
            "pure_guidance",
            "--protein",
            "1VME",
            "--density",
            "test.ccp4",
            "--resolution",
            "1.8",
        ]
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)


class TestFromCliDefaults:
    """Test that defaults are applied correctly."""

    def test_generic_defaults(self):
        argv = ["--model", "boltz1", "--guidance-type", "pure_guidance"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.output_dir == "output"
        assert config.partial_diffusion_step == 0
        assert config.loss_order == 2
        assert config.guidance_start == -1
        assert config.augmentation is False
        assert config.align_to_input is False
        assert config.ensemble_size == 4

    def test_boolean_flags(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--augmentation",
            "--align-to-input",
            "--gradient-normalization",
            "--em",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.augmentation is True
        assert config.align_to_input is True
        assert config.gradient_normalization is True
        assert config.em is True


class TestCrossModelArgRejection:
    """Test that model-specific args are rejected for the wrong model."""

    def test_method_rejected_for_protenix(self):
        argv = [
            "--model",
            "protenix",
            "--guidance-type",
            "pure_guidance",
            "--method",
            "MD",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_method_rejected_for_rf3(self):
        argv = [
            "--model",
            "rf3",
            "--guidance-type",
            "pure_guidance",
            "--method",
            "MD",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_msa_path_rejected_for_boltz1(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--msa-path",
            "/data/msa.a3m",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_conflicting_model_in_preset_mode(self):
        argv = ["--model", "boltz2"] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv, model_name="rf3", guidance_type="fk_steering")

    def test_conflicting_guidance_type_in_preset_mode(self):
        argv = ["--guidance-type", "fk_steering"] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv, model_name="boltz1", guidance_type="pure_guidance")

    def test_fk_args_rejected_for_pure_guidance(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--num-particles",
            "5",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_pure_args_rejected_for_fk_steering(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "fk_steering",
            "--step-scaler-type",
            "dataspace",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_msa_path_rejected_for_boltz2(self):
        argv = [
            "--model",
            "boltz2",
            "--guidance-type",
            "pure_guidance",
            "--msa-path",
            "/data/msa.a3m",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_chiral_flags_rejected_for_boltz1(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--disable-chiral-features",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)


class TestPresetMatchingAccepted:
    """Matching preset values should not trigger false-positive rejection."""

    def test_matching_model_accepted(self):
        argv = ["--model", "boltz1"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv, model_name="boltz1", guidance_type="fk_steering")
        assert config.model_name == "boltz1"

    def test_matching_guidance_type_accepted(self):
        argv = ["--guidance-type", "pure_guidance"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv, model_name="boltz2", guidance_type="pure_guidance")
        assert config.guidance_type == "pure_guidance"


class TestArgPassthrough:
    """Test that non-default argument values propagate to GuidanceConfig."""

    def test_model_checkpoint(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--model-checkpoint",
            "/custom/path.ckpt",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.model_checkpoint == "/custom/path.ckpt"

    def test_device(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--device",
            "cpu",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.device == "cpu"

    def test_output_dir_override(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--output-dir",
            "/custom/output",
            "--protein",
            "1VME",
            "--structure",
            "test.cif",
            "--density",
            "test.ccp4",
            "--resolution",
            "1.8",
        ]
        config = GuidanceConfig.from_cli(argv)
        assert config.output_dir == "/custom/output"

    def test_partial_diffusion_step(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--partial-diffusion-step",
            "50",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.partial_diffusion_step == 50

    def test_log_path(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--log-path",
            "/tmp/run.log",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.log_path == "/tmp/run.log"

    def test_ensemble_size(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "fk_steering",
            "--ensemble-size",
            "8",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.ensemble_size == 8

    def test_rf3_chiral_features(self):
        argv = [
            "--model",
            "rf3",
            "--guidance-type",
            "pure_guidance",
            "--disable-chiral-features",
            "--track-chiral-features",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.disable_chiral_features is True
        assert config.track_chiral_features is True

    def test_loss_order(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--loss-order",
            "1",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.loss_order == 1

    def test_guidance_start(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--guidance-start",
            "10",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.guidance_start == 10

    def test_recycling_steps(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--recycling-steps",
            "3",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.recycling_steps == 3

    def test_num_diffusion_steps(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--num-diffusion-steps",
            "100",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.num_diffusion_steps == 100

    def test_recycling_steps_default_none(self):
        argv = ["--model", "boltz1", "--guidance-type", "pure_guidance"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.recycling_steps is None

    def test_num_diffusion_steps_default(self):
        argv = ["--model", "boltz1", "--guidance-type", "pure_guidance"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.num_diffusion_steps == 200


class TestValidationEdgeCases:
    """Additional validation edge cases."""

    def test_no_args_at_all(self):
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli([])

    def test_missing_resolution(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--protein",
            "1VME",
            "--structure",
            "test.cif",
            "--density",
            "test.ccp4",
        ]
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_missing_density(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--protein",
            "1VME",
            "--structure",
            "test.cif",
            "--resolution",
            "1.8",
        ]
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_invalid_loss_order(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--loss-order",
            "3",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_invalid_step_scaler_type(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--step-scaler-type",
            "invalid",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)


class TestEntrypointSmoke:
    """Smoke test the actual CLI entrypoint."""

    def test_help_exits_zero(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sampleworks.cli.guidance",
                "--model",
                "boltz1",
                "--guidance-type",
                "pure_guidance",
                "--help",
            ],
            capture_output=True,
        )
        assert result.returncode == 0
        assert b"--protein" in result.stdout
        assert b"--structure" in result.stdout

    def test_invalid_preset_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            GuidanceConfig.from_cli(COMMON_ARGS, model_name="typo", guidance_type="fk_steering")

    def test_invalid_preset_guidance_type_raises(self):
        with pytest.raises(ValueError, match="Unknown guidance type"):
            GuidanceConfig.from_cli(COMMON_ARGS, model_name="boltz1", guidance_type="typo")


class TestAlignmentReverseDiffusion:
    """--alignment-reverse-diffusion can be none, yes, or no and propagates into the config."""

    BASE = ["--model", "boltz2", "--guidance-type", "pure_guidance"] + COMMON_ARGS

    def test_omitted_defaults_to_none(self):
        config = GuidanceConfig.from_cli(self.BASE)
        assert config.alignment_reverse_diffusion is None

    def test_enable_flag_sets_true(self):
        config = GuidanceConfig.from_cli(self.BASE + ["--alignment-reverse-diffusion"])
        assert config.alignment_reverse_diffusion is True

    def test_disable_flag_sets_false(self):
        """--no-alignment-reverse-diffusion must be able to force the feature off."""
        config = GuidanceConfig.from_cli(self.BASE + ["--no-alignment-reverse-diffusion"])
        assert config.alignment_reverse_diffusion is False


class TestRewardSelection:
    """--reward-type picks a reward and brings that reward's options with it."""

    MODEL_ARGS = ["--model", "boltz2", "--guidance-type", "pure_guidance"]
    STRUCTURE_ARGS = ["--protein", "1VME", "--structure", "test.cif"]

    def test_density_is_the_default_reward(self):
        """Every command line written before rewards were selectable still means density."""
        config = GuidanceConfig.from_cli(self.MODEL_ARGS + COMMON_ARGS)

        assert list(config.reward_config) == ["real_space_density"]
        # Defaults are written out, so the run records what it actually used.
        assert config.reward_config["real_space_density"]["reward_options"] == {
            "density": "test.ccp4",
            "resolution": 1.8,
            "loss_order": 2,
            "em": False,
        }

    def test_density_options_still_reach_the_flat_config_fields(self):
        """Grid search and the evaluation scripts read these; they must keep working."""
        config = GuidanceConfig.from_cli(
            self.MODEL_ARGS + COMMON_ARGS + ["--loss-order", "1", "--em"]
        )

        assert (config.density, config.resolution, config.loss_order, config.em) == (
            "test.ccp4",
            1.8,
            1,
            True,
        )

    def test_structure_factor_reward_takes_its_own_options(self):
        config = GuidanceConfig.from_cli(
            self.MODEL_ARGS
            + self.STRUCTURE_ARGS
            + [
                "--reward-type",
                "structure_factor",
                "--mtzfile",
                "1vme.mtz",
                "--bulk-solvent",
                "combined",
                "--expcolumns",
                "FP",
                "SIGFP",
                "--normalize-amplitude",
            ]
        )

        assert list(config.reward_config) == ["structure_factor"]
        options = config.reward_config["structure_factor"]["reward_options"]
        assert options["mtzfile"] == "1vme.mtz"
        assert options["expcolumns"] == ["FP", "SIGFP"]
        assert options["bulk_solvent"] == "combined"
        assert options["normalize_amplitude"] is True
        assert options["batch_partition"] == 10  # defaulted, and recorded as such

    def test_options_of_another_reward_are_rejected(self):
        """--density means nothing to the structure-factor reward, so it must not be accepted."""
        argv = (
            self.MODEL_ARGS
            + self.STRUCTURE_ARGS
            + [
                "--reward-type",
                "structure_factor",
                "--mtzfile",
                "1vme.mtz",
                "--density",
                "test.ccp4",
            ]
        )

        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_an_unknown_reward_type_is_rejected(self):
        argv = self.MODEL_ARGS + COMMON_ARGS + ["--reward-type", "diffuse_scattering"]

        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_a_reward_missing_a_required_input_fails_before_anything_is_loaded(self):
        argv = self.MODEL_ARGS + self.STRUCTURE_ARGS + ["--reward-type", "structure_factor"]

        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_help_lists_the_selected_reward_s_options(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sampleworks.cli.guidance",
                "--model",
                "boltz1",
                "--guidance-type",
                "pure_guidance",
                "--reward-type",
                "structure_factor",
                "--help",
            ],
            capture_output=True,
        )

        assert result.returncode == 0
        assert b"--mtzfile" in result.stdout
        assert b"--density" not in result.stdout


class TestRewardConfigFile:
    """--reward-config is the same configuration, from a file, and the only way to compose."""

    BASE = [
        "--model",
        "boltz2",
        "--guidance-type",
        "pure_guidance",
        "--protein",
        "1VME",
        "--structure",
        "test.cif",
    ]

    def write_config(self, tmp_path, text: str) -> str:
        config_file = tmp_path / "rewards.yaml"
        config_file.write_text(text)
        return str(config_file)

    def test_file_and_flags_describe_the_same_run(self, tmp_path):
        from_file = GuidanceConfig.from_cli(
            self.BASE
            + [
                "--reward-config",
                self.write_config(
                    tmp_path,
                    "real_space_density:\n"
                    "  reward_options:\n"
                    "    density: test.ccp4\n"
                    "    resolution: 1.8\n"
                    "    loss_order: 1\n",
                ),
            ]
        )
        from_flags = GuidanceConfig.from_cli(
            self.BASE + ["--density", "test.ccp4", "--resolution", "1.8", "--loss-order", "1"]
        )

        assert from_file.reward_config == from_flags.reward_config

    def test_several_weighted_rewards_can_be_configured(self, tmp_path):
        config = GuidanceConfig.from_cli(
            self.BASE
            + [
                "--reward-config",
                self.write_config(
                    tmp_path,
                    "real_space_density:\n"
                    "  weight: 0.4\n"
                    "  reward_options: {density: test.ccp4, resolution: 1.8}\n"
                    "structure_factor:\n"
                    "  weight: 0.6\n"
                    "  reward_options: {mtzfile: test.mtz}\n",
                ),
            ]
        )

        assert config.resolved_reward_config().resolved_weights() == (0.4, 0.6)
        assert config.density == "test.ccp4"  # the density term still mirrors out

    def test_a_config_file_and_a_reward_type_together_are_rejected(self, tmp_path):
        argv = self.BASE + [
            "--reward-config",
            self.write_config(tmp_path, "real_space_density: {}\n"),
            "--reward-type",
            "real_space_density",
        ]

        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_per_reward_flags_are_not_accepted_alongside_a_config_file(self, tmp_path):
        argv = self.BASE + [
            "--reward-config",
            self.write_config(tmp_path, "real_space_density: {}\n"),
            "--loss-order",
            "1",
        ]

        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_a_broken_config_file_is_a_usage_error(self, tmp_path):
        argv = self.BASE + [
            "--reward-config",
            self.write_config(tmp_path, "real_space_density:\n  reward_options: {looss: 1}\n"),
        ]

        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)
