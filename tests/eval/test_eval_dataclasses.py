"""Tests for Trial and TrialList, focused on how they are used in eval scripts."""

import math
from pathlib import Path

import pytest
from sampleworks.eval.eval_dataclasses import ProteinConfig, Trial, TrialList


@pytest.fixture
def trial(tmp_path: Path) -> Trial:
    """Minimal valid Trial, mirroring what scan_grid_search_results produces."""
    return Trial(
        protein="1vme",
        altloc_occupancies={"A": 1},
        model="boltz2",
        method="MD",
        scaler="pure_guidance",
        ensemble_size=10,
        guidance_weight=1.0,
        gd_steps=None,
        trial_dir=tmp_path / "trial_dir",
        refined_cif_path=tmp_path / "trial_dir" / "refined.cif",
        protein_dir_name="1vme_0.5occA_0.5occB",
    )


class TestTrial:
    def test_field_access(self, trial: Trial):
        """Fields set at construction are readable, matching script access patterns."""
        assert trial.protein == "1vme"
        assert trial.altloc_occupancies == {"A": 1}
        assert trial.model == "boltz2"
        assert trial.scaler == "pure_guidance"
        assert trial.ensemble_size == 10
        assert trial.guidance_weight == 1.0
        assert trial.protein_dir_name == "1vme_0.5occA_0.5occB"
        assert isinstance(trial.trial_dir, Path)
        assert isinstance(trial.refined_cif_path, Path)

    def test_defaults(self, trial: Trial):
        """rscc defaults to nan, base_map_path defaults to None, error defaults to None"""
        assert math.isnan(trial.rscc)
        assert trial.base_map_path is None
        assert trial.error is None

    def test_nullable_params_accept_none(self, tmp_path: Path):
        """method, guidance_weight, gd_steps can be None (e.g. protenix / pure_guidance trials)."""
        t = Trial(
            protein="6b8x",
            altloc_occupancies={"A": 0.3, "B": 0.7},
            model="protenix",
            method=None,
            scaler="pure_guidance",
            ensemble_size=5,
            guidance_weight=None,
            gd_steps=None,
            trial_dir=tmp_path,
            refined_cif_path=tmp_path / "refined.cif",
            protein_dir_name="6b8x_0.3occA",
        )
        assert t.method is None
        assert t.guidance_weight is None
        assert t.gd_steps is None

    def test_dict_copy_contains_all_fields(self, trial: Trial):
        """trial.__dict__.copy() — the pattern used in rscc and lddt scripts to seed result rows —
        contains all expected keys."""
        d = trial.__dict__.copy()
        expected_keys = {
            "protein",
            "altloc_occupancies",
            "model",
            "method",
            "scaler",
            "ensemble_size",
            "guidance_weight",
            "gd_steps",
            "trial_dir",
            "refined_cif_path",
            "protein_dir_name",
            "input_structure_path",
            "density_path",
            "resolution",
            "sampler",
            "rscc",
            "base_map_path",
            "error",
        }
        assert expected_keys == set(d.keys())

    def test_dict_copy_can_be_extended(self, trial: Trial):
        """Scripts add keys like 'selection' and 'rscc' to the dict copy."""
        d = trial.__dict__.copy()
        d["selection"] = "chain A and resi 1-10"
        d["rscc"] = 0.95
        assert not hasattr(trial, "selection")
        assert math.isnan(trial.rscc)  # original unchanged


class TestTrialList:
    def test_is_a_list(self, trial: Trial):
        tl = TrialList([trial])
        assert isinstance(tl, list)

    def test_len_and_iteration(self, trial: Trial):
        tl = TrialList([trial, trial])
        assert len(tl) == 2
        for t in tl:
            assert t.protein == "1vme"

    def test_extend(self, trial: Trial):
        """extend() is called in scan_grid_search_results to merge recursive results."""
        tl1 = TrialList([trial])
        tl2 = TrialList([trial])
        tl1.extend(tl2)
        assert len(tl1) == 2

    def test_summarize_logs_proteins_models_scalers(self, trial: Trial, caplog):
        """summarize() logs the unique proteins, models, and scalers — called right after
        scan_grid_search_results in every eval script."""
        import logging

        tl = TrialList([trial])
        with caplog.at_level(logging.INFO):
            tl.summarize()
        assert "1vme" in caplog.text
        assert "boltz2" in caplog.text
        assert "pure_guidance" in caplog.text

    def test_summarize_empty_list(self, caplog):
        """summarize() on an empty TrialList does not raise."""
        import logging

        tl = TrialList()
        with caplog.at_level(logging.INFO):
            tl.summarize()  # should not raise


@pytest.fixture
def protein_config(tmp_path: Path) -> ProteinConfig:
    return ProteinConfig(
        protein="test",
        base_map_dir=tmp_path,
        selection=["chain A"],
        resolution=2.0,
        map_pattern="{occ_str}.ccp4",
    )


class TestProteinConfigGetBaseMapPath:
    """Tests for ProteinConfig.get_base_map_path_for_occupancy (accepts dict, not float)."""

    def test_returns_path_when_file_exists(self, protein_config, tmp_path):
        map_file = tmp_path / "0.5occA_0.5occB.ccp4"
        map_file.touch()
        assert protein_config.get_base_map_path_for_occupancy({"A": 0.5, "B": 0.5}) == map_file

    def test_returns_none_when_file_missing(self, protein_config):
        assert protein_config.get_base_map_path_for_occupancy({"A": 0.5, "B": 0.5}) is None

    def test_zero_occ_altloc_dropped_from_filename(self, protein_config, tmp_path):
        """{"A": 1.0, "B": 0.0} should resolve to "1.0occA.ccp4", drop the zero-occ altloc."""
        map_file = tmp_path / "1.0occA.ccp4"
        map_file.touch()
        assert protein_config.get_base_map_path_for_occupancy({"A": 1.0, "B": 0.0}) == map_file

    def test_all_zero_occupancies_returns_none_with_warning(self, protein_config, caplog):
        """All-zero dict → occupancy_to_str raises → None + warning."""
        result = protein_config.get_base_map_path_for_occupancy({"A": 0.0, "B": 0.0})
        assert result is None
        assert "Cannot determine occupancy string" in caplog.text

    def test_sum_exceeds_one_returns_none_with_warning(self, protein_config, caplog):
        """Sum > 1 → occupancy_to_str raises → None + warning."""
        result = protein_config.get_base_map_path_for_occupancy({"A": 0.8, "B": 0.8})
        assert result is None
        assert "Cannot determine occupancy string" in caplog.text
