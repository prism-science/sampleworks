"""Tests for MSAManager and the MSA cache helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from sampleworks.utils import msa as msa_module
from sampleworks.utils.guidance_constants import StructurePredictor
from sampleworks.utils.msa import _compute_msa, _msa_data_key_sort_key, MSAManager


# ============================================================================
# _hash_arguments (test 5)
# ============================================================================


def test_hash_arguments_is_deterministic_and_input_sensitive():
    data = {"A": "MKTAY", "B": "GGGAA"}
    base = MSAManager._hash_arguments(data, "greedy")

    assert base == MSAManager._hash_arguments({"A": "MKTAY", "B": "GGGAA"}, "greedy")

    # Different sequence -> different hash.
    assert base != MSAManager._hash_arguments({"A": "MKTAY", "B": "GGGAC"}, "greedy")
    # Different pairing strategy -> different hash.
    assert base != MSAManager._hash_arguments(data, "complete")
    # Different value order -> different hash (hash is over tuple(values)).
    assert base != MSAManager._hash_arguments({"B": "GGGAA", "A": "MKTAY"}, "greedy")


def test_msa_data_key_sort_preserves_numeric_order_and_handles_mixed_keys():
    """Ensure Protenix MSA key sorting preserves numeric order and mixed-key stability."""
    keys = [10, 2, "1", "A"]

    assert sorted(keys, key=_msa_data_key_sort_key) == [2, 10, "1", "A"]


# ============================================================================
# __init__ cache directory handling (test 4)
# ============================================================================


def test_cache_dir_explicit_is_used(tmp_path: Path):
    manager = MSAManager(msa_cache_dir=tmp_path)

    assert manager.msa_dir == tmp_path
    assert manager.msa_dir.exists()


def test_cache_dir_explicit_string_converted_to_path(tmp_path: Path):
    manager = MSAManager(msa_cache_dir=str(tmp_path))

    assert isinstance(manager.msa_dir, Path)
    assert manager.msa_dir == tmp_path


def test_cache_dir_explicit_is_created_when_missing(tmp_path: Path):
    cache_dir = tmp_path / "nested" / "msa"

    manager = MSAManager(msa_cache_dir=cache_dir)

    assert manager.msa_dir == cache_dir
    assert cache_dir.is_dir()


@pytest.mark.parametrize("env_value", [None, "", "  "], ids=["unset", "empty", "blank"])
def test_cache_dir_default_created_under_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_value: str | None
):
    # Redirect Path.home() so we don't pollute the real home directory.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    if env_value is None:
        monkeypatch.delenv(msa_module.MSA_CACHE_DIR_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(msa_module.MSA_CACHE_DIR_ENV_VAR, env_value)

    manager = MSAManager(msa_cache_dir=None)

    expected = tmp_path / ".sampleworks" / "msa"
    assert manager.msa_dir == expected
    assert expected.exists()


def test_cache_dir_from_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache_dir = tmp_path / "shared" / "msa"
    monkeypatch.setenv(msa_module.MSA_CACHE_DIR_ENV_VAR, str(cache_dir))

    manager = MSAManager()

    assert manager.msa_dir == cache_dir
    assert cache_dir.is_dir()


def test_cache_dir_environment_variable_expands_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(msa_module.MSA_CACHE_DIR_ENV_VAR, "~/msa-cache")

    manager = MSAManager()

    assert manager.msa_dir == tmp_path / "msa-cache"
    assert manager.msa_dir.is_dir()


def test_cache_dir_explicit_overrides_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(msa_module.MSA_CACHE_DIR_ENV_VAR, str(tmp_path / "from_env"))

    manager = MSAManager(msa_cache_dir=tmp_path / "explicit")

    assert manager.msa_dir == tmp_path / "explicit"
    assert not (tmp_path / "from_env").exists()


# ============================================================================
# get_msa -> _compute_msa wiring (test 1)
# ============================================================================


def _write_valid_cache_files(msa_dir: Path, hash_key: str, sequences: list[str]) -> None:
    """Write a matching csv/a3m pair so _validate_msa_cache_contents passes."""
    csv_lines = ["key,sequence"] + [f"-1,{s}" for s in sequences]
    (msa_dir / f"{hash_key}_0.csv").write_text("\n".join(csv_lines))

    a3m_lines = []
    for i, s in enumerate(sequences):
        a3m_lines.append(f">{hash_key}_0_{i}")
        a3m_lines.append(s)
    (msa_dir / f"{hash_key}_0.a3m").write_text("\n".join(a3m_lines) + "\n")


def test_get_msa_calls_compute_when_cache_missing_and_skips_when_present(tmp_path: Path):
    manager = MSAManager(msa_cache_dir=tmp_path)
    data = {"A": "MKTAY"}
    pairing = "greedy"
    hash_key = MSAManager._hash_arguments(data, pairing)

    def fake_compute(data_arg, target_id, msa_dir, *args, **kwargs):
        _write_valid_cache_files(msa_dir, target_id, ["MKTAY", "MKTAC"])
        return {name: msa_dir / f"{target_id}_0.csv" for name in data_arg}

    with patch.object(msa_module, "_compute_msa", side_effect=fake_compute) as mock_compute:
        result = manager.get_msa(data, pairing, structure_predictor=StructurePredictor.BOLTZ_2)

        assert mock_compute.call_count == 1
        # First positional arg is data, second is the hash used as target_id.
        call_args = mock_compute.call_args
        assert call_args.args[0] == data
        assert call_args.args[1] == hash_key
        assert call_args.args[2] == tmp_path

        assert result == {"A": tmp_path / f"{hash_key}_0.csv"}
        assert manager._api_calls == 1
        assert manager._cache_hits == 0

        # Second call should hit the cache and not invoke compute again.
        mock_compute.reset_mock()
        manager.get_msa(data, pairing, structure_predictor=StructurePredictor.BOLTZ_2)
        mock_compute.assert_not_called()
        assert manager._cache_hits == 1


# ============================================================================
# _compute_msa argument forwarding to run_mmseqs2 (test 2)
# ============================================================================


def test_compute_msa_calls_run_mmseqs2_with_correct_arguments(tmp_path: Path):
    data = {"A": "AAA", "B": "BBB"}
    target_id = "tgt"
    fake_return = [">q\nAAA\n", ">q\nBBB\n"]

    with patch.object(msa_module, "run_mmseqs2", return_value=fake_return) as mock_run:
        _compute_msa(
            data=data,
            target_id=target_id,
            msa_dir=tmp_path,
            msa_server_url="https://example.org/api",
            msa_pairing_strategy="greedy",
            api_key_header="X-Custom-Key",
            api_key_value="secret",
        )

    # With len(data) == 2, both the paired and unpaired branches should fire.
    assert mock_run.call_count == 2

    paired_call, unpaired_call = mock_run.call_args_list

    # Paired call.
    assert paired_call.args[0] == ["AAA", "BBB"]
    assert paired_call.args[1] == str(tmp_path / f"{target_id}_paired_tmp")
    assert paired_call.kwargs["use_pairing"] is True
    assert paired_call.kwargs["use_env"] is True
    assert paired_call.kwargs["host_url"] == "https://example.org/api"
    assert paired_call.kwargs["pairing_strategy"] == "greedy"
    assert paired_call.kwargs["auth_headers"] == {
        "Content-Type": "application/json",
        "X-Custom-Key": "secret",
    }

    # Unpaired call: should forward the same env/auth flags as the paired call.
    assert unpaired_call.args[0] == ["AAA", "BBB"]
    assert unpaired_call.args[1] == str(tmp_path / f"{target_id}_unpaired_tmp")
    assert unpaired_call.kwargs["use_pairing"] is False
    assert unpaired_call.kwargs["use_env"] is True
    assert unpaired_call.kwargs["host_url"] == "https://example.org/api"
    assert unpaired_call.kwargs["pairing_strategy"] == "greedy"
    assert unpaired_call.kwargs["auth_headers"] == {
        "Content-Type": "application/json",
        "X-Custom-Key": "secret",
    }


# ============================================================================
# _compute_msa file output consistency (test 3)
# ============================================================================


def test_compute_msa_writes_matching_csv_and_a3m(tmp_path: Path):
    data = {"chainA": "MKTAY"}
    target_id = "thehash"
    # FASTA-style: alternating header / sequence lines. _compute_msa parses [1::2].
    fake_return = [">query\nMKTAY\n>hit1\nMKTAC\n>hit2\nMKTAG\n"]

    with patch.object(msa_module, "run_mmseqs2", return_value=fake_return):
        _compute_msa(
            data=data,
            target_id=target_id,
            msa_dir=tmp_path,
            msa_server_url="https://example.org/api",
            msa_pairing_strategy="greedy",
        )

    csv_path = tmp_path / f"{target_id}_0.csv"
    a3m_path = tmp_path / f"{target_id}_0.a3m"
    assert csv_path.exists()
    assert a3m_path.exists()

    csv_lines = csv_path.read_text().splitlines()
    assert csv_lines[0] == "key,sequence"
    csv_sequences = [line.split(",", 1)[1] for line in csv_lines[1:]]

    a3m_lines = a3m_path.read_text().splitlines()
    a3m_sequences = [line for i, line in enumerate(a3m_lines) if i % 2 == 1]

    assert csv_sequences == a3m_sequences == ["MKTAY", "MKTAC", "MKTAG"]


# ============================================================================
# get_msa Protenix cache path (regression guard)
# ============================================================================


def test_get_msa_protenix_caches_under_protenix_subdir_and_skips_second_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Protenix MSAs are cached under ``msa_dir/protenix/<hash>/<key>/`` (written by
    ``protenix_msa_search``), independent of the boltz/rf3 ``_compute_msa`` path. A warm
    cache must not re-invoke ``protenix_msa_search`` — that call is the live MSA-server
    request we must not regress into. Also guards that the Protenix path does not depend
    on any files under ``msa_dir/<idx>/``.
    """
    manager = MSAManager(msa_cache_dir=tmp_path)
    data = {"A": "MKTAY"}
    pairing = "greedy"
    hash_key = MSAManager._hash_arguments(data, pairing)
    out_dir = tmp_path / "protenix" / hash_key

    call_count = {"n": 0}

    def fake_protenix_search(sequences, search_out_dir, mode):
        # Mirror protenix_msa_search: write per-key non_pairing/pairing.a3m into out_dir.
        call_count["n"] += 1
        search_out_dir = Path(search_out_dir)
        dirs = []
        for key in sorted(data.keys()):
            d = search_out_dir / str(key)
            d.mkdir(parents=True, exist_ok=True)
            (d / "non_pairing.a3m").write_text(f">{key}\n{data[key]}\n")
            (d / "pairing.a3m").write_text(f">{key}\n{data[key]}\n")
            dirs.append(d)
        return dirs

    monkeypatch.setattr(msa_module, "PROTENIX_AVAILABLE", True)
    monkeypatch.setattr(msa_module, "protenix_msa_search", fake_protenix_search, raising=False)

    # Cold cache: the search runs once and writes under msa_dir/protenix/<hash>/<key>/.
    result = manager.get_msa(data, pairing, structure_predictor=StructurePredictor.PROTENIX)
    assert call_count["n"] == 1
    assert manager._api_calls == 1
    assert manager._cache_hits == 0
    assert (out_dir / "A" / "non_pairing.a3m").exists()
    # The Protenix path never reads the (removed) msa_dir/<idx>/ files.
    assert not (tmp_path / "0").exists()
    assert result == {"A": out_dir / "A"}

    # Warm cache: an identical request must NOT hit the MSA server again.
    manager.get_msa(data, pairing, structure_predictor=StructurePredictor.PROTENIX)
    assert call_count["n"] == 1
    assert manager._api_calls == 1
    assert manager._cache_hits == 1
