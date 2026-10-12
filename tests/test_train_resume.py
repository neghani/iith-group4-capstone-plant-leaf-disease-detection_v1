"""Refuse auto-resume when Drive checkpoints were trained from a different CODE_COMMIT."""

import json
from pathlib import Path

import pytest

from common import checkpoint_code_commit, refuse_stale_resume


def test_checkpoint_code_commit_reads_run_info(tmp_path: Path):
    (tmp_path / "run_info.json").write_text(json.dumps({"code_commit": "abc123"}))
    assert checkpoint_code_commit(tmp_path) == "abc123"


def test_checkpoint_code_commit_missing_file(tmp_path: Path):
    assert checkpoint_code_commit(tmp_path) is None


def test_refuse_stale_resume_exits_on_mismatch(tmp_path: Path):
    (tmp_path / "run_info.json").write_text(json.dumps({"code_commit": "oldsha"}))
    with pytest.raises(SystemExit, match="oldsha"):
        refuse_stale_resume(tmp_path, "newsha")


def test_refuse_stale_resume_allows_same_commit(tmp_path: Path):
    (tmp_path / "run_info.json").write_text(json.dumps({"code_commit": "same"}))
    refuse_stale_resume(tmp_path, "same")


def test_refuse_stale_resume_allows_missing_run_info(tmp_path: Path):
    refuse_stale_resume(tmp_path, "newsha")


def test_force_resume_allows_mismatch(tmp_path: Path, monkeypatch):
    (tmp_path / "run_info.json").write_text(json.dumps({"code_commit": "oldsha"}))
    monkeypatch.setenv("FORCE_RESUME", "1")
    refuse_stale_resume(tmp_path, "newsha")
