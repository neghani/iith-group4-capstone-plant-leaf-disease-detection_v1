"""Final-test guard: Drive JSON named by revision; Hub backend; smoke never touches it."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifacts import collect
from guard import DirGuard, HubGuard, check, make_guard


def test_dir_guard_roundtrip(tmp_path):
    g = DirGuard(tmp_path)
    assert g.read("v1.1") is None
    g.write("v1.1", {"date": "2026-10-11", "model_key": "abc"})
    rec = g.read("v1.1")
    assert rec["model_key"] == "abc"
    assert (tmp_path / "v1.1.json").exists()
    assert "v1.1.json" in g.describe("v1.1")


def test_check_blocks_second_run(tmp_path):
    g = DirGuard(tmp_path)
    g.write("v1.1", {"date": "2026-10-11", "model_key": "aaa"})
    with pytest.raises(SystemExit, match="already ran"):
        check(g, "v1.1", "aaa", force=False)
    check(g, "v1.1", "aaa", force=True)  # --force allowed
    check(g, "v1.2", "bbb", force=False)  # different revision is a first run


def test_check_mentions_new_revision_when_weights_differ(tmp_path):
    g = DirGuard(tmp_path)
    g.write("v1.1", {"date": "2026-10-11", "model_key": "old"})
    with pytest.raises(SystemExit, match="new revision"):
        check(g, "v1.1", "new-weights", force=False)


def test_make_guard_drive_and_hub():
    d = make_guard("drive:/tmp/guards")
    assert isinstance(d, DirGuard)
    h = make_guard("hub:neghani/plantvillage-mobilenetv2")
    assert isinstance(h, HubGuard)
    assert h.repo_id == "neghani/plantvillage-mobilenetv2"
    with pytest.raises(ValueError, match="plain tag"):
        DirGuard("/tmp").describe("v1.1/../x")


def test_hub_guard_uses_injected_api(tmp_path):
    stored = {}

    class Refs:
        branches = [SimpleNamespace(name="stage2-runs")]

    class Api:
        def list_repo_refs(self, repo_id):
            return Refs()

        def file_exists(self, repo_id, path, revision=None):
            return (revision, path) in stored

        def hf_hub_download(self, repo_id, path, revision=None, force_download=False):
            dest = tmp_path / "dl.json"
            dest.write_text(json.dumps(stored[(revision, path)]))
            return str(dest)

        def create_branch(self, repo_id, branch, exist_ok=True):
            return None

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, revision, commit_message):
            stored[(revision, path_in_repo)] = json.loads(Path(path_or_fileobj).read_text())

    g = HubGuard("owner/model", api=Api())
    assert g.read("v1.1") is None
    g.write("v1.1", {"n": 50})
    assert g.read("v1.1")["n"] == 50
    assert g.describe("v1.1").startswith("hf://owner/model@stage2-runs/")


def test_smoke_refuses_guard_flag():
    from final_test import resolve_guard

    with pytest.raises(SystemExit, match="never reads or writes"):
        resolve_guard(argparse.Namespace(smoke=True, guard="drive:/tmp"))
    assert resolve_guard(argparse.Namespace(smoke=True, guard=None)) is None


def test_artifacts_skip_weights_and_missing(tmp_path):
    (tmp_path / "run_info.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "checkpoint-1").mkdir()
    (tmp_path / "checkpoint-1" / "notes.txt").write_text("no")
    files = collect([str(tmp_path / "run_info.json"), str(tmp_path / "model.safetensors"),
                     str(tmp_path / "missing.json"), str(tmp_path / "checkpoint-1")])
    names = [rel for _, rel in files]
    assert names == ["run_info.json"]
