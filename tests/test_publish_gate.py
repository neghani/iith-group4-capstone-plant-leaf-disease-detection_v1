"""Publish gate for v1.1 / v1.2 and --branch dry-run wiring (no Hub)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from publish import GATED_TAGS, publish_gate, refuse_if_tag_exists


def _model_dir(tmp: Path, *, pd_f1: float, pv_acc: float) -> Path:
    d = tmp / "model"
    d.mkdir()
    (d / "eval_results.json").write_text(json.dumps({"eval_plantdoc_macro_f1": pd_f1}))
    reports = d / "reports"
    reports.mkdir()
    (reports / "metrics.json").write_text(json.dumps({"accuracy": pv_acc}))
    return d


def _diag(tmp: Path, f1: float) -> Path:
    p = tmp / "diagnostics.json"
    p.write_text(json.dumps({"plantdoc_macro_f1": f1}))
    return p


def test_gated_tags_map():
    assert GATED_TAGS["v1.1"] == "v1.0"
    assert GATED_TAGS["v1.2"] == "v1.1"


def test_v12_gate_passes_when_f1_beats_v11(tmp_path):
    model = _model_dir(tmp_path, pd_f1=0.55, pv_acc=0.99)
    diag = _diag(tmp_path, 0.50)
    publish_gate(model, "v1.2", diag)


def test_v12_gate_fails_when_f1_not_higher(tmp_path):
    model = _model_dir(tmp_path, pd_f1=0.50, pv_acc=0.99)
    diag = _diag(tmp_path, 0.50)
    with pytest.raises(SystemExit, match="v1.2 publish gate failed"):
        publish_gate(model, "v1.2", diag)


def test_v12_gate_fails_on_pv_drop(tmp_path):
    model = _model_dir(tmp_path, pd_f1=0.60, pv_acc=0.90)
    diag = _diag(tmp_path, 0.50)
    with pytest.raises(SystemExit, match="PlantVillage test accuracy"):
        publish_gate(model, "v1.2", diag)


def test_v10_tag_skips_gate(tmp_path):
    model = _model_dir(tmp_path, pd_f1=0.1, pv_acc=0.1)
    publish_gate(model, "v1.0", tmp_path / "missing.json")


class _FakeRefs:
    def __init__(self, tags):
        self.tags = [type("T", (), {"name": t})() for t in tags]


class _FakeApi:
    def __init__(self, tags):
        self._tags = tags

    def list_repo_refs(self, repo_id, repo_type=None):
        return _FakeRefs(self._tags)


def test_refuse_if_v12_tag_exists():
    with pytest.raises(SystemExit, match="already exists"):
        refuse_if_tag_exists(_FakeApi(["v1.1", "v1.2"]), "neghani/plantvillage-mobilenetv2", "v1.2")


def test_refuse_allows_missing_v12_tag():
    refuse_if_tag_exists(_FakeApi(["v1.0", "v1.1"]), "neghani/plantvillage-mobilenetv2", "v1.2")


def test_refuse_skips_non_v12_tags():
    # Even if v1.1 exists, refuse_if_tag_exists only guards v1.2.
    refuse_if_tag_exists(_FakeApi(["v1.1"]), "neghani/plantvillage-mobilenetv2", "v1.1")


def test_release_v12_preset_sets_branch_tag_and_messages(tmp_path, monkeypatch):
    import publish

    monkeypatch.setattr("sys.argv", ["publish.py", "--release", "v1.2", "--model", str(tmp_path / "stage2")])
    args = publish.parse_args()
    assert args.branch == "v1.2"
    assert args.tag == "v1.2"
    assert args.tag_message.startswith("MobileNetV2 v1.2")
    assert args.commit_message.startswith("MobileNetV2 v1.2")
    # The release preset never skips the card check; that bypass is debug-only.
    assert args.skip_card_check is False


def test_plain_run_keeps_phase1_defaults(tmp_path, monkeypatch):
    import publish

    monkeypatch.setattr("sys.argv", ["publish.py", "--model", str(tmp_path / "m")])
    args = publish.parse_args()
    assert args.tag == "v1.0"
    assert args.branch is None
    assert args.commit_message == "Phase 1: ResNet-18 on PlantVillage"


def test_explicit_flags_override_release_preset(tmp_path, monkeypatch):
    import publish

    monkeypatch.setattr("sys.argv", ["publish.py", "--release", "v1.2", "--branch", "scratch", "--model", str(tmp_path / "s")])
    args = publish.parse_args()
    assert args.branch == "scratch"
    assert args.tag == "v1.2"


def test_default_diagnostics_path_for_v12_is_v12_artifacts(tmp_path):
    import publish

    stage2 = tmp_path / "v1.2" / "stage2"
    assert publish.default_diagnostics_path(stage2, "v1.2") == tmp_path / "v1.2" / "artifacts" / "diagnostics_v1_1.json"


def test_ensure_diagnostics_reuses_existing_file(tmp_path, monkeypatch):
    import publish

    path = tmp_path / "diagnostics_v1_1.json"
    path.write_text("{}")

    def no_subprocess(*a, **k):
        raise AssertionError("must not recompute an existing baseline")

    monkeypatch.setattr(publish.subprocess, "run", no_subprocess)
    assert publish.ensure_diagnostics(path, "v1.1", "neghani/plantvillage-mobilenetv2") == path


def test_ensure_diagnostics_computes_missing_baseline_once(tmp_path, monkeypatch):
    import publish

    artifacts = tmp_path / "artifacts"
    path = artifacts / "diagnostics_v1_1.json"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "model.diagnostics" in cmd:
            artifacts.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        return None

    monkeypatch.setattr(publish.subprocess, "run", fake_run)
    monkeypatch.setattr(publish, "HERE", tmp_path / "model")  # no manifest here, so prep runs
    assert publish.ensure_diagnostics(path, "v1.1", "neghani/plantvillage-mobilenetv2") == path
    modules = [c[c.index("-m") + 1] for c in calls]
    assert modules == ["model.prepare_plantdoc", "model.diagnostics"]
    diag_cmd = calls[1]
    assert diag_cmd[diag_cmd.index("--revision") + 1] == "v1.1"
    assert diag_cmd[diag_cmd.index("--model") + 1] == "neghani/plantvillage-mobilenetv2"


def test_release_v12_targets_mobilenet_repo_not_resnet(tmp_path, monkeypatch):
    import publish

    monkeypatch.setattr("sys.argv", ["publish.py", "--release", "v1.2", "--model", str(tmp_path / "s")])
    assert publish.parse_args().hub_model_id == "neghani/plantvillage-mobilenetv2"
    monkeypatch.setattr("sys.argv", ["publish.py", "--model", str(tmp_path / "m")])
    assert publish.parse_args().hub_model_id == publish.DEFAULT_HF_MODEL_REPO
