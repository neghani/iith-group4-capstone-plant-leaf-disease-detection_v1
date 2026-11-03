"""Publish gate for v1.1 / v1.2 and --branch dry-run wiring (no Hub)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from publish import GATED_TAGS, publish_gate


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
