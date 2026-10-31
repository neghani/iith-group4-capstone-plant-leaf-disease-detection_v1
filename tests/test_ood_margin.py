"""OOD margin gate: flat top-1/top-2 softmax → not_sure."""

import json
from pathlib import Path

from shared.vision.strategies import decide_status

REPO = Path(__file__).resolve().parents[1]


def test_decide_status_rejects_low_margin():
    assert decide_status(0.9, 1.0, 0.5, margin=0.01, ood_threshold=0.05) == "not_sure"


def test_decide_status_keeps_confident_separated_top1():
    assert decide_status(0.9, 1.0, 0.5, margin=0.4, ood_threshold=0.05) == "ok"


def test_decide_status_confidence_still_applies():
    assert decide_status(0.4, 1.0, 0.5, margin=0.3, ood_threshold=0.05) == "not_sure"


def test_ood_threshold_zero_disables_margin_gate():
    assert decide_status(0.9, 1.0, 0.5, margin=0.0, ood_threshold=0.0) == "ok"


def test_baseline_shows_mobilenet_rejects_strawberry_resnet_does_not():
    summary = json.loads((REPO / "model/outputs/ood/baseline_summary.json").read_text())
    local = summary["local_whole_frame"]
    assert local["mobilenet"]["max_softmax"] < 0.5
    assert local["resnet"]["max_softmax"] >= 0.5
    assert summary["api_resnet_v1.0"]["report_status"] in ("confirmed_likely", "possible")
    assert (REPO / "model/outputs/ood/FALLBACK.md").is_file()
