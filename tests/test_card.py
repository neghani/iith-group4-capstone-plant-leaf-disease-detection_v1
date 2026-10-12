"""Assemble the Hub model card from artifacts. No Hub, no torch."""

import json
from pathlib import Path

from card import NOT_RECORDED, assemble_card, epoch_rows
from stats import frac_ci

TEMPLATE = """# plantvillage-mobilenetv2

<!-- WEAK_CLASSES_START -->
Not recorded.
<!-- WEAK_CLASSES_END -->

<!-- BEFORE_AFTER_START -->
Not recorded.
<!-- BEFORE_AFTER_END -->

<!-- CALIBRATION_START -->
Not recorded.
<!-- CALIBRATION_END -->

<!-- LATENCY_START -->
Not recorded.
<!-- LATENCY_END -->

<!-- STRATEGIES_START -->
Not recorded.
<!-- STRATEGIES_END -->

<!-- EPOCHS_START -->
Not recorded.
<!-- EPOCHS_END -->
"""

STRATEGY_MD = """# Crop strategy comparison (tune split)

| Strategy | Thr | Pass rate (95% CI) | Top-1 reached | Wrong-confident (of ok) | Not sure | Gate | s/img | vs B (95% CI) |
|---|---|---|---|---|---|---|---|---|
| A whole frame | 0.50 | 2/10 (20.0%), 5.7-51.0% | 3/10 | 1/3 | 5/10 | 0/10 | 0.100 | +0.0 pts |
| B sharpest leaf | 0.85 | 3/10 (30.0%), 10.8-60.3% | 4/10 | 1/4 | 4/10 | 0/10 | 0.120 | reference |
| C most symptomatic | 0.80 | 2/10 (20.0%), 5.7-51.0% | 3/10 | 0/2 | 6/10 | 0/10 | 0.130 | -10.0 pts |
| D top-3 leaves | 0.70 | 2/10 (20.0%), 5.7-51.0% | 3/10 | 1/3 | 5/10 | 0/10 | 0.200 | -10.0 pts |
| E tight patch | 0.60 | 1/10 (10.0%), 1.8-40.4% | 2/10 | 1/2 | 7/10 | 0/10 | 0.110 | -20.0 pts |

## Decision

**Chosen: strategy B.**
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_epoch_rows_one_per_epoch_prefers_plantdoc():
    history = [
        {"loss": 0.8, "epoch": 0.5, "step": 10},
        {"eval_plantdoc_loss": 0.40, "eval_plantdoc_accuracy": 0.80,
         "eval_plantvillage_loss": 0.10, "eval_plantvillage_accuracy": 0.99, "epoch": 1.0},
        {"eval_plantdoc_loss": 0.30, "eval_plantdoc_accuracy": 0.85, "epoch": 2.0},
        {"loss": 0.20, "eval_plantdoc_loss": 0.35, "eval_plantdoc_accuracy": 0.84, "epoch": 3.0},
        {"eval_plantdoc_loss": 0.36, "eval_plantdoc_accuracy": 0.83, "epoch": 3.0},
    ]
    rows = epoch_rows(history)
    assert [r["epoch"] for r in rows] == [1.0, 2.0, 3.0]
    assert rows[0]["eval_loss"] == 0.40
    assert rows[0]["eval_accuracy"] == 0.80
    assert "train_loss" not in rows[0]
    assert rows[2]["eval_loss"] == 0.36
    assert "train_loss" not in rows[2]


def test_assemble_card_fills_all_six_sections(tmp_path):
    model_dir = tmp_path / "model"
    outputs = tmp_path / "outputs"
    model_dir.mkdir()
    _write(model_dir / "README.md", TEMPLATE)
    _write(model_dir / "trainer_state.json", json.dumps({
        "log_history": [
            {"eval_loss": 0.40, "eval_accuracy": 0.90, "epoch": 1.0},
            {"eval_loss": 0.25, "eval_accuracy": 0.95, "epoch": 2.0},
            {"loss": 0.18, "eval_loss": 0.30, "eval_accuracy": 0.94, "epoch": 3.0},
        ]
    }))
    _write(model_dir / "reports" / "metrics.json", json.dumps({
        "per_class": [
            {"class": "Potato___healthy", "precision": 0.80, "recall": 0.733, "f1": 0.765, "support": 15},
            {"class": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
             "precision": 0.86, "recall": 0.91, "f1": 0.884, "support": 50},
            {"class": "Tomato___healthy", "precision": 0.99, "recall": 0.99, "f1": 0.99, "support": 100},
        ]
    }))
    v10_label = frac_ci(3, 25)
    v11_label = frac_ci(12, 25)
    _write(model_dir / "final_summary.json", json.dumps({
        "v1.0": {"real_world_set": {"passes": 3, "n": 25, "rate": 0.12,
                                    "ci_low": 0.042, "ci_high": 0.30, "label": v10_label}},
        "v1.1": {"real_world_set": {"passes": 12, "n": 25, "rate": 0.48,
                                    "ci_low": 0.30, "ci_high": 0.66, "label": v11_label}},
    }))
    _write(model_dir / "calibration.json", json.dumps({
        "ece_before": 0.182, "ece_after": 0.041, "temperature": 1.25, "bins": 15,
    }))
    _write(outputs / "reports" / "reliability.png", "png")
    _write(outputs / "strategy_comparison.md", STRATEGY_MD)
    latency = tmp_path / "latency.json"
    _write(latency, json.dumps({
        "model_forward_ms": 52.6,
        "predict_warm_s": 1.5,
        "predict_warm_label": "about 1.5 s",
        "note": "The gap is the app path (quality gate, crop, HTTP), not the forward pass.",
    }))

    status = assemble_card(model_dir, outputs=outputs, latency_path=latency)
    assert status["missing"] == []
    assert status["filled"] == [
        "epochs", "weak_classes", "before_after", "calibration", "latency", "strategies",
    ]
    card = (model_dir / "README.md").read_text()

    assert "| Epoch | Val loss | Val accuracy | Train loss |" in card
    assert "| 1 | 0.4000 | 90.0% |" in card
    assert "| 3 | 0.3000 | 94.0% | 0.1800 |" in card
    assert "overfitting" in card

    assert "Potato healthy" in card
    assert "Corn Cercospora" in card
    assert "76.5%" in card
    assert "88.4%" in card

    assert v10_label in card
    assert v11_label in card
    assert "v1.0 (before fine-tune)" in card
    assert "v1.1 (after fine-tune)" in card
    assert "25" in card
    assert "CI" in card

    assert "18.2%" in card
    assert "4.1%" in card
    assert "![Reliability](reports/reliability.png)" in card
    assert (model_dir / "reports" / "reliability.png").exists()

    assert "52.6" in card
    assert "1.5" in card

    for letter in "ABCDE":
        assert f"| {letter} " in card


def test_write_final_summary_real_world_stratum(tmp_path):
    from final_test import write_final_summary

    rows = [{"stratum": "real_world_set"}] * 25 + [{"stratum": "plantdoc_test"}] * 5
    m10 = {"pass_vector": [True] * 3 + [False] * 27}
    m11 = {"pass_vector": [True] * 12 + [False] * 18}
    path = tmp_path / "final_summary.json"
    summary = write_final_summary(path, m10, m11, rows, 0.5, 1.25)
    rw10, rw11 = summary["v1.0"]["real_world_set"], summary["v1.1"]["real_world_set"]
    assert rw10["passes"] == 3 and rw10["n"] == 25
    assert rw11["passes"] == 12 and rw11["n"] == 25
    assert rw10["label"] == frac_ci(3, 25)
    assert rw11["label"] == frac_ci(12, 25)
    assert json.loads(path.read_text())["v1.1"]["temperature"] == 1.25


def test_assemble_card_missing_sections_stay_not_recorded(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write(model_dir / "README.md", TEMPLATE)
    status = assemble_card(model_dir, outputs=tmp_path / "outputs",
                           latency_path=tmp_path / "missing-latency.json")
    assert status["filled"] == []
    assert status["missing"] == [
        "epochs", "weak_classes", "before_after", "calibration", "latency", "strategies",
    ]
    card = (model_dir / "README.md").read_text()
    assert card.count(NOT_RECORDED) == 6
    assert "52.6" not in card
    assert "Potato healthy" not in card
