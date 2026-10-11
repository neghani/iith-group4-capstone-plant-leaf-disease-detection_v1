"""Fit one temperature T on PlantDoc validation logits; sweep the confidence threshold.

    python model/calibrate.py --model model/outputs/plantvillage-plantdoc-mobilenetv2

Never opens the final test. If no threshold reaches 70% precision among "ok" answers,
threshold.json is marked provisional and the config threshold is left unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import Predictor, load_image  # noqa: E402
from shared.vision.calibration import apply_temperature  # noqa: E402
from stats import (  # noqa: E402
    MIN_PRECISION,
    SWEEP,
    ece,
    fit_temperature,
    frac,
    nll,
    reliability,
    reliability_md,
    select_threshold,
    softmax,
    threshold_sweep,
)


def load_val(manifest: Path, extra_dir: Path) -> tuple[list, list[str]]:
    rows = [r for r in csv.DictReader(open(manifest, newline="", encoding="utf-8")) if r["split"] == "validation"]
    images, labels = [], []
    for r in rows:
        path = extra_dir / r["file"]
        if not path.exists():
            raise SystemExit(f"missing {path}")
        images.append(load_image(path))
        labels.append(r["label"])
    if not images:
        raise SystemExit(f"No validation rows in {manifest}")
    return images, labels


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="Local run dir (the fine-tuned weights).")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--manifest", default=str(HERE / "data" / "plantdoc_train" / "manifest.csv"))
    ap.add_argument("--extra_dir", default=str(HERE / "data" / "plantdoc_train"))
    ap.add_argument("--out", default=None, help="Default: next to --model.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--strategy", default="B")
    ap.add_argument("--smoke", action="store_true", help="synthetic data under outputs/smoke (see smoke.py)")
    a = ap.parse_args(argv)

    if a.smoke:
        from smoke import ensure

        p = ensure("calibrate")
        a.model, a.revision, a.device = str(p.run), None, "cpu"
        a.manifest, a.extra_dir, a.out = str(p.pd_train / "manifest.csv"), str(p.pd_train), str(p.run)
    if not a.model:
        raise SystemExit("--model is required (or --smoke)")

    out = Path(a.out) if a.out else Path(a.model)
    out.mkdir(parents=True, exist_ok=True)
    images, labels = load_val(Path(a.manifest), Path(a.extra_dir))
    predictor = Predictor(a.model, revision=a.revision, device=a.device, temperature=1.0, crop_mask=False)
    names = predictor.label_names
    name_to_id = {n: i for i, n in enumerate(names)}
    y = np.array([name_to_id[l] for l in labels])

    logits_all = []
    for i in range(0, len(images), a.batch_size):
        logits_all.append(predictor.logits(images[i:i + a.batch_size]))
        print(f"[calib] logits {min(i + a.batch_size, len(images))}/{len(images)}")
    logits = np.concatenate(logits_all, axis=0)
    T = fit_temperature(logits, y)
    probs_before = softmax(logits)
    probs_after = softmax(logits / T)
    conf_b, pred_b = probs_before.max(axis=1), probs_before.argmax(axis=1)
    conf_a, pred_a = probs_after.max(axis=1), probs_after.argmax(axis=1)
    correct_b = pred_b == y
    correct_a = pred_a == y
    ece_b = ece(conf_b, correct_b)
    ece_a = ece(conf_a, correct_a)

    calib = {
        "temperature": T,
        "n": len(labels),
        "nll_before": nll(logits, y, 1.0),
        "nll_after": nll(logits, y, T),
        "ece_before": ece_b,
        "ece_after": ece_a,
        "split": "plantdoc_validation",
        "bins": 15,
    }
    (out / "calibration.json").write_text(json.dumps(calib, indent=2) + "\n")
    # Also copy next to the weights so Predictor picks T up automatically.
    if Path(a.model).is_dir() and Path(a.model).resolve() != out.resolve():
        (Path(a.model) / "calibration.json").write_text(json.dumps(calib, indent=2) + "\n")

    # Threshold sweep on the live chain (gate + strategy B crop of the uncropped photo) with T applied.
    from app_predictor import WirePredictor, diagnose_bytes
    from predict import get_predictor

    get_predictor.cache_clear()
    wire = WirePredictor(a.model, None if Path(a.model).exists() else a.revision, a.device)
    wire.model.temperature = T
    extra = Path(a.extra_dir)
    rows = [r for r in csv.DictReader(open(a.manifest, newline="", encoding="utf-8")) if r["split"] == "validation"]
    conf, correct, reached = [], [], []
    for r in rows:
        src = extra / (r.get("original_file") or r["file"])
        res = diagnose_bytes(src.read_bytes(), strategy=a.strategy, threshold=0.0, predictor=wire)
        if res["status"] == "retake_photo":
            conf.append(0.0)
            correct.append(False)
            reached.append(False)
        else:
            conf.append(float(res["confidence"] or 0.0))
            correct.append(res.get("disease") == r["label"])
            reached.append(True)
    table = threshold_sweep(conf, correct, reached, SWEEP)
    chosen, best = select_threshold(table, MIN_PRECISION)
    if chosen is None:
        status = "provisional"
        best_p = "n/a (no 'ok' answers)" if best["precision"] is None else f"{100 * best['precision']:.1f}%"
        note = (f"no threshold meets {MIN_PRECISION:.0%} precision. "
                f"Best available: precision {best_p} "
                f"at {best['threshold']:.2f} (coverage {frac(best['ok_n'], best['n'])}). "
                "Config threshold is left unchanged.")
        threshold = None
    else:
        status = "selected"
        note = (f"threshold {chosen['threshold']:.2f} has the highest pass rate "
                f"({frac(chosen['passes'], chosen['n'])}) among those with "
                f">={MIN_PRECISION:.0%} of 'ok' answers correct.")
        threshold = chosen["threshold"]

    thr_payload = {
        "status": status,
        "threshold": threshold,
        "note": note,
        "best_available": {k: best[k] for k in ("threshold", "precision", "pass_rate", "ok_n", "n", "coverage")},
        "chosen": None if chosen is None else {k: chosen[k] for k in ("threshold", "precision", "pass_rate", "ok_n", "n", "coverage")},
        "sweep": table,
        "min_precision": MIN_PRECISION,
        "strategy": a.strategy,
        "temperature": T,
    }
    (out / "threshold.json").write_text(json.dumps(thr_payload, indent=2) + "\n")
    if Path(a.model).is_dir():
        (Path(a.model) / "threshold.json").write_text(json.dumps(thr_payload, indent=2) + "\n")

    lines = [
        "# Calibration (PlantDoc validation)", "",
        f"Images: {len(labels)}. Temperature fitted by minimising NLL over log T (PlantDoc validation only).",
        "",
        f"- T = **{T:.3f}**",
        f"- ECE before: {100 * ece_b:.1f}%  after: {100 * ece_a:.1f}%  (15 bins)",
        "",
        "## Reliability after scaling", "",
        *reliability_md(reliability(conf_a, correct_a)),
        "",
        "## Threshold sweep (strategy B, T applied; gate rejections count as misses)", "",
        f"Rule: highest pass rate with at least {MIN_PRECISION:.0%} of 'ok' answers correct.",
        "",
        f"**{note}**",
        "",
        "| Thr | Passes | ok answers | Precision | Coverage |", "|---|---|---|---|---|",
    ]
    for row in table:
        p = "-" if row["precision"] is None else f"{100 * row['precision']:.1f}%"
        lines.append(f"| {row['threshold']:.2f} | {row['passes']}/{row['n']} | {row['ok_n']} | {p} | "
                     f"{frac(row['ok_n'], row['n'])} |")
    lines += ["", "Files: `calibration.json`, `threshold.json`."]
    (out / "calibration_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"Wrote {out}/calibration.json, threshold.json, calibration_report.md")


if __name__ == "__main__":
    main()
