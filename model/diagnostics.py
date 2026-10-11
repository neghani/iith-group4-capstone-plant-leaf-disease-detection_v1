"""Diagnostics on v1.0 using the PlantDoc validation split only. Never opens the final test.

    python model/diagnostics.py --model neghani/plantvillage-mobilenetv2 --revision v1.0

Writes model/outputs/diagnostics_v1_0.md and diagnostics_v1_0.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import Predictor, crop_from_class, load_image  # noqa: E402
from shared.vision.calibration import crop_id, crop_mask  # noqa: E402
from stats import absorbing_classes, confusion_pairs, frac, macro_f1_present  # noqa: E402

BINS = [0.5, 0.7, 0.85, 0.95]


def load_val_rows(manifest: Path) -> list[dict]:
    rows = [r for r in csv.DictReader(open(manifest, newline="", encoding="utf-8")) if r["split"] == "validation"]
    if not rows:
        raise SystemExit(f"No validation rows in {manifest}. Run prepare_plantdoc.py first.")
    return rows


def histogram_md(conf: np.ndarray, n_bins: int = 10) -> list[str]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(conf, bins=edges)
    lines = ["| Confidence | Wrong answers |", "|---|---|"]
    for i, c in enumerate(counts):
        lines.append(f"| {edges[i]:.1f}–{edges[i + 1]:.1f} | {c} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="neghani/plantvillage-mobilenetv2")
    ap.add_argument("--revision", default="v1.0")
    ap.add_argument("--manifest", default=str(HERE / "data" / "plantdoc_train" / "manifest.csv"))
    ap.add_argument("--extra_dir", default=str(HERE / "data" / "plantdoc_train"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()

    rows = load_val_rows(Path(a.manifest))
    extra = Path(a.extra_dir)
    images, labels = [], []
    for r in rows:
        path = extra / r["file"]
        if not path.exists():
            raise SystemExit(f"missing {path}")
        images.append(load_image(path))
        labels.append(r["label"])

    revision = None if Path(a.model).exists() else (a.revision or None)
    predictor = Predictor(a.model, revision=revision, device=a.device, temperature=1.0, crop_mask=False)
    names = predictor.label_names
    name_to_id = {n: i for i, n in enumerate(names)}
    unknown = sorted({l for l in labels if l not in name_to_id})
    if unknown:
        raise SystemExit(f"labels not in the model: {unknown}")

    logits_all, probs_all = [], []
    for i in range(0, len(images), a.batch_size):
        batch = images[i:i + a.batch_size]
        z = predictor.logits(batch)
        logits_all.append(z)
        p = predictor.probabilities(batch)
        probs_all.append(p)
        print(f"[diag] {min(i + a.batch_size, len(images))}/{len(images)}")
    logits = np.concatenate(logits_all, axis=0)
    probs = np.concatenate(probs_all, axis=0)
    pred_ids = probs.argmax(axis=1)
    pred = [names[i] for i in pred_ids]
    y = np.array([name_to_id[l] for l in labels])
    conf = probs.max(axis=1)
    correct = pred_ids == y
    wrong = ~correct
    n = len(labels)
    n_wrong = int(wrong.sum())

    shares = {t: float(((wrong) & (conf >= t)).sum() / n_wrong) if n_wrong else 0.0 for t in BINS}
    top1 = float(correct.mean())
    masked_ok = 0
    for i, lab in enumerate(labels):
        pm = crop_mask(probs[i], names, crop_id(lab))
        masked_ok += int(names[int(pm.argmax())] == lab)
    masked_top1 = masked_ok / n
    m_f1 = macro_f1_present(labels, pred)

    payload = {
        "model": a.model,
        "revision": a.revision,
        "n": n,
        "top1": top1,
        "crop_masked_top1": masked_top1,
        "plantdoc_macro_f1": m_f1,
        "wrong": n_wrong,
        "wrong_share_above": shares,
        "top_confusion": [{"true": t, "pred": p, "n": c} for t, p, c in confusion_pairs(labels, pred)],
        "absorbing": [{"pred": p, "n": c} for p, c in absorbing_classes(labels, pred)],
        "logits_path": "diagnostics_v1_0_logits.npz",
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "diagnostics_v1_0_logits.npz", logits=logits, labels=y, names=np.array(names))
    (out / "diagnostics_v1_0.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Diagnostics: v1.0 on PlantDoc validation", "",
        f"Model `{a.model}@{a.revision}`. Images: **{n}** (validation only; the final test was not opened).",
        "",
        "## 1. Confidence of wrong answers", "",
        f"{frac(n_wrong, n)} answers are wrong. Of those, the share whose confidence is at least:",
        "",
        "| Threshold | Wrong answers at or above | Share of wrong |", "|---|---|---|",
    ]
    for t in BINS:
        k = int(((wrong) & (conf >= t)).sum())
        lines.append(f"| {t:.2f} | {k}/{n_wrong} | {100 * shares[t]:.1f}% |")
    lines += ["", *histogram_md(conf[wrong] if n_wrong else np.array([0.0])), "",
              "## 2. Confusion summary", "",
              "Most common (true, predicted) pairs among mistakes:", "",
              "| True | Predicted | Count |", "|---|---|---|"]
    lines += [f"| `{t}` | `{p}` | {c} |" for t, p, c in confusion_pairs(labels, pred)]
    lines += ["", "Predicted classes that absorb the most mistakes:", "",
              "| Predicted | Mistakes absorbed |", "|---|---|"]
    lines += [f"| `{p}` | {c} |" for p, c in absorbing_classes(labels, pred)]
    lines += [
        "",
        "## 3. Crop-masked top-1", "",
        "Using the true crop from the label, other crops' classes are zeroed and the remaining",
        "logits are renormalised. This estimates what a required crop dropdown would give.",
        "",
        "| Metric | Value |", "|---|---|",
        f"| 38-way top-1 | {frac(int(correct.sum()), n)} |",
        f"| Crop-masked top-1 | {frac(masked_ok, n)} |",
        f"| Macro F1 (classes present) | {100 * m_f1:.1f}% |",
        "",
        "CROP_MASK stays **off** in the API until a later decision; this number is diagnostic only.",
        "",
        "Logits for calibration: `diagnostics_v1_0_logits.npz`.",
        "",
    ]
    (out / "diagnostics_v1_0.md").write_text("\n".join(lines))
    print("\n".join(lines[:20]))
    print(f"Wrote {out}/diagnostics_v1_0.md")


if __name__ == "__main__":
    main()
