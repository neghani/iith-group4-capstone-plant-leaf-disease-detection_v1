"""Phase 1: test the model on real-world images collected from the internet (proposal target: >= 50% correct).

Layout: model/data/real_world/<PlantVillage class name>/*.jpg  (see model/data/real_world/README.md)

Examples:
    python model/real_world_eval.py --model model/outputs/plantvillage-mobilenetv2
    python model/real_world_eval.py --model neghani/plantvillage-mobilenetv2 --revision v1.0 --push
    python model/real_world_eval.py --list-classes
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_HF_MODEL_ID,
    IMAGE_EXTS,
    Predictor,
    check_image_quality,
    load_image,
)

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "outputs" / "plantvillage-mobilenetv2"
DEFAULT_DATA_DIR = HERE / "data" / "real_world"
START, END = "<!-- REAL_WORLD_START -->", "<!-- REAL_WORLD_END -->"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="Local model dir or Hub id.")
    p.add_argument("--revision", default=None)
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--threshold", type=float, default=None, help="Confidence threshold for 'not sure' (default: from reports/metrics.json, else 0.6).")
    p.add_argument("--target", type=float, default=0.5, help="Pass mark for top-1 accuracy.")
    p.add_argument("--reports_dir", default=None, help="Default: <model>/reports (local) or model/outputs/reports.")
    p.add_argument("--card", default=None, help="Model card to update (default: <model>/README.md if present).")
    p.add_argument("--skip_quality_gate", action="store_true", help="Classify every image even when the quality check would reject it.")
    p.add_argument("--hub_model_id", default=DEFAULT_HF_MODEL_ID)
    p.add_argument("--push", action="store_true", help="Upload updated README.md and reports/real_world_*.* to the Hub.")
    p.add_argument("--list-classes", action="store_true", help="Print the model's class names and exit.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def default_threshold(reports_dir: Path) -> float:
    try:
        return float(json.loads((reports_dir / "metrics.json").read_text())["recommended_confidence_threshold"])
    except Exception:
        return 0.6


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    args = parse_args()
    model_is_local = Path(args.model).exists()
    predictor = Predictor(args.model, revision=args.revision, device=args.device)
    if args.list_classes:
        print("\n".join(predictor.label_names))
        return

    reports_dir = Path(args.reports_dir) if args.reports_dir else (Path(args.model) / "reports" if model_is_local else HERE / "outputs" / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    threshold = args.threshold if args.threshold is not None else default_threshold(reports_dir)

    data_dir = Path(args.data_dir)
    known = set(predictor.label_names)
    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir()) if data_dir.exists() else []
    rows: list[dict] = []
    skipped_dirs = []
    for d in class_dirs:
        if d.name not in known:
            skipped_dirs.append(d.name)
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            img = load_image(f)
            quality = check_image_quality(img)
            rejected = (not quality.ok) and not args.skip_quality_gate
            topk = predictor.predict_topk(img, k=3)
            rows.append(
                {
                    "file": str(f.relative_to(data_dir)),
                    "true": d.name,
                    "pred": topk.top1,
                    "confidence": round(topk.confidence, 4),
                    "top1_correct": topk.top1 == d.name,
                    "top3_correct": d.name in topk.labels,
                    "confident": topk.confidence >= threshold,
                    "quality_ok": quality.ok,
                    "quality_message": quality.message or "",
                    "quality_rejected": rejected,
                    "top3": " | ".join(f"{l} ({s:.2f})" for l, s in zip(topk.labels, topk.scores)),
                }
            )
            flag = "OK " if rows[-1]["top1_correct"] else "MISS"
            note = " [quality: " + quality.message + "]" if not quality.ok else ""
            print(f"{flag} {rows[-1]['file']:<55} -> {rows[-1]['top3']}{note}")

    for name in skipped_dirs:
        print(f"WARNING: folder '{name}' is not a model class; skipped. Run --list-classes for valid names.")
    if not rows:
        print(f"No images found under {data_dir}. See {data_dir / 'README.md'} for the expected layout.")
        return

    n = len(rows)
    top1 = sum(r["top1_correct"] for r in rows) / n
    top3 = sum(r["top3_correct"] for r in rows) / n
    confident = [r for r in rows if r["confident"]]
    top1_conf = (sum(r["top1_correct"] for r in confident) / len(confident)) if confident else float("nan")
    quality_rejects = sum(r["quality_rejected"] for r in rows)
    passed = top1 >= args.target

    per_class: dict[str, dict] = {}
    for r in rows:
        c = per_class.setdefault(r["true"], {"n": 0, "top1": 0, "top3": 0})
        c["n"] += 1
        c["top1"] += r["top1_correct"]
        c["top3"] += r["top3_correct"]

    summary = {
        "model": args.model,
        "revision": args.revision,
        "num_images": n,
        "num_classes_present": len(per_class),
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "threshold": threshold,
        "coverage_at_threshold": len(confident) / n,
        "top1_accuracy_on_confident": top1_conf,
        "quality_gate_rejections": quality_rejects,
        "target": args.target,
        "passed": passed,
        "per_class": {k: {**v, "top1_accuracy": v["top1"] / v["n"]} for k, v in per_class.items()},
    }
    (reports_dir / "real_world_results.json").write_text(json.dumps(summary, indent=2))
    with (reports_dir / "real_world_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Real-world results ===")
    print(f"images: {n}   top-1: {pct(top1)}   top-3: {pct(top3)}   target: {pct(args.target)}   -> {'PASS' if passed else 'FAIL'}")
    print(f"threshold {threshold}: coverage {pct(len(confident) / n)}, top-1 on confident {pct(top1_conf) if confident else 'n/a'}")
    print(f"quality gate would reject {quality_rejects} image(s)")

    # ---- update model card ----
    card_path = Path(args.card) if args.card else (Path(args.model) / "README.md" if model_is_local else reports_dir / "README.md")
    if card_path.exists():
        per_class_table = "| Class | Images | Top-1 | Top-3 |\n|---|---|---|---|\n" + "\n".join(
            f"| `{k}` | {v['n']} | {pct(v['top1'] / v['n'])} | {pct(v['top3'] / v['n'])} |" for k, v in sorted(per_class.items())
        )
        section = f"""{START}
## Real-world images

Internet photos of the same crops and diseases, outside PlantVillage (sources listed in the project
repository under `model/data/real_world/sources.csv`). Proposal target: at least 50% top-1.

| Metric | Value |
|---|---|
| Images | {n} ({len(per_class)} classes) |
| Top-1 accuracy | **{pct(top1)}** ({'meets' if passed else 'below'} the {pct(args.target)} target) |
| Top-3 accuracy | {pct(top3)} |
| Coverage at threshold {threshold} | {pct(len(confident) / n)} |
| Top-1 on confident predictions | {pct(top1_conf) if confident else 'n/a'} |
| Quality gate rejections | {quality_rejects} |

{per_class_table}

Per-image predictions: [`reports/real_world_predictions.csv`](reports/real_world_predictions.csv)
{END}"""
        text = card_path.read_text()
        new_text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: section, text, flags=re.S) if START in text else text + "\n\n" + section + "\n"
        card_path.write_text(new_text)
        print(f"Model card updated: {card_path}")
    else:
        print(f"No model card at {card_path}; run evaluate.py first to generate it.")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        for name in ("real_world_results.json", "real_world_predictions.csv"):
            api.upload_file(path_or_fileobj=str(reports_dir / name), path_in_repo=f"reports/{name}", repo_id=args.hub_model_id, commit_message="Add real-world results")
        if card_path.exists():
            api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=args.hub_model_id, commit_message="Model card: real-world results")
        print(f"Pushed to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
