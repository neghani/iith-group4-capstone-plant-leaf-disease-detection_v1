"""Phase 1: evaluate a trained classifier on the held-out PlantVillage test split and build the model card.

Writes to <reports_dir>:
    metrics.json                 accuracy, macro/weighted precision-recall-F1, top-3, per-class metrics
    classification_report.txt    sklearn per-class report
    confusion_matrix.png         normalised confusion matrix
    threshold_sweep.csv          coverage / accuracy per confidence threshold + recommendation
and fills model_card_template.md into README.md next to the model (or reports_dir for Hub models).

Examples:
    python model/evaluate.py --model model/outputs/plantvillage-resnet18
    python model/evaluate.py --model model/outputs/plantvillage-resnet18 --push
    python model/evaluate.py --model neghani/plantvillage-resnet18 --revision v1.0 --data_dir "$DATA_DIR"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from card import assemble_card  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_HF_MODEL_ID,
    DEFAULT_HF_MODEL_REPO,
    Predictor,
    crop_from_class,
    hf_api,
    load_image,
    load_splits,
    resolve_data_dir,
)

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "outputs" / "plantvillage-resnet18"
THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="Local model dir or Hub id.")
    p.add_argument("--revision", default=None, help="Hub revision when --model is a Hub id.")
    p.add_argument("--splits", default=None, help="splits.json path (default: <model>/splits.json, or downloaded from the Hub).")
    p.add_argument("--data_dir", default=None, help="Override the image root recorded in splits.json (needed on a new machine).")
    p.add_argument("--split", default="test", choices=["test", "validation", "train"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_samples", type=int, default=None, help="Evaluate only the first N images (smoke tests).")
    p.add_argument("--reports_dir", default=None, help="Default: <model>/reports (local) or model/outputs/reports (Hub).")
    p.add_argument("--card_out", default=None, help="Where to write the filled model card (default: <model>/README.md).")
    p.add_argument("--template", default=str(HERE / "model_card_template.md"))
    p.add_argument("--card_revision", default="v1.0", help="Revision tag shown in the model card usage snippet.")
    p.add_argument("--target_precision", type=float, default=0.97, help="Accuracy-on-covered target used to recommend the threshold.")
    p.add_argument("--hub_model_id", default=DEFAULT_HF_MODEL_REPO, help="Hub repo id written into the model card (defaults to neghani/plantvillage-resnet18).")
    p.add_argument("--push", action="store_true", help="Upload README.md and reports/ to --hub_model_id.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def locate_splits(args: argparse.Namespace, model_is_local: bool) -> Path:
    if args.splits:
        return Path(args.splits)
    if model_is_local:
        candidate = Path(args.model) / "splits.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{candidate} not found; pass --splits.")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=args.model, filename="splits.json", revision=args.revision))


def try_load_run_info(args: argparse.Namespace, model_is_local: bool) -> dict:
    try:
        if model_is_local:
            p = Path(args.model) / "run_info.json"
            return json.loads(p.read_text()) if p.exists() else {}
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(repo_id=args.model, filename="run_info.json", revision=args.revision)).read_text())
    except Exception:
        return {}


def run_inference(predictor: Predictor, data_dir: Path, rows: list[tuple[str, int]], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    probs_all: list[np.ndarray] = []
    labels: list[int] = []
    total = len(rows)
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        images = [load_image(data_dir / p) for p, _ in batch]
        probs_all.append(predictor.probabilities(images))
        labels.extend(l for _, l in batch)
        done = min(start + batch_size, total)
        if done == total or (start // batch_size) % 10 == 0:
            print(f"  evaluated {done}/{total}", flush=True)
    return np.concatenate(probs_all, axis=0), np.asarray(labels)


def threshold_sweep(probs: np.ndarray, labels: np.ndarray, target_precision: float) -> tuple[list[dict], float, str]:
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = preds == labels
    rows = []
    for t in THRESHOLDS:
        covered = conf >= t
        n_cov = int(covered.sum())
        acc_cov = float(correct[covered].mean()) if n_cov else float("nan")
        rows.append(
            {
                "threshold": t,
                "coverage": float(n_cov / len(conf)),
                "accuracy_on_covered": acc_cov,
                "accuracy_counting_abstain_wrong": float((correct & covered).mean()),
                "n_covered": n_cov,
            }
        )
    candidates = [r for r in rows if r["coverage"] >= 0.5 and r["n_covered"] and r["accuracy_on_covered"] >= target_precision]
    if candidates:
        rec = candidates[0]["threshold"]
        rule = f"lowest threshold whose accuracy on covered images is at least {target_precision:.0%} with coverage of at least 50%"
    else:
        fallback = [r for r in rows if r["coverage"] >= 0.5]
        rec = fallback[-1]["threshold"] if fallback else 0.6
        rule = f"no threshold reached {target_precision:.0%} accuracy on covered images; highest threshold keeping 50% coverage"
    return rows, rec, rule


def plot_confusion_matrix(cm: np.ndarray, label_names: list[str], out: Path) -> None:
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed; skipping confusion_matrix.png")
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    n = len(label_names)
    size = max(6, min(22, 0.45 * n + 3))
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=90, fontsize=7)
    ax.set_yticklabels(label_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    if n <= 20:
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def fill_template(template: str, values: dict[str, object]) -> str:
    out = template
    for key, val in values.items():
        out = out.replace("{{" + key + "}}", str(val))
    return out


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    model_is_local = model_path.exists()
    if model_is_local and model_path.is_dir() and not (model_path / "config.json").is_file():
        raise SystemExit(
            f"{model_path} exists but is not a trained model (no config.json). "
            "Section 5 skips training when RUN_STAGE1 is False and still points OUTPUT_DIR here. "
            "Skip section 6, or set RUN_STAGE1 = True and train first. "
            "To score the published model: --model neghani/plantvillage-resnet18 --revision v1.0"
        )
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

    splits_path = locate_splits(args, model_is_local)
    splits = load_splits(splits_path)
    label_names: list[str] = splits["label_names"]
    data_dir = resolve_data_dir(args.data_dir or splits["data_dir"])
    rows = splits["splits"][args.split]
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"Model: {args.model}  split: {args.split}  images: {len(rows)}  data_dir: {data_dir}")

    predictor = Predictor(args.model, revision=args.revision, device=args.device)
    if predictor.label_names != label_names:
        raise SystemExit("Model id2label does not match splits.json label_names; wrong splits file?")

    probs, labels = run_inference(predictor, data_dir, rows, args.batch_size)
    preds = probs.argmax(axis=1)
    n_classes = len(label_names)
    k = min(3, n_classes)
    topk = np.argsort(-probs, axis=1)[:, :k]
    top3 = float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))
    accuracy = float((preds == labels).mean())
    all_ids = list(range(n_classes))
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(labels, preds, labels=all_ids, average="macro", zero_division=0)
    w_p, w_r, w_f, _ = precision_recall_fscore_support(labels, preds, labels=all_ids, average="weighted", zero_division=0)
    per_p, per_r, per_f, support = precision_recall_fscore_support(labels, preds, labels=all_ids, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=all_ids)
    report_txt = classification_report(labels, preds, labels=all_ids, target_names=label_names, zero_division=0, digits=4)
    sweep, recommended, rule = threshold_sweep(probs, labels, args.target_precision)

    reports_dir = Path(args.reports_dir) if args.reports_dir else (Path(args.model) / "reports" if model_is_local else HERE / "outputs" / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    per_class = [
        {"class": name, "precision": float(per_p[i]), "recall": float(per_r[i]), "f1": float(per_f[i]), "support": int(support[i])}
        for i, name in enumerate(label_names)
    ]
    metrics = {
        "model": args.model,
        "revision": args.revision,
        "split": args.split,
        "num_images": int(len(labels)),
        "num_classes": n_classes,
        "accuracy": accuracy,
        "top3_accuracy": top3,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f),
        "weighted_precision": float(w_p),
        "weighted_recall": float(w_r),
        "weighted_f1": float(w_f),
        "recommended_confidence_threshold": recommended,
        "threshold_rule": rule,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "label_names": label_names,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports_dir / "classification_report.txt").write_text(report_txt)
    with (reports_dir / "threshold_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
    plot_confusion_matrix(cm, label_names, reports_dir / "confusion_matrix.png")

    print("\n" + report_txt)
    print(f"accuracy={pct(accuracy)}  macro_f1={pct(macro_f)}  top3={pct(top3)}  recommended_threshold={recommended}")

    # ---- model card ----
    template_path = Path(args.template)
    if template_path.exists():
        run_info = try_load_run_info(args, model_is_local)
        counts = splits.get("counts", {})
        crops = sorted({crop_from_class(n) for n in label_names})
        per_class_table = "| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n" + "\n".join(
            f"| `{r['class']}` | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} | {r['support']} |" for r in per_class
        )
        threshold_table = "| Threshold | Coverage | Accuracy on covered | Accuracy (abstain = wrong) |\n|---|---|---|---|\n" + "\n".join(
            f"| {r['threshold']:.2f} | {pct(r['coverage'])} | {pct(r['accuracy_on_covered']) if r['n_covered'] else 'n/a'} | {pct(r['accuracy_counting_abstain_wrong'])} |"
            for r in sweep
        )
        if run_info:
            details = "\n".join(
                f"- {k}: `{v}`" for k, v in run_info.items() if k not in ("label_names", "split_counts") and v is not None
            )
        else:
            details = "- see `run_info.json` / `training_args.bin` in this repository"
        hub_id = args.hub_model_id if model_is_local else (args.hub_model_id if args.push else args.model)
        values = {
            "model_id": hub_id,
            "model_name": hub_id.split("/")[-1],
            "base_model": run_info.get("base_model") or splits.get("base_model") or predictor.model.config._name_or_path,
            "revision": args.card_revision,
            "num_classes": n_classes,
            "crop_scope": ", ".join(crops) if len(crops) <= 8 else f"{len(crops)} crops",
            "date": dt.date.today().isoformat(),
            "seed": splits.get("seed", "n/a"),
            "num_train": counts.get("train", "n/a"),
            "num_val": counts.get("validation", "n/a"),
            "num_test": counts.get("test", len(labels)),
            "accuracy": round(accuracy, 4),
            "macro_f1": round(float(macro_f), 4),
            "top3_accuracy": round(top3, 4),
            "accuracy_pct": pct(accuracy),
            "macro_precision_pct": pct(macro_p),
            "macro_recall_pct": pct(macro_r),
            "macro_f1_pct": pct(macro_f),
            "top3_accuracy_pct": pct(top3),
            "recommended_threshold": recommended,
            "threshold_rule": rule,
            "per_class_table": per_class_table,
            "threshold_table": threshold_table,
            "training_details": details,
        }
        card = fill_template(template_path.read_text(), values)
        card_out = Path(args.card_out) if args.card_out else (Path(args.model) / "README.md" if model_is_local else reports_dir / "README.md")
        card_out.write_text(card)
        if model_is_local:
            status = assemble_card(Path(args.model), outputs=HERE / "outputs", latency_path=HERE / "latency.json")
            print(f"Model card sections filled: {', '.join(status['filled']) or '(none)'}")
            if status["missing"]:
                print(f"Model card sections missing: {', '.join(status['missing'])}")
        print(f"Model card written to {card_out}")
    else:
        card_out = None
        print(f"Template {template_path} not found; skipping model card.")

    print(f"Reports written to {reports_dir}")

    if args.push:
        api = hf_api()
        api.create_repo(args.hub_model_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(reports_dir), path_in_repo="reports", repo_id=args.hub_model_id, commit_message="Add evaluation reports")
        if card_out is not None:
            api.upload_file(path_or_fileobj=str(card_out), path_in_repo="README.md", repo_id=args.hub_model_id, commit_message="Update model card")
        print(f"Pushed reports and model card to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
