"""Score a MasterDataset split with the serving processor (no Trainer).

Fits temperature and a confidence threshold on validation only. Scores the test
split once through the same processor the API / Space will use.

    python model/score_split.py --model-dir model/outputs/masterdataset-resnet18 \\
        --data-dir data/masterdataset --split validation --calibrate
    python model/score_split.py --model-dir model/outputs/masterdataset-resnet18 \\
        --data-dir data/masterdataset --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import align_processor  # noqa: E402
from stats import ece, fit_temperature, select_threshold, softmax, threshold_sweep  # noqa: E402

SPLIT_DIRS = {"train": "train", "validation": "val", "test": "test"}


class FolderSet(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int]):
        self.samples: list[tuple[Path, int]] = []
        for name, idx in sorted(class_to_idx.items(), key=lambda kv: kv[1]):
            folder = root / name
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    self.samples.append((path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        return Image.open(path).convert("RGB"), label


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load(model_dir: Path):
    processor = AutoImageProcessor.from_pretrained(model_dir)
    align_processor(processor)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    device = _device()
    model.to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    class_to_idx = {id2label[i]: i for i in sorted(id2label)}
    return processor, model, device, class_to_idx, id2label


@torch.inference_mode()
def collect_logits(model_dir: Path, data_dir: Path, split: str, batch_size: int = 64):
    processor, model, device, class_to_idx, id2label = _load(model_dir)
    folder = data_dir / SPLIT_DIRS[split]
    if not folder.is_dir():
        # accept validation/ or valid/
        for alias in ("validation", "valid", "val"):
            candidate = data_dir / alias
            if split == "validation" and candidate.is_dir():
                folder = candidate
                break
        else:
            raise SystemExit(f"Split folder missing under {data_dir} for {split}")
    dataset = FolderSet(folder, class_to_idx)
    if not dataset:
        raise SystemExit(f"No images in {folder}")

    def collate(batch):
        images, labels = zip(*batch)
        enc = processor(list(images), return_tensors="pt")
        return enc["pixel_values"], torch.tensor(labels)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=collate)
    logits, labels = [], []
    for i, (pixels, y) in enumerate(loader):
        out = model(pixel_values=pixels.to(device)).logits
        logits.append(out.float().cpu().numpy())
        labels.append(y.numpy())
        if i % 40 == 0:
            print(f"[{split}] batch {i}/{len(loader)}", flush=True)
    return (
        np.concatenate(logits),
        np.concatenate(labels),
        id2label,
        processor,
    )


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> dict:
    probs = softmax(logits / temperature)
    pred = probs.argmax(axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    correct = pred == labels
    loss = float(np.mean([-np.log(max(probs[i, labels[i]], 1e-12)) for i in range(len(labels))]))
    return {
        "n": int(len(labels)),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "top3_accuracy": float(np.mean([labels[i] in top3[i] for i in range(len(labels))])),
        "loss": loss,
    }


def calibrate(logits: np.ndarray, labels: np.ndarray) -> dict:
    temperature = fit_temperature(logits, labels)
    probs = softmax(logits / temperature)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == labels
    raw = softmax(logits)
    raw_conf = raw.max(axis=1)
    raw_correct = raw.argmax(axis=1) == labels
    table = threshold_sweep(conf, correct, np.ones(len(conf), dtype=bool))
    chosen, best = select_threshold(table)
    return {
        "split": "validation",
        "n": int(len(labels)),
        "temperature": temperature,
        "accuracy": float(correct.mean()),
        "ece_before": ece(raw_conf, raw_correct),
        "ece_after": ece(conf, correct),
        "threshold": None if chosen is None else chosen["threshold"],
        "threshold_row": chosen,
        "best_precision_row": best,
        "sweep": table,
        "processor": "aligned_serving",
    }


def write_processor(model_dir: Path, processor) -> Path:
    """Persist the aligned processor next to the weights (for Hub v1.0.1)."""
    out = model_dir / "preprocessor_config.json"
    processor.save_pretrained(model_dir)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--calibrate", action="store_true", help="Fit T and threshold (validation only).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--write-processor", action="store_true", help="Save aligned preprocessor_config.json.")
    p.add_argument("--temperature", type=float, default=1.0, help="Used when scoring (not calibrating).")
    args = p.parse_args(argv)

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    if args.calibrate and args.split != "validation":
        raise SystemExit("--calibrate requires --split validation")

    logits, labels, _id2label, processor = collect_logits(
        model_dir, data_dir, args.split, batch_size=args.batch_size
    )
    if args.write_processor:
        path = write_processor(model_dir, processor)
        print(f"Wrote aligned processor to {path}", flush=True)

    if args.calibrate:
        summary = calibrate(logits, labels)
        out = model_dir / "calibration.json"
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: summary[k] for k in (
            "n", "temperature", "accuracy", "ece_before", "ece_after", "threshold"
        )}, indent=2), flush=True)
        print(f"Wrote {out}", flush=True)
        return

    temperature = args.temperature
    if temperature == 1.0 and (model_dir / "calibration.json").is_file():
        temperature = float(json.loads((model_dir / "calibration.json").read_text())["temperature"])
    metrics = metrics_from_logits(logits, labels, temperature=temperature)
    if args.split == "test":
        payload = {
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_top3_accuracy": metrics["top3_accuracy"],
            "test_loss": metrics["loss"],
            "n": metrics["n"],
            "temperature": temperature,
            "processor": "aligned_serving",
            "note": (
                "Test scored once with the serving processor after the Trainer "
                "eval-cache bug and crop_pct mismatch were fixed. The split was "
                "also viewed during diagnosis."
            ),
        }
        out = model_dir / "test_results.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)
        print(f"Wrote {out}", flush=True)
    else:
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
