"""Smoke mode: the whole stage-2 chain on synthetic data, on CPU, offline, in a couple of minutes.

    python -m model.train --smoke         prep + assertions, 2 training steps, diagnostics, calibration,
                                          PlantVillage eval, final test, publish dry run, artifact dry run
    python -m model.calibrate --smoke     calibration only (builds what it needs first)
    python -m model.final_test --smoke    final test only (builds what it needs first)

Everything is written under model/outputs/smoke/. No downloads, no Hugging Face token
(HF_HUB_OFFLINE=1, HF_TOKEN removed), and the real final-test guard is never read or written.
The "v1.0" baseline is a tiny randomly initialised ResNet-18, so the numbers mean nothing.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from plantdoc_map import MAP, SKIP_CLASS  # noqa: E402

ROOT = HERE / "outputs" / "smoke"
PV_PER_CLASS = 6
PD_TRAIN_PER_CLASS = 5
PD_TEST_PER_CLASS = 2
IMG = 96
STEPS = ["prep", "train", "diagnostics", "calibrate", "evaluate", "final", "publish", "artifacts"]


class P:
    """Smoke paths."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.data = root / "data"
        self.pv = self.data / "plantvillage"
        self.pd_src = self.data / "plantdoc_src"
        self.rw = self.data / "real_world_set"
        self.pd_train = self.data / "plantdoc_train"
        self.pd_test = self.data / "plantdoc_test"
        self.base = root / "base_model"
        self.run = root / "run"
        self.artifacts = root / "artifacts"


def label_names() -> list[str]:
    path = REPO_ROOT / "api" / "tests" / "model_labels.json"
    labels = json.loads(path.read_text())
    labels = labels if isinstance(labels, list) else list(labels.values())
    return sorted(labels)


def _image(rng: np.random.Generator, tint: tuple[int, int, int]) -> Image.Image:
    """Smooth random pattern with a class tint; distinct enough for pHash."""
    low = rng.integers(0, 255, (8, 8, 3)).astype(np.float64)
    low = 0.5 * low + 0.5 * np.array(tint, dtype=np.float64)
    return Image.fromarray(low.clip(0, 255).astype(np.uint8)).resize((IMG, IMG), Image.Resampling.BICUBIC)


def _tint(i: int) -> tuple[int, int, int]:
    return ((i * 37) % 200 + 30, (i * 91) % 200 + 30, (i * 53) % 200 + 30)


def _recompress(img: Image.Image) -> Image.Image:
    buf = BytesIO()
    img.resize((IMG - 8, IMG - 8)).save(buf, format="JPEG", quality=70)
    return Image.open(BytesIO(buf.getvalue())).convert("RGB")


def build_data(p: P, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    labels = label_names()
    for i, lab in enumerate(labels):
        d = p.pv / lab
        d.mkdir(parents=True, exist_ok=True)
        for k in range(PV_PER_CLASS):
            _image(rng, _tint(i)).save(d / f"pv_{k}.jpg", quality=92)

    for i, cls in enumerate(sorted(MAP)):
        for split, n in (("train", PD_TRAIN_PER_CLASS), ("test", PD_TEST_PER_CLASS)):
            d = p.pd_src / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                _image(rng, _tint(i + 50)).save(d / f"{split}_{k}.jpg", quality=92)
    (p.pd_src / "train" / SKIP_CLASS).mkdir(parents=True, exist_ok=True)
    for k in range(2):
        _image(rng, (120, 60, 60)).save(p.pd_src / "train" / SKIP_CLASS / f"mite_{k}.jpg")
    # A re-upload of a test image inside train: must be excluded as a near-duplicate.
    first = sorted(MAP)[0]
    _recompress(Image.open(p.pd_src / "test" / first / "test_0.jpg")).save(p.pd_src / "train" / first / "reupload.jpg")

    # real_world_set: 4 images taken from PlantDoc train/, listed in sources.csv by original_path.
    picks = sorted(MAP)[:4]
    rows, srcs = [], []
    for cls in picks:
        rel = f"images/{MAP[cls]}/rw_{len(rows):03d}.jpg"
        (p.rw / rel).parent.mkdir(parents=True, exist_ok=True)
        orig = f"train/{cls}/train_0.jpg"
        shutil.copy2(p.pd_src / orig, p.rw / rel)
        rows.append({"file": rel, "label": MAP[cls], "plantdoc_class": cls, "split": "final"})
        srcs.append({"file": rel, "source": "Cropped-PlantDoc", "original_path": orig,
                     "original_class": cls, "license": "CC BY 4.0", "url": "smoke"})
    for name, data in (("labels.csv", rows), ("sources.csv", srcs)):
        with open(p.rw / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)


def build_base_model(p: P, seed: int = 0) -> None:
    import torch
    from transformers import ResNetConfig, ResNetForImageClassification, ConvNextImageProcessor

    torch.manual_seed(seed)
    labels = label_names()
    cfg = ResNetConfig(
        num_labels=len(labels),
        depths=[1, 1, 1, 1],
        hidden_sizes=[16, 32, 64, 128],
        id2label={i: n for i, n in enumerate(labels)},
        label2id={n: i for i, n in enumerate(labels)},
    )
    ResNetForImageClassification(cfg).save_pretrained(p.base)
    ConvNextImageProcessor(size={"shortest_edge": 72}, crop_size={"height": 64, "width": 64}).save_pretrained(p.base)


def smoke_env() -> dict:
    env = dict(os.environ)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(k, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               HF_DATASETS_OFFLINE="1", CUDA_VISIBLE_DEVICES="", PYTORCH_ENABLE_MPS_FALLBACK="1",
               MODEL_SMOKE="1", CROP_MASK="", CALIBRATION_TEMPERATURE="")
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def run_module(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", f"model.{module}", *map(str, args)]
    print(f"\n[smoke] $ {' '.join(cmd[2:])}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=smoke_env(), check=True)


def done(step: str, p: P) -> bool:
    marker = {
        "prep": p.pd_train / "items.json",
        "train": p.run / "model.safetensors",
        "diagnostics": p.artifacts / "diagnostics_v1_0.json",
        "calibrate": p.run / "threshold.json",
        "evaluate": p.run / "reports" / "metrics.json",
        "final": p.artifacts / "final_report_v1_1.md",
    }.get(step)
    return bool(marker and marker.exists())


def run_step(step: str, p: P) -> None:
    if step == "prep":
        if not (p.pv.exists() and p.pd_src.exists()):
            build_data(p)
        if not (p.base / "config.json").exists():
            build_base_model(p)
        run_module("prepare_plantdoc", "--src", p.pd_src, "--no_clone", "--out_train", p.pd_train,
                   "--out_test", p.pd_test, "--exclude_set", p.rw, "--artifacts_dir", p.artifacts)
    elif step == "train":
        run_module(
            "train", "--init_from", p.base, "--data_dir", p.pv, "--extra_data", p.pd_train,
            "--artifacts_dir", p.artifacts, "--plantdoc_fraction", "0.5", "--epoch_samples", "16",
            "--output_dir", p.run, "--do_train", "--do_eval", "--max_steps", "2", "--num_train_epochs", "1",
            "--per_device_train_batch_size", "8", "--per_device_eval_batch_size", "32",
            "--learning_rate", "1e-4", "--weight_decay", "0.01", "--label_smoothing_factor", "0.1",
            "--lr_scheduler_type", "cosine", "--eval_strategy", "epoch", "--save_strategy", "epoch",
            "--save_total_limit", "1", "--load_best_model_at_end", "--metric_for_best_model", "plantdoc_macro_f1",
            "--greater_is_better", "True", "--logging_steps", "1", "--report_to", "none", "--seed", "42",
            "--use_cpu", "--dataloader_num_workers", "0",
        )
    elif step == "diagnostics":
        run_module("diagnostics", "--model", p.base, "--revision", "", "--manifest", p.pd_train / "manifest.csv",
                   "--extra_dir", p.pd_train, "--out", p.artifacts, "--device", "cpu")
    elif step == "calibrate":
        run_module("calibrate", "--smoke")
    elif step == "evaluate":
        run_module("evaluate", "--model", p.run, "--data_dir", p.pv, "--device", "cpu", "--card_revision", "v1.1")
    elif step == "final":
        run_module("final_test", "--smoke")
    elif step == "publish":
        run_module("publish", "--model", p.run, "--tag", "v1.1", "--dry_run",
                   "--diagnostics", p.artifacts / "diagnostics_v1_0.json")
    elif step == "artifacts":
        run_module("artifacts", "--step", "smoke", "--run", "smoke", "--dry_run",
                   p.artifacts, p.run / "run_info.json", p.run / "calibration.json",
                   p.run / "threshold.json", p.run / "reports")
    else:
        raise ValueError(step)


def ensure(until: str, p: P | None = None) -> P:
    """Run every step before `until` that has not produced its output yet."""
    p = p or P()
    for step in STEPS[: STEPS.index(until)]:
        if not done(step, p):
            run_step(step, p)
    return p


def run_all(fresh: bool = True) -> P:
    p = P()
    if fresh and p.root.exists():
        shutil.rmtree(p.root)
    t0 = time.time()
    for step in STEPS:
        run_step(step, p)
    print(f"\n[smoke] all {len(STEPS)} steps passed in {time.time() - t0:.0f}s. Outputs: {p.root}")
    return p


def assert_smoke_paths(*paths: Path) -> None:
    """Refuse to write outside outputs/smoke while in smoke mode."""
    for path in paths:
        resolved = Path(path).resolve()
        if ROOT.resolve() not in (resolved, *resolved.parents):
            raise SystemExit(f"smoke mode may only write under {ROOT}, got {resolved}")


if __name__ == "__main__":
    run_all()
