"""Shared helpers for the Phase 1 model scripts (train, evaluate, real-world test, predict).

Keep this file dependency-light: it is imported by `predict.py`, which is the reference
implementation Phase 5 copies into the API service.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------------------------------------------
# Hugging Face identifiers
# --------------------------------------------------------------------------------------

HF_USERNAME = "neghani"
DEFAULT_HF_MODEL_ID = os.environ.get("HF_MODEL_ID", f"{HF_USERNAME}/plantvillage-mobilenetv2")
DEFAULT_HF_REVISION = os.environ.get("HF_MODEL_REVISION", "main")
DEFAULT_BASE_MODEL = "google/mobilenet_v2_1.0_224"
FALLBACK_BASE_MODEL = "microsoft/resnet-50"
KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# --------------------------------------------------------------------------------------
# Image quality gate (same thresholds as api/app/vision/quality.py)
# --------------------------------------------------------------------------------------

MIN_WIDTH = 128
MIN_HEIGHT = 128
MAX_DIMENSION = 4096
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 230.0
MIN_BLUR_VARIANCE = 80.0


@dataclass
class QualityResult:
    ok: bool
    message: str | None = None
    brightness: float | None = None
    blur_score: float | None = None


def laplacian_variance(gray: np.ndarray) -> float:
    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    lap = (
        -4 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(lap.var())


def check_image_quality(img: Image.Image) -> QualityResult:
    """Return a farmer-facing message when the photo should be retaken."""
    w, h = img.size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return QualityResult(False, "Image is too small. Move closer so the leaf fills more of the frame.")
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        return QualityResult(False, "Image dimensions are too large.")

    gray = np.array(img.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    blur_score = laplacian_variance(gray)

    if brightness < MIN_BRIGHTNESS:
        return QualityResult(False, "Photo is too dark. Retake in better light without strong shadows.", brightness, blur_score)
    if brightness > MAX_BRIGHTNESS:
        return QualityResult(False, "Photo is overexposed. Avoid direct glare on the leaf.", brightness, blur_score)
    if blur_score < MIN_BLUR_VARIANCE:
        return QualityResult(False, "Photo looks blurry. Hold steady and focus on the leaf surface.", brightness, blur_score)
    return QualityResult(True, None, brightness, blur_score)


def load_image(path: str | Path) -> Image.Image:
    """Open an image, apply EXIF orientation (phones store rotation there) and force RGB."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


# --------------------------------------------------------------------------------------
# Class names
# --------------------------------------------------------------------------------------


def crop_from_class(class_name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato'; 'Corn_(maize)___Common_rust_' -> 'Corn (maize)'."""
    crop = class_name.split("___")[0]
    return crop.replace("_", " ").strip()


def display_name(class_name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato - Early blight'."""
    parts = class_name.split("___")
    crop = parts[0].replace("_", " ").strip()
    if len(parts) == 1:
        return crop
    cond = parts[1].replace("_", " ").strip()
    return f"{crop} - {cond}"


# --------------------------------------------------------------------------------------
# Dataset scanning and splitting (imagefolder layout: <root>/<class>/<image>)
# --------------------------------------------------------------------------------------


def _has_class_folders(root: Path) -> bool:
    for child in root.iterdir():
        if child.is_dir():
            for f in child.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    return True
    return False


def resolve_data_dir(path: str | Path) -> Path:
    """Accept the kagglehub download root or any parent and find the `color` imagefolder."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    if _has_class_folders(root):
        return root
    for candidate in ("color", "plantvillage dataset/color", "PlantVillage/color"):
        c = root / candidate
        if c.exists() and _has_class_folders(c):
            return c
    for c in sorted(root.rglob("color")):
        if c.is_dir() and _has_class_folders(c):
            return c
    raise FileNotFoundError(
        f"No <class>/<image> folders found under {root}. Point --data_dir at the PlantVillage 'color' folder."
    )


def scan_imagefolder(
    root: Path,
    crop_filter: str | None = None,
    max_samples_per_class: int | None = None,
    seed: int = 42,
) -> tuple[list[str], list[tuple[str, int]]]:
    """Return (label_names, [(relative_path, label_id), ...])."""
    crops = None
    if crop_filter:
        crops = {c.strip().lower() for c in crop_filter.split(",") if c.strip()}

    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if crops:
        class_dirs = [d for d in class_dirs if crop_from_class(d.name).lower() in crops or d.name.split("___")[0].lower() in crops]
    if not class_dirs:
        raise ValueError(f"No class folders matched crop_filter={crop_filter!r} under {root}")

    label_names = [d.name for d in class_dirs]
    rng = random.Random(seed)
    items: list[tuple[str, int]] = []
    for label_id, d in enumerate(class_dirs):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if max_samples_per_class is not None and len(files) > max_samples_per_class:
            rng.shuffle(files)
            files = sorted(files[:max_samples_per_class])
        items.extend((str(f.relative_to(root)), label_id) for f in files)
    return label_names, items


def stratified_split(
    items: list[tuple[str, int]],
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
) -> dict[str, list[tuple[str, int]]]:
    """Deterministic per-class split. Every class keeps at least one image per non-empty split."""
    by_label: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for item in items:
        by_label[item[1]].append(item)

    rng = random.Random(seed)
    out: dict[str, list[tuple[str, int]]] = {"train": [], "validation": [], "test": []}
    for label in sorted(by_label):
        group = sorted(by_label[label])
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * test_fraction)) if test_fraction > 0 else 0
        n_val = int(round(n * val_fraction)) if val_fraction > 0 else 0
        if test_fraction > 0 and n_test == 0 and n >= 3:
            n_test = 1
        if val_fraction > 0 and n_val == 0 and n >= 3:
            n_val = 1
        n_train = n - n_val - n_test
        if n_train <= 0:
            n_train, n_val, n_test = n, 0, 0
        out["train"].extend(group[:n_train])
        out["validation"].extend(group[n_train : n_train + n_val])
        out["test"].extend(group[n_train + n_val :])
    return out


def save_splits(path: Path, data_dir: Path, label_names: list[str], splits: dict[str, list[tuple[str, int]]], extra: dict | None = None) -> None:
    payload = {
        "data_dir": str(data_dir),
        "label_names": label_names,
        "counts": {k: len(v) for k, v in splits.items()},
        **(extra or {}),
        "splits": {k: [[p, l] for p, l in v] for k, v in splits.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_splits(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    data["splits"] = {k: [(p, int(l)) for p, l in v] for k, v in data["splits"].items()}
    return data


# --------------------------------------------------------------------------------------
# Inference wrapper used by evaluate.py, real_world_eval.py and predict.py
# --------------------------------------------------------------------------------------


@dataclass
class TopK:
    labels: list[str]
    scores: list[float]

    @property
    def top1(self) -> str:
        return self.labels[0]

    @property
    def confidence(self) -> float:
        return self.scores[0]


class Predictor:
    """Loads a classifier from a local dir or the Hub and returns softmax top-k per image.

    Uses the model's own AutoImageProcessor (resize + center-crop + normalize), i.e. exactly
    what `transformers.pipeline("image-classification")` does, so Phase 5 can use either.
    """

    def __init__(self, model_id: str = DEFAULT_HF_MODEL_ID, revision: str | None = None, device: str | None = None, token: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        kwargs = {"revision": revision} if revision else {}
        if token:
            kwargs["token"] = token
        self.model_id = model_id
        self.revision = revision
        self.processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForImageClassification.from_pretrained(model_id, **kwargs)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.label_names = [self.id2label[i] for i in range(len(self.id2label))]

    def probabilities(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        inputs = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()

    def predict_topk(self, image: Image.Image, k: int = 3) -> TopK:
        probs = self.probabilities([image])[0]
        order = np.argsort(-probs)[:k]
        return TopK(labels=[self.id2label[int(i)] for i in order], scores=[float(probs[i]) for i in order])
