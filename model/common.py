"""Shared helpers for the Phase 1 model scripts (train, evaluate, real-world test, predict).

Keep this file dependency-light: it is imported by `predict.py`, which is the reference
implementation Phase 5 copies into the API service.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# CROP_MASK is off until diagnostics.py shows crop-masked top-1 is worth a required dropdown.
CROP_MASK = os.environ.get("CROP_MASK", "").lower() in ("1", "true", "yes")

# --------------------------------------------------------------------------------------
# Environment / Hugging Face token
# --------------------------------------------------------------------------------------


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from the repo-root .env into os.environ (never overrides real env vars)."""
    candidates = [Path(path)] if path else [Path(__file__).resolve().parent.parent / ".env", Path.cwd() / ".env"]
    for env_file in candidates:
        if not env_file.is_file():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip("'\"")
            if value:
                os.environ.setdefault(key.strip(), value)


load_dotenv()  # before the DEFAULT_* constants below so HF_MODEL_ID in .env is honoured


def get_hf_token() -> str | None:
    """HF token from, in order: HF_TOKEN env / .env, Colab secret HF_TOKEN, cached `hf auth login`."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None
    if token:
        os.environ["HF_TOKEN"] = token  # so train.py / Trainer / subprocesses see it too
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def hf_api():
    """HfApi bound to the token above. Raises a clear error if there is none."""
    from huggingface_hub import HfApi

    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token. Set HF_TOKEN in .env (local), add the HF_TOKEN secret in Colab, "
            "or run `hf auth login`. It must be a WRITE token."
        )
    return HfApi(token=token)


# --------------------------------------------------------------------------------------
# Hugging Face identifiers
# --------------------------------------------------------------------------------------

HF_USERNAME = "neghani"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from shared.release import (  # noqa: E402
    CALIBRATION_TEMPERATURE as RELEASE_TEMPERATURE,
    HF_MODEL_ID as RELEASE_MODEL_ID,
    HF_MODEL_REVISION as RELEASE_REVISION,
)

DEFAULT_HF_MODEL_ID = os.environ.get("HF_MODEL_ID", RELEASE_MODEL_ID)
DEFAULT_HF_REVISION = os.environ.get("HF_MODEL_REVISION", RELEASE_REVISION)
DEFAULT_BASE_MODEL = "google/mobilenet_v2_1.0_224"
FALLBACK_BASE_MODEL = "microsoft/resnet-50"
KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# --------------------------------------------------------------------------------------
# Image quality gate: the API's leaf-based, scale-normalised gate from shared/vision/quality.py
# (docs/challenges-and-approach.md #1-3). The Phase 1 whole-frame Laplacian gate was removed
# because it rejected large clean photos as blurry.
# --------------------------------------------------------------------------------------


@dataclass
class QualityResult:
    ok: bool
    message: str | None = None
    brightness: float | None = None
    blur_score: float | None = None
    reason: str | None = None


def check_image_quality(img: Image.Image) -> QualityResult:
    """Return a farmer-facing message when the photo should be retaken."""
    from shared.vision.candidates import find_candidates
    from shared.vision.config import WORK_MAX_SIDE
    from shared.vision.quality import check_gate

    size = img.size
    if max(size) > WORK_MAX_SIDE:
        img = img.copy()
        img.thumbnail((WORK_MAX_SIDE, WORK_MAX_SIDE), Image.Resampling.LANCZOS)
    gate = check_gate(img, find_candidates(img), size)
    return QualityResult(gate.ok, gate.message, gate.brightness, gate.blur_score, gate.reason)


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


def gpu_train_flags(
    *,
    cpu_count: int | None = None,
    cuda_available: bool | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> list[str]:
    """DataLoader + AMP flags for a Colab GPU run.

    Workers: 8 when the machine has at least 8 CPUs, else 4 (High-RAM Colab).
    pin_memory and persistent_workers are always on so page-locked host tensors
    reach the GPU without restarting workers every epoch.
    AMP: ``--bf16`` on compute capability 8+ (A100 / L4), else ``--fp16`` (T4).
    CPU callers get the DataLoader flags only (no AMP).
    """
    n = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    workers = 8 if n >= 8 else 4
    flags = [
        "--dataloader_num_workers",
        str(workers),
        "--dataloader_pin_memory",
        "true",
        "--dataloader_persistent_workers",
        "true",
    ]
    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    if not cuda_available:
        return flags
    if compute_capability is None:
        try:
            import torch

            compute_capability = torch.cuda.get_device_capability()
        except Exception:
            compute_capability = (7, 5)
    flags.append("--bf16" if compute_capability[0] >= 8 else "--fp16")
    return flags


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


def load_temperature(model_id: str, explicit: float | None = None) -> float:
    """T from an argument, else <model>/calibration.json, else shared/release.json."""
    if explicit is not None:
        return float(explicit)
    path = Path(model_id) / "calibration.json"
    if path.exists():
        return float(json.loads(path.read_text())["temperature"])
    env = os.environ.get("CALIBRATION_TEMPERATURE")
    return float(env) if env else float(RELEASE_TEMPERATURE)


class Predictor:
    """Loads a classifier from a local dir or the Hub and returns softmax top-k per image.

    Uses the model's own AutoImageProcessor (resize + center-crop + normalize), i.e. exactly
    what `transformers.pipeline("image-classification")` does, so Phase 5 can use either.
    Logits are divided by `temperature` (calibration.json) before softmax. `crop_hint` plus
    CROP_MASK (off by default) keeps only that crop's classes.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_HF_MODEL_ID,
        revision: str | None = None,
        device: str | None = None,
        token: str | None = None,
        temperature: float | None = None,
        crop_hint: str | None = None,
        crop_mask: bool | None = None,
    ):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        kwargs = {"revision": revision} if revision else {}
        token = token or get_hf_token()  # needed if the Hub repo is private
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
        self.temperature = load_temperature(model_id, temperature)
        self.crop_hint = crop_hint
        self.crop_mask = CROP_MASK if crop_mask is None else crop_mask

    def logits(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        inputs = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt").to(self.device)
        with torch.no_grad():
            z = self.model(**inputs).logits
        return z.float().cpu().numpy()

    def probabilities(self, images: list[Image.Image], crop_hint: str | None = None) -> np.ndarray:
        import torch
        from shared.vision.calibration import crop_mask as mask_crop

        z = self.logits(list(images))
        t = self.temperature
        if t != 1.0:
            z = z / t
        probs = torch.softmax(torch.from_numpy(z), dim=-1).numpy()
        hint = crop_hint if crop_hint is not None else self.crop_hint
        if self.crop_mask and hint:
            probs = mask_crop(probs, self.label_names, hint)
        return probs

    def predict_proba(self, images: list[Image.Image], crop_hint: str | None = None) -> list[dict[str, float]]:
        """shared.vision.strategies.Predictor: full softmax per image, one batch."""
        probs = self.probabilities(list(images), crop_hint=crop_hint)
        return [{self.id2label[i]: float(p[i]) for i in range(len(p))} for p in probs]

    def predict_topk(self, image: Image.Image, k: int = 3, crop_hint: str | None = None) -> TopK:
        probs = self.probabilities([image], crop_hint=crop_hint)[0]
        order = np.argsort(-probs)[:k]
        return TopK(labels=[self.id2label[int(i)] for i in order], scores=[float(probs[i]) for i in order])
