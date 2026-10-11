"""The live app's image chain as a `fn(PIL.Image) -> dict` for eval_real_world_set.py.

Upload bytes -> shared decode + quality gate + crop strategy (shared/vision) -> 256x256 JPEG
crops -> the same Hub weights the Space serves (hf_space/app.py runs `pipeline` on exactly these
bytes) -> CONFIDENCE_THRESHOLD. Classification runs locally, so no Space or endpoint is needed.
Needs `scipy` (shared/vision); it is in model/requirements.txt.

    python model/eval_real_world_set.py --set_dir model/data/real_world_set --split final \
        --predictor app_predictor:diagnose --strategy B --threshold 0.5
"""

from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import DEFAULT_HF_MODEL_ID, DEFAULT_HF_REVISION  # noqa: E402
from predict import get_predictor  # noqa: E402
from shared.vision.quality import encode_jpeg, load_working_copy, rejected  # noqa: E402
from shared.vision.strategies import retake_result, run_strategy  # noqa: E402

# Set by eval_real_world_set.py --model / --revision / --strategy / --threshold; a local run dir works too.
MODEL = os.environ.get("EVAL_MODEL", DEFAULT_HF_MODEL_ID)
REVISION = None if Path(MODEL).exists() else os.environ.get("EVAL_REVISION", DEFAULT_HF_REVISION)
STRATEGY = os.environ.get("CROP_STRATEGY", "B")
THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
TTA = os.environ.get("CROP_TTA", "").lower() in ("1", "true", "yes")
TOP_K = 3


class WirePredictor:
    """Local model on the exact JPEG bytes the API would send to the Space (q95, 256x256)."""

    def __init__(self, model: str = MODEL, revision: str | None = REVISION, device: str | None = None):
        self.model = get_predictor(model, revision, device)

    def predict_proba(self, images: list[Image.Image]) -> list[dict[str, float]]:
        sent = [Image.open(BytesIO(encode_jpeg(im))).convert("RGB") for im in images]
        return self.model.predict_proba(sent)


def upload_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def diagnose_bytes(data: bytes, strategy: str = STRATEGY, threshold: float = THRESHOLD,
                   predictor=None, tta: bool = TTA) -> dict:
    """Same path as POST /predict: decode -> gate -> strategy -> classifier -> decision."""
    img, meta, reason = load_working_copy(data)
    if img is None:
        return retake_result(rejected(reason or "unreadable", meta), strategy, threshold)
    return run_strategy(img, strategy, predictor or WirePredictor(), threshold=threshold, top_k=TOP_K,
                        original_size=meta["original_size"], tta=tta)


def diagnose(img: Image.Image | bytes) -> dict:
    return diagnose_bytes(img if isinstance(img, bytes) else upload_bytes(img))


diagnose.accepts_bytes = True  # eval_real_world_set.py passes the original file, as uploaded
