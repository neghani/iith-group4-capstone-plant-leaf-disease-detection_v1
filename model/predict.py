"""Reference inference for Phase 5: quality gate -> MobileNetV2 on Hugging Face -> top-k with "not sure".

This is what `api/app/vision/predictor.py` should do once the placeholder is replaced:

    from predict import diagnose
    result = diagnose("leaf.jpg")              # uses HF_MODEL_ID / HF_MODEL_REVISION env or defaults
    result["status"]                           # "ok" | "not_sure" | "retake_photo"

CLI:
    python model/predict.py leaf.jpg [--model neghani/plantvillage-mobilenetv2 --revision v1.0 --threshold 0.6]
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_HF_MODEL_ID,
    DEFAULT_HF_REVISION,
    Predictor,
    check_image_quality,
    crop_from_class,
    display_name,
    load_image,
)

DEFAULT_THRESHOLD = 0.6  # replace with reports/metrics.json -> recommended_confidence_threshold


@lru_cache(maxsize=4)
def get_predictor(model_id: str, revision: str | None, device: str | None = None) -> Predictor:
    return Predictor(model_id, revision=revision, device=device)


def diagnose(
    image: str | Path | Image.Image,
    model_id: str = DEFAULT_HF_MODEL_ID,
    revision: str | None = DEFAULT_HF_REVISION,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 3,
    skip_quality_gate: bool = False,
    device: str | None = None,
) -> dict:
    """Return a JSON-serialisable dict the API/agent can consume directly."""
    img = image if isinstance(image, Image.Image) else load_image(image)
    img = img.convert("RGB")

    quality = check_image_quality(img)
    out: dict = {
        "quality": {"ok": quality.ok, "message": quality.message, "brightness": quality.brightness, "blur_score": quality.blur_score},
        "model": model_id,
        "revision": revision,
        "threshold": threshold,
    }
    if not quality.ok and not skip_quality_gate:
        out.update({"status": "retake_photo", "disease": None, "confidence": None, "top_k": []})
        return out

    topk = get_predictor(model_id, revision, device).predict_topk(img, k=top_k)
    low = topk.confidence < threshold
    out.update(
        {
            "status": "not_sure" if low else "ok",
            "disease": topk.top1,
            "display_name": display_name(topk.top1),
            "crop": crop_from_class(topk.top1),
            "confidence": round(topk.confidence, 4),
            "low_confidence": low,
            "top_k": [{"label": l, "display_name": display_name(l), "score": round(s, 4)} for l, s in zip(topk.labels, topk.scores)],
        }
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", nargs="+", help="Image file(s).")
    p.add_argument("--model", default=DEFAULT_HF_MODEL_ID, help="Hub id or local model dir.")
    p.add_argument("--revision", default=DEFAULT_HF_REVISION)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--skip_quality_gate", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    revision = None if args.revision in ("", "main") and Path(args.model).exists() else args.revision
    for path in args.image:
        result = diagnose(path, args.model, revision, args.threshold, args.top_k, args.skip_quality_gate, args.device)
        result["image"] = path
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
