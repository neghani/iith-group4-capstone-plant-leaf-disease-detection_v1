"""Reference inference: shared quality gate + crop strategy -> MobileNetV2 -> top-k with "not sure".

Same preprocessing as the API (shared/vision), with the model run locally:

    from predict import diagnose
    result = diagnose("leaf.jpg")              # uses HF_MODEL_ID / HF_MODEL_REVISION env or defaults
    result["status"]                           # "ok" | "not_sure" | "retake_photo"

CLI:
    python model/predict.py leaf.jpg [--model neghani/plantvillage-mobilenetv2 --revision v1.0 --threshold 0.6 --strategy B]
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
    strategy: str = "B",
) -> dict:
    """Return a JSON-serialisable dict the API/agent can consume directly."""
    from shared.vision.candidates import find_candidates
    from shared.vision.config import WORK_MAX_SIDE
    from shared.vision.quality import check_gate
    from shared.vision.strategies import run_strategy

    img = image if isinstance(image, Image.Image) else load_image(image)
    img = img.convert("RGB")
    size = img.size
    if max(size) > WORK_MAX_SIDE:
        img.thumbnail((WORK_MAX_SIDE, WORK_MAX_SIDE), Image.Resampling.LANCZOS)

    candidates = find_candidates(img)
    gate = check_gate(img, candidates, size)
    out: dict = {
        "quality": {"ok": gate.ok, "message": gate.message, "reason": gate.reason,
                    "brightness": gate.brightness, "blur_score": gate.blur_score},
        "model": model_id,
        "revision": revision,
        "threshold": threshold,
    }
    if skip_quality_gate:
        gate.ok, gate.reason = True, None
    res = run_strategy(img, strategy, get_predictor(model_id, revision, device), threshold=threshold,
                       top_k=top_k, candidates=candidates, gate=gate)
    out.update(res)
    if res["disease"]:
        out.update(display_name=display_name(res["disease"]), crop=crop_from_class(res["disease"]),
                   confidence=round(res["confidence"], 4), low_confidence=res["status"] != "ok")
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
    p.add_argument("--strategy", default="B", choices=list("ABCDE"))
    args = p.parse_args()

    revision = None if args.revision in ("", "main") and Path(args.model).exists() else args.revision
    for path in args.image:
        result = diagnose(path, args.model, revision, args.threshold, args.top_k, args.skip_quality_gate,
                          args.device, args.strategy)
        result["image"] = path
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
