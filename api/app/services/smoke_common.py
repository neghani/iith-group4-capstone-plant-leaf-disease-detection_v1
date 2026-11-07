"""Shared helpers for Web Test and Local Test smoke tables.

Gate + leaf crop + confidence band + optional GT crop hint — same idea as Check a crop.
"""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from app.config import get_settings
from app.services.crops import same_crop
from app.vision.hf_client import HFInferenceError
from app.vision.predictor import crop_from_class, display_name, predict
from app.vision.quality import PreparedImage, prepare_image

_THUMB_SIDE = 320
_THUMB_JPEG_QUALITY = 88


def score_verdict(
    *,
    pred_id: str | None,
    gt_id: str,
    score: float | None,
    model_status: str,
    use_gt_crop_hint: bool,
    threshold: float,
) -> dict:
    """App-like band: abstain below threshold / not_sure; crop hint flags wrong-crop highs."""
    gt_crop = crop_from_class(gt_id)
    pred_crop = crop_from_class(pred_id) if pred_id else ""
    crop_mismatch = bool(
        use_gt_crop_hint and pred_crop and gt_crop and not same_crop(pred_crop, gt_crop)
    )

    below = score is None or float(score) < float(threshold) - 1e-9
    if model_status in ("not_sure", "retake_photo", "error") or below:
        return {
            "match": False,
            "verdict": "abstain",
            "crop_mismatch": crop_mismatch,
            "gt_crop": gt_crop,
            "pred_crop": pred_crop or None,
            "confidence_threshold": threshold,
        }

    if crop_mismatch:
        return {
            "match": False,
            "verdict": "crop_mismatch",
            "crop_mismatch": True,
            "gt_crop": gt_crop,
            "pred_crop": pred_crop,
            "confidence_threshold": threshold,
        }

    ok = bool(pred_id and pred_id == gt_id)
    return {
        "match": ok,
        "verdict": "yes" if ok else "no",
        "crop_mismatch": False,
        "gt_crop": gt_crop,
        "pred_crop": pred_crop or None,
        "confidence_threshold": threshold,
    }


def thumb_data_url(img: Image.Image) -> str:
    preview = img.convert("RGB")
    preview.thumbnail((_THUMB_SIDE, _THUMB_SIDE), Image.Resampling.LANCZOS)
    buf = BytesIO()
    preview.save(buf, format="JPEG", quality=_THUMB_JPEG_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def preview_image(prepared: PreparedImage, fallback_jpeg: bytes) -> Image.Image:
    if prepared.candidates:
        return prepared.candidates[0].crop
    if prepared.image is not None:
        return prepared.image
    return Image.open(BytesIO(fallback_jpeg)).convert("RGB")


def classify_prepared_row(
    index: int,
    class_id: str,
    source_url: str,
    jpeg: bytes,
    prepared: PreparedImage | None = None,
    *,
    use_gt_crop_hint: bool = True,
) -> dict:
    """Run gate (if needed) + strategy predict; return one NDJSON row dict."""
    settings = get_settings()
    pin_id, pin_rev = settings.hf_model_id, settings.hf_model_revision
    threshold = float(settings.confidence_threshold)
    if prepared is None:
        prepared = prepare_image(jpeg)
    gate = prepared.quality
    preview = thumb_data_url(preview_image(prepared, jpeg))
    base = {
        "status": "row",
        "index": index,
        "image": preview,
        "source_url": source_url,
        "gt_id": class_id,
        "gt_display": display_name(class_id),
        "gate_ok": gate.ok,
        "leaf_found": gate.leaf_found,
        "crop_strategy": settings.crop_strategy,
        "use_gt_crop_hint": use_gt_crop_hint,
        "model_id": pin_id,
        "model_revision": pin_rev,
        "confidence_threshold": threshold,
    }

    if not gate.ok:
        scored = score_verdict(
            pred_id=None,
            gt_id=class_id,
            score=None,
            model_status="retake_photo",
            use_gt_crop_hint=use_gt_crop_hint,
            threshold=threshold,
        )
        return {
            **base,
            **scored,
            "pred_id": None,
            "pred_display": None,
            "score": None,
            "model_status": "retake_photo",
            "error": gate.message or "Quality gate rejected this photo.",
            "latency_ms": None,
        }

    try:
        pred = predict(prepared)
    except HFInferenceError as exc:
        scored = score_verdict(
            pred_id=None,
            gt_id=class_id,
            score=None,
            model_status="error",
            use_gt_crop_hint=use_gt_crop_hint,
            threshold=threshold,
        )
        return {
            **base,
            **scored,
            "pred_id": None,
            "pred_display": None,
            "score": None,
            "model_status": "error",
            "error": str(exc),
            "latency_ms": None,
        }

    top = pred.top_k[0] if pred.top_k else None
    pred_id = pred.disease_id or (top.label if top else None)
    score = pred.confidence if pred.disease_id else (top.score if top else None)
    score_f = round(float(score), 4) if score is not None else None
    meta = pred.inference or {}
    scored = score_verdict(
        pred_id=pred_id,
        gt_id=class_id,
        score=score_f,
        model_status=pred.status,
        use_gt_crop_hint=use_gt_crop_hint,
        threshold=threshold,
    )
    return {
        **base,
        **scored,
        "pred_id": pred_id,
        "pred_display": display_name(pred_id) if pred_id else None,
        "score": score_f,
        "model_status": pred.status,
        "top_k": [
            {
                "label": item.label,
                "display_name": item.display_name,
                "score": item.score,
            }
            for item in pred.top_k[:3]
        ],
        "model_id": meta.get("model_id") or pin_id,
        "model_revision": meta.get("model_revision") or pin_rev,
        "latency_ms": meta.get("latency_ms"),
        "leaf_found": bool((pred.strategy or {}).get("leaf_found", gate.leaf_found)),
    }


def tally_verdicts(rows: list[dict]) -> dict:
    correct = abstain = wrong = 0
    for row in rows:
        if row.get("match"):
            correct += 1
        elif row.get("verdict") == "abstain":
            abstain += 1
        else:
            wrong += 1
    n = len(rows)
    answered = correct + wrong
    return {
        "n": n,
        "correct": correct,
        "abstain": abstain,
        "wrong": wrong,
        "accuracy": round(correct / n, 3) if n else 0.0,
        "answered_accuracy": round(correct / answered, 3) if answered else None,
    }
