"""Decode + upload quality gate. The only implementation of the gate; API and model both call it.

One decode, two derived versions (see docs/challenges-and-approach.md, entries 1-3):

1. **Working copy** (long side <= WORK_MAX_SIDE): used to find the leaf and judge quality.
   Large phone photos are decoded with JPEG draft mode so a 48 MP file never becomes a
   full-size numpy array.
2. **Inference copy** (INFERENCE_SIZE square): built by the crop strategy from the working copy
   and resized exactly once, straight to the size the classifier's preprocessing expects.

The gate always judges the best leaf by leaf_score (or the whole frame if no leaf), whatever
strategy runs afterwards, so gate-rejections are the same for every strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image, ImageOps

from shared.vision import config as C
from shared.vision.candidates import Candidate

logger = logging.getLogger("leafcheck.vision")

MESSAGES = {
    "unreadable": "Could not read image. Use JPEG, PNG, or WebP.",
    "too_many_pixels": "Image has too many pixels to process.",
    "image_too_small": "Image is too small. Move closer so the leaf fills more of the frame.",
    "leaf_too_small": "The leaf is too small in the photo. Move closer so it fills more of the frame.",
    "too_dark": "Photo is too dark. Retake in better light without strong shadows.",
    "overexposed": "Photo is overexposed. Avoid direct glare on the leaf.",
    "blurry": "Photo looks blurry. Hold steady and focus on the leaf surface.",
}


@dataclass
class GateResult:
    ok: bool
    reason: str | None = None  # key of MESSAGES when rejected
    brightness: float | None = None
    blur_score: float | None = None  # leaf re-blur ratio 0 (blurry) .. 1 (sharp)
    leaf_found: bool | None = None
    leaf_coverage: float | None = None
    original_size: list[int] | None = None
    trace: dict = field(default_factory=dict)

    @property
    def message(self) -> str | None:
        return MESSAGES.get(self.reason) if self.reason else None


def load_working_copy(data: bytes) -> tuple[Image.Image | None, dict, str | None]:
    """Decode once, upright and RGB, long side <= WORK_MAX_SIDE. Returns (img, meta, reason)."""
    try:
        img = Image.open(BytesIO(data))
        orig_size = img.size
        if img.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # phone stored it rotated 90 degrees
            orig_size = (orig_size[1], orig_size[0])
        if orig_size[0] * orig_size[1] > C.MAX_PIXELS:
            return None, {"original_size": list(orig_size)}, "too_many_pixels"
        if img.format == "JPEG":
            img.draft("RGB", (C.WORK_MAX_SIDE, C.WORK_MAX_SIDE))  # fast, low-memory DCT downscale
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception:
        return None, {}, "unreadable"

    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(flat, rgba).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if max(img.size) > C.WORK_MAX_SIDE:
        img.thumbnail((C.WORK_MAX_SIDE, C.WORK_MAX_SIDE), Image.Resampling.LANCZOS)
    meta = {"original_size": list(orig_size), "original_bytes": len(data), "work_size": list(img.size)}
    return img, meta, None


def check_gate(img: Image.Image, candidates: list[Candidate], original_size: tuple[int, int] | list[int] | None = None) -> GateResult:
    """Judge the best leaf (candidates[0]); a whole-frame candidate means no leaf was found."""
    orig_w, orig_h = original_size or img.size
    best = candidates[0]
    leaf_found = best.leaf_found
    scale = max(orig_w, orig_h) / max(img.size)
    native_leaf_px = max(best.width, best.height) * scale if leaf_found else float(max(orig_w, orig_h))
    coverage = best.area_frac if leaf_found else 0.0
    sharp = best.sharpness

    trace = dict(
        leaf_found=leaf_found,
        candidates=[c.trace() for c in candidates if c.leaf_found],
        chosen_bbox=list(best.box),
        sharpness=round(sharp.ratio, 3),
        laplacian_var=round(sharp.variance, 1),
        brightness=round(best.brightness, 1),
        leaf_coverage=round(coverage, 3),
    )
    result = GateResult(
        ok=True,
        brightness=best.brightness,
        blur_score=sharp.ratio,
        leaf_found=leaf_found,
        leaf_coverage=coverage,
        original_size=[orig_w, orig_h],
        trace=trace,
    )
    if orig_w < C.MIN_WIDTH or orig_h < C.MIN_HEIGHT:
        result.reason = "image_too_small"
    elif leaf_found and coverage < C.MIN_LEAF_COVERAGE and native_leaf_px < C.MIN_LEAF_NATIVE_PX:
        result.reason = "leaf_too_small"
    elif best.brightness < C.MIN_BRIGHTNESS:
        result.reason = "too_dark"
    elif best.brightness > C.MAX_BRIGHTNESS:
        result.reason = "overexposed"
    elif not sharp.passes_gate:
        result.reason = "blurry"
    result.ok = result.reason is None
    if not result.ok:
        trace["gate_reason"] = result.reason
        logger.info("gate rejected: reason=%s %s", result.reason, trace)
    return result


def rejected(reason: str, meta: dict | None = None) -> GateResult:
    """Gate result for a photo that could not be decoded or measured at all."""
    trace = {**(meta or {}), "gate_reason": reason}
    logger.info("gate rejected: reason=%s %s", reason, trace)
    return GateResult(ok=False, reason=reason, original_size=(meta or {}).get("original_size"), trace=trace)


def encode_jpeg(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=C.INFERENCE_JPEG_QUALITY)
    return buf.getvalue()
