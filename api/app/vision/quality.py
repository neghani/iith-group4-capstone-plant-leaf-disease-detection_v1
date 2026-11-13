"""API adapter for the shared upload gate (shared/vision/quality.py).

Decodes the upload, finds leaf candidates once and runs the gate. The candidates are kept on
`PreparedImage` so the crop strategy in `predictor.py` reuses them instead of searching again.
"""

import logging
import time
from dataclasses import dataclass, field

from PIL import Image

from app.demo_log import Stage
from app.schemas import ImageQualityResult
from shared.vision.candidates import Candidate, find_candidates
from shared.vision.quality import GateResult, check_gate, load_working_copy, rejected

logger = logging.getLogger("leafcheck.vision")


@dataclass
class PreparedImage:
    quality: ImageQualityResult
    image: Image.Image | None = None  # working copy; None when the gate rejected the upload
    candidates: list[Candidate] = field(default_factory=list)
    gate: GateResult | None = None
    trace: dict = field(default_factory=dict)


def _quality(gate: GateResult) -> ImageQualityResult:
    return ImageQualityResult(
        ok=gate.ok,
        reason=gate.reason,
        message=gate.message,
        brightness=gate.brightness,
        blur_score=gate.blur_score,
        leaf_found=gate.leaf_found,
        leaf_coverage=gate.leaf_coverage,
        original_size=gate.original_size,
    )


def prepare_image(data: bytes) -> PreparedImage:
    stage = Stage("QUALITY", f"gate ▶ decoding upload ({len(data) / 1024:.0f} KB)")
    started = time.perf_counter()
    img, meta, reason = load_working_copy(data)
    if img is None:
        gate = rejected(reason or "unreadable", meta)
        stage.fail(f"unreadable image — gate REJECT ({gate.reason})")
        return PreparedImage(_quality(gate), gate=gate, trace=gate.trace)

    decoded = time.perf_counter()
    w, h = meta["original_size"]
    stage.info(f"decoded {w}×{h} · finding leaf candidates")
    candidates = find_candidates(img)
    stage.info(f"{len(candidates)} leaf candidate(s) found · running quality checks")
    gate = check_gate(img, candidates, meta["original_size"])
    trace = {
        **meta,
        **gate.trace,
        "decode_ms": round((decoded - started) * 1000),
        "analyse_ms": round((time.perf_counter() - decoded) * 1000),
    }
    if not gate.ok:
        stage.fail(f"gate REJECT ({gate.reason}) — {gate.message}")
        return PreparedImage(_quality(gate), gate=gate, trace=trace)
    stage.ok(
        f"PASS — brightness {gate.brightness:.2f} · blur {gate.blur_score:.1f}"
        f" · leaf {gate.leaf_coverage:.1%}"
    )
    logger.info("image prepared: %s", trace)
    return PreparedImage(_quality(gate), img, candidates, gate, trace)


def validate_upload(content_type: str | None, size: int, max_bytes: int) -> str | None:
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if content_type and content_type not in allowed:
        return "Unsupported file type. Upload JPEG, PNG, or WebP."
    if size <= 0:
        return "Empty file uploaded."
    if size > max_bytes:
        return f"Image exceeds maximum size ({max_bytes // (1024 * 1024)} MB)."
    return None
