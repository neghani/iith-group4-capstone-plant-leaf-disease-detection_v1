from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from app.schemas import ImageQualityResult

MIN_WIDTH = 128
MIN_HEIGHT = 128
MAX_DIMENSION = 4096
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 230.0
MIN_BLUR_VARIANCE = 80.0


def _laplacian_variance(gray: np.ndarray) -> float:
    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    lap = (
        -4 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(lap.var())


def check_image_quality(image_bytes: bytes) -> tuple[ImageQualityResult, Image.Image | None]:
    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()
    except Exception:
        return ImageQualityResult(ok=False, message="Could not read image. Use JPEG, PNG, or WebP."), None

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    w, h = img.size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return (
            ImageQualityResult(
                ok=False,
                message="Image is too small. Move closer so the leaf fills more of the frame.",
            ),
            None,
        )
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        return (
            ImageQualityResult(ok=False, message="Image dimensions are too large."),
            None,
        )

    gray = np.array(img.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    blur_score = _laplacian_variance(gray)

    if brightness < MIN_BRIGHTNESS:
        return (
            ImageQualityResult(
                ok=False,
                message="Photo is too dark. Retake in better light without strong shadows.",
                brightness=brightness,
                blur_score=blur_score,
            ),
            img,
        )
    if brightness > MAX_BRIGHTNESS:
        return (
            ImageQualityResult(
                ok=False,
                message="Photo is overexposed. Avoid direct glare on the leaf.",
                brightness=brightness,
                blur_score=blur_score,
            ),
            img,
        )
    if blur_score < MIN_BLUR_VARIANCE:
        return (
            ImageQualityResult(
                ok=False,
                message="Photo looks blurry. Hold steady and focus on the leaf surface.",
                brightness=brightness,
                blur_score=blur_score,
            ),
            img,
        )

    return (
        ImageQualityResult(ok=True, brightness=brightness, blur_score=blur_score),
        img,
    )


def validate_upload(content_type: str | None, size: int, max_bytes: int) -> str | None:
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if content_type and content_type not in allowed:
        return "Unsupported file type. Upload JPEG, PNG, or WebP."
    if size <= 0:
        return "Empty file uploaded."
    if size > max_bytes:
        return "Image exceeds maximum size (10 MB)."
    return None
