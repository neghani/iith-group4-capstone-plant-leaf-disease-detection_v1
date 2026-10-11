import hashlib
from dataclasses import dataclass

# Placeholder until Phase 5 Hugging Face model integration.

DEMO_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___Early_blight",
    "Apple___Apple_scab",
    "Corn_(maize)___Common_rust_",
]


@dataclass
class PredictionResult:
    disease_id: str
    confidence: float
    crop: str


def placeholder_predict(image_bytes: bytes, crop_hint: str | None = None) -> PredictionResult:
    digest = hashlib.sha256(image_bytes).hexdigest()
    idx = int(digest[:8], 16) % len(DEMO_CLASSES)
    disease_id = DEMO_CLASSES[idx]

    # Slightly lower confidence for some hashes to exercise low-confidence UI.
    conf_raw = (int(digest[8:16], 16) % 1000) / 1000.0
    confidence = 0.55 + conf_raw * 0.4

    if crop_hint:
        crop_hint_l = crop_hint.strip().lower()
        for cls in DEMO_CLASSES:
            crop_part = cls.split("___")[0].replace("_", " ").lower()
            if crop_hint_l in crop_part or crop_part.startswith(crop_hint_l):
                disease_id = cls
                confidence = min(0.92, confidence + 0.15)
                break

    from app.services.knowledge import get_entry

    entry = get_entry(disease_id)
    crop = entry.get("crop", "Unknown") if entry else "Unknown"
    return PredictionResult(disease_id=disease_id, confidence=confidence, crop=crop)
