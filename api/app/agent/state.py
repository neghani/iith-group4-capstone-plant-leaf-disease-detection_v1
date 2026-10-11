from typing import Any, TypedDict


class DiagnosisState(TypedDict, total=False):
    image_bytes: bytes
    crop_hint: str
    additional_symptoms: str
    quality_ok: bool
    quality_message: str | None
    disease_id: str
    confidence: float
    crop: str
    knowledge_entry: dict[str, Any]
    report: dict[str, Any]
    session_context: dict[str, Any]
    error: str
