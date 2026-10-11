from typing import Any, TypedDict


class DiagnosisState(TypedDict, total=False):
    # Quality-gated working copy + leaf candidates (app.vision.quality.PreparedImage);
    # only predict_node reads it.
    prepared: Any
    crop_hint: str
    additional_symptoms: str
    disease_id: str
    confidence: float
    crop: str
    top_k: list[dict[str, Any]]
    inference: dict[str, Any]  # provenance of the Hugging Face call
    strategy: dict[str, Any]  # crop strategy, number of crops, agreement
    # "" | "mismatch" (hint != predicted crop) | "unsupported" (hint is outside the model's crops)
    crop_flag: str
    crop_note: str
    low_confidence: bool
    knowledge_entry: dict[str, Any]
    report: dict[str, Any]
    session_context: dict[str, Any]
    error: str
