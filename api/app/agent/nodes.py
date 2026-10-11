from dataclasses import asdict

from app.agent.state import DiagnosisState
from app.config import get_settings
from app.schemas import DiagnosisReport
from app.services.crops import resolve_hint, same_crop
from app.services.knowledge import get_entry
from app.services.report import (
    apply_crop_mismatch,
    build_error_report,
    build_report_from_knowledge,
    build_unsupported_crop_report,
    enrich_report_with_llm,
)
from app.vision.hf_client import HFInferenceError
from app.vision.predictor import predict


def predict_node(state: DiagnosisState) -> DiagnosisState:
    try:
        pred = predict(state["prepared"])
    except HFInferenceError as exc:
        return {**state, "error": str(exc)}
    flag, note = _check_crop_hint(state.get("crop_hint", ""), pred.crop)
    return {
        **state,
        "disease_id": pred.disease_id,
        "confidence": pred.confidence,
        "crop": pred.crop,
        "top_k": [asdict(item) for item in pred.top_k],
        "inference": pred.inference,
        "strategy": pred.strategy,
        "crop_flag": flag,
        "crop_note": note,
        "low_confidence": pred.status != "ok" or bool(flag),
    }


def _check_crop_hint(hint: str, predicted_crop: str) -> tuple[str, str]:
    hint = hint.strip()
    if not hint:
        return "", ""
    canonical = resolve_hint(hint)
    if canonical is None:
        return "unsupported", f"the farmer says the crop is {hint}, which the classifier was never trained on"
    if not same_crop(canonical, predicted_crop):
        return (
            "mismatch",
            f"the farmer says the crop is {hint} but the classifier matched {predicted_crop}",
        )
    return "", ""


def predict_error_node(state: DiagnosisState) -> DiagnosisState:
    report = build_error_report()
    return {**state, "report": report.model_dump(), "top_k": []}


def unsupported_crop_node(state: DiagnosisState) -> DiagnosisState:
    report = build_unsupported_crop_report(state.get("crop_hint", ""))
    return {**state, "report": report.model_dump()}


def knowledge_node(state: DiagnosisState) -> DiagnosisState:
    entry = get_entry(state["disease_id"]) or {}
    return {**state, "knowledge_entry": entry}


def generate_report_node(state: DiagnosisState) -> DiagnosisState:
    report = build_report_from_knowledge(
        state["disease_id"],
        state["confidence"],
        state.get("crop", "Unknown"),
        state.get("top_k", []),
    )
    report = enrich_report_with_llm(
        report,
        reference=state.get("knowledge_entry") or {},
        top_k=state.get("top_k", []),
        low_confidence=state.get("low_confidence", False),
        crop_hint=state.get("crop_hint", ""),
        additional_symptoms=state.get("additional_symptoms", ""),
        crop_note=state.get("crop_note", ""),
    )
    if state.get("crop_flag") == "mismatch":
        report = apply_crop_mismatch(report, state.get("crop_hint", ""), state.get("crop", ""))
    if state.get("additional_symptoms"):
        extra = state["additional_symptoms"].strip()
        report.description = f"{report.description} You also noted: {extra}"
    return {**state, "report": report.model_dump()}


def safety_node(state: DiagnosisState) -> DiagnosisState:
    report_dict = state.get("report") or {}
    report = DiagnosisReport.model_validate(report_dict)
    settings = get_settings()
    low = (
        report.confidence < settings.confidence_threshold
        or bool(state.get("crop_flag"))
        or bool(state.get("low_confidence"))
    )
    if low and report.status not in ("quality_issue", "error"):
        report.expert_confirmation_required = True
        if report.status in ("confirmed_likely", "healthy"):
            report.status = "possible"
            report.status_label = "Possible match"
    disclaimer = (
        "General guidance only—not a substitute for a local agricultural expert or lab test."
    )
    if disclaimer not in report.diagnostic_limitations:
        report.diagnostic_limitations = (
            f"{report.diagnostic_limitations} {disclaimer}".strip()
        )
    ctx = {
        "disease_id": report.disease_id,
        "diagnosis_name": report.diagnosis_name,
        "crop": report.crop,
        "confidence": report.confidence,
        "status": report.status,
        "low_confidence": low,
        "top_k": state.get("top_k", []),
        "summary": report.initial_message,
    }
    return {**state, "report": report.model_dump(), "session_context": ctx}
