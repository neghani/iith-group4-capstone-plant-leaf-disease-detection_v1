from app.agent.state import DiagnosisState
from app.config import get_settings
from app.schemas import DiagnosisReport
from app.services.knowledge import get_entry
from app.services.report import (
    build_quality_report,
    build_report_from_knowledge,
    enrich_report_with_llm,
)
from app.vision.predictor import placeholder_predict
from app.vision.quality import check_image_quality


def check_quality_node(state: DiagnosisState) -> DiagnosisState:
    quality, _ = check_image_quality(state["image_bytes"])
    return {
        **state,
        "quality_ok": quality.ok,
        "quality_message": quality.message,
    }


def predict_node(state: DiagnosisState) -> DiagnosisState:
    pred = placeholder_predict(state["image_bytes"], state.get("crop_hint"))
    return {
        **state,
        "disease_id": pred.disease_id,
        "confidence": pred.confidence,
        "crop": pred.crop,
    }


def knowledge_node(state: DiagnosisState) -> DiagnosisState:
    entry = get_entry(state["disease_id"]) or {}
    return {**state, "knowledge_entry": entry}


def generate_report_node(state: DiagnosisState) -> DiagnosisState:
    report = build_report_from_knowledge(
        state["disease_id"],
        state["confidence"],
        state.get("crop", "Unknown"),
    )
    if state.get("knowledge_entry"):
        report = enrich_report_with_llm(report, state["knowledge_entry"])
    if state.get("additional_symptoms"):
        extra = state["additional_symptoms"].strip()
        report.description = f"{report.description} You also noted: {extra}"
    return {**state, "report": report.model_dump()}


def quality_fail_node(state: DiagnosisState) -> DiagnosisState:
    msg = state.get("quality_message") or "Please upload a clearer photo."
    report = build_quality_report(msg)
    return {**state, "report": report.model_dump(), "quality_ok": False}


def safety_node(state: DiagnosisState) -> DiagnosisState:
    report_dict = state.get("report") or {}
    report = DiagnosisReport.model_validate(report_dict)
    settings = get_settings()
    low = report.confidence < settings.confidence_threshold
    if low and report.status not in ("quality_issue", "error"):
        report.expert_confirmation_required = True
        if report.status == "confirmed_likely":
            report.status = "possible"
            report.status_label = "Possible disease"
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
        "summary": report.initial_message,
    }
    return {**state, "report": report.model_dump(), "session_context": ctx}
