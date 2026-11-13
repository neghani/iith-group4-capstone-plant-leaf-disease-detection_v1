from dataclasses import asdict

from app.demo_log import demo
from app.agent.state import DiagnosisState
from app.config import get_settings
from app.schemas import DiagnosisReport
from app.services.crops import resolve_hint, same_crop
from app.services.knowledge import get_entry
from app.services.report import (
    apply_crop_mismatch,
    build_error_report,
    build_not_sure_report,
    build_quality_report,
    build_report_from_knowledge,
    build_unsupported_crop_report,
    enrich_report_with_llm,
)
from app.vision.hf_client import HFInferenceError
from app.vision.predictor import predict


def predict_node(state: DiagnosisState) -> DiagnosisState:
    demo("AGENT", "▶", "node: predict — sending leaf crop to the classifier")
    try:
        pred = predict(state["prepared"])
    except HFInferenceError as exc:
        demo("AGENT", "✖", f"predict failed — classifier unavailable ({exc})")
        return {**state, "error": str(exc)}
    top1 = (
        f"{pred.top_k[0].display_name} {pred.top_k[0].score:.1%}"
        if pred.top_k
        else "n/a"
    )
    demo(
        "AGENT",
        "✔",
        f"predict → {pred.disease_id or '(no class)'} conf={pred.confidence:.3f}"
        f" status={pred.status} · top1 {top1}",
    )
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
        "model_status": pred.status,
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
    demo("AGENT", "▶", "node: predict_error — building 'Analysis unavailable' report")
    report = build_error_report()
    return {**state, "report": report.model_dump(), "top_k": []}


def unsupported_crop_node(state: DiagnosisState) -> DiagnosisState:
    demo("AGENT", "▶", "node: unsupported_crop — building 'Crop not supported' report")
    report = build_unsupported_crop_report(state.get("crop_hint", ""))
    return {**state, "report": report.model_dump()}


def retake_node(state: DiagnosisState) -> DiagnosisState:
    strategy = state.get("strategy") or {}
    message = strategy.get("message") or "Please retake the photo with the leaf in focus."
    demo("AGENT", "▶", f"node: retake — {message}")
    report = build_quality_report(message)
    return {**state, "report": report.model_dump(), "top_k": []}


def not_sure_node(state: DiagnosisState) -> DiagnosisState:
    strategy = state.get("strategy") or {}
    ood_thr = float(strategy.get("ood_threshold") or 0.0)
    margin = strategy.get("margin")
    demo(
        "AGENT",
        "▶",
        f"node: not_sure — conf={state.get('confidence', 0):.3f} margin={margin} ood_thr={ood_thr}",
    )
    reason = ""
    if (
        ood_thr > 0
        and margin is not None
        and float(margin) < ood_thr - 1e-9
        and float(state.get("confidence") or 0.0) >= get_settings().confidence_threshold
    ):
        reason = "ood_margin"
    report = build_not_sure_report(
        top_k=state.get("top_k") or [],
        confidence=float(state.get("confidence") or 0.0),
        crop=state.get("crop") or "",
        reason=reason,
    )
    return {**state, "report": report.model_dump()}


def knowledge_node(state: DiagnosisState) -> DiagnosisState:
    demo("AGENT", "▶", "node: knowledge — KB lookup in diseases.json")
    entry = get_entry(state["disease_id"]) or {}
    if entry:
        demo(
            "AGENT",
            "✔",
            f"KB hit: '{entry.get('display_name', state['disease_id'])}'"
            f" ({len(entry.get('symptoms', []))} symptoms, {len(entry.get('actions', []))} actions)",
        )
    else:
        demo("AGENT", "→", "no curated KB entry — falling back to a label-only report")
    return {**state, "knowledge_entry": entry}


def generate_report_node(state: DiagnosisState) -> DiagnosisState:
    demo("AGENT", "▶", "node: generate_report — building the structured report")
    report = build_report_from_knowledge(
        state["disease_id"],
        state["confidence"],
        state.get("crop", "Unknown"),
        state.get("top_k", []),
    )
    # Before enrichment, initial_message is the fixed template; if the LLM changes
    # it, provenance can honestly say the summary text was LLM-written.
    template_message = report.initial_message
    report = enrich_report_with_llm(
        report,
        reference=state.get("knowledge_entry") or {},
        top_k=state.get("top_k", []),
        low_confidence=state.get("low_confidence", False),
        crop_hint=state.get("crop_hint", ""),
        additional_symptoms=state.get("additional_symptoms", ""),
        crop_note=state.get("crop_note", ""),
    )
    llm_used = report.initial_message != template_message
    demo(
        "AGENT",
        "✔",
        "generate_report done — summary "
        + ("rewritten by the LLM" if llm_used else "from the curated KB template"),
    )
    if state.get("crop_flag") == "mismatch":
        demo("AGENT", "→", "crop hint mismatch — downgrading report to 'Possible match'")
        report = apply_crop_mismatch(report, state.get("crop_hint", ""), state.get("crop", ""))
    if state.get("additional_symptoms"):
        extra = state["additional_symptoms"].strip()
        report.description = f"{report.description} You also noted: {extra}"
    return {**state, "report": report.model_dump(), "llm_used": llm_used}


def safety_node(state: DiagnosisState) -> DiagnosisState:
    demo("AGENT", "▶", "node: safety — final guardrails and disclaimers")
    report_dict = state.get("report") or {}
    report = DiagnosisReport.model_validate(report_dict)
    settings = get_settings()
    low = (
        report.confidence < settings.confidence_threshold
        or bool(state.get("crop_flag"))
        or bool(state.get("low_confidence"))
    )
    if low and report.status not in ("quality_issue", "error", "not_sure"):
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
    demo(
        "AGENT",
        "✔",
        f"safety done → status={report.status} expert_confirmation={report.expert_confirmation_required}"
        f" low_confidence={low} · graph END",
    )
    return {**state, "report": report.model_dump(), "session_context": ctx}
