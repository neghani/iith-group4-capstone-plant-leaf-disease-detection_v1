from app.config import get_settings
from app.schemas import (
    ActionItem,
    ConfidenceCategory,
    DiagnosisReport,
    DiagnosisStatus,
    SymptomItem,
)
from app.services.knowledge import get_entry
from app.llm.provider import generate_text


def _confidence_category(confidence: float, threshold: float) -> ConfidenceCategory:
    if confidence < threshold - 0.15:
        return "inconclusive"
    if confidence < threshold:
        return "low"
    if confidence < 0.85:
        return "moderate"
    return "high"


def _confidence_explanation(category: ConfidenceCategory) -> str:
    return {
        "high": "Visual pattern and model score align well; still confirm before major treatment spend.",
        "moderate": "Likely match, but similar diseases are possible—monitor and confirm if symptoms spread.",
        "low": "Weak match from the placeholder model; treat this as a tentative guess.",
        "inconclusive": "Not enough certainty for a firm diagnosis from this photo alone.",
    }[category]


def build_quality_report(message: str) -> DiagnosisReport:
    return DiagnosisReport(
        status="quality_issue",
        status_label="Photo needs improvement",
        diagnosis_name="Unable to analyze",
        confidence=0.0,
        confidence_category="inconclusive",
        confidence_explanation="Image quality checks failed before prediction.",
        description=message,
        expert_confirmation_required=True,
        actions=[
            ActionItem(
                step="Retake the photo with the leaf centered, in daylight, and in focus.",
            )
        ],
        monitoring_guidance="After retaking, upload again for a new assessment.",
        diagnostic_limitations="No disease prediction was run on this image.",
        initial_message=message,
    )


def build_report_from_knowledge(
    disease_id: str,
    confidence: float,
    crop: str,
) -> DiagnosisReport:
    settings = get_settings()
    entry = get_entry(disease_id)
    if not entry:
        return DiagnosisReport(
            status="inconclusive",
            status_label="Unknown disease class",
            disease_id=disease_id,
            diagnosis_name=disease_id.replace("___", " — ").replace("_", " "),
            crop=crop,
            confidence=confidence,
            confidence_category="inconclusive",
            confidence_explanation=_confidence_explanation("inconclusive"),
            description="No curated guidance found for this class in the demo knowledge base.",
            expert_confirmation_required=True,
            diagnostic_limitations="Demo knowledge file has limited entries.",
            initial_message="We could not load detailed guidance for this prediction.",
        )

    cat = _confidence_category(confidence, settings.confidence_threshold)
    low = confidence < settings.confidence_threshold
    status: DiagnosisStatus = "possible" if low else "confirmed_likely"
    status_label = "Possible disease" if low else "Likely disease"

    symptoms = [
        SymptomItem(name=s["name"], explanation=s["explanation"], evidence_source="reference")
        for s in entry.get("symptoms", [])
    ]
    actions = [
        ActionItem(step=a["step"], safety_note=a.get("safety_note"))
        for a in entry.get("actions", [])
    ]

    report = DiagnosisReport(
        status=status,
        status_label=status_label,
        disease_id=disease_id,
        diagnosis_name=entry["display_name"],
        crop=entry.get("crop", crop),
        confidence=round(confidence, 3),
        confidence_category=cat,
        confidence_explanation=_confidence_explanation(cat),
        description=entry["description"],
        symptoms=symptoms,
        alternatives=entry.get("alternatives", []),
        expert_confirmation_required=low or cat in ("low", "inconclusive"),
        actions=actions,
        monitoring_guidance=entry.get("monitoring", ""),
        diagnostic_limitations=entry.get("limitations", ""),
    )
    report.initial_message = _compose_initial_message(report)
    return report


def _compose_initial_message(report: DiagnosisReport) -> str:
    lines = [
        f"{report.status_label}: {report.diagnosis_name}.",
        report.description,
        "What you can do now:",
    ]
    for i, action in enumerate(report.actions[:3], start=1):
        lines.append(f"{i}. {action.step}")
    if report.diagnostic_limitations:
        lines.append(f"Note: {report.diagnostic_limitations}")
    return " ".join(lines)


def enrich_report_with_llm(report: DiagnosisReport, reference: dict) -> DiagnosisReport:
    """Optional polish of farmer-facing summary; keeps structured fields from knowledge."""
    prompt = (
        "Rewrite a short farmer-friendly summary (3-5 sentences) using ONLY the facts below. "
        "Do not add new treatments or products.\n\n"
        f"Disease: {report.diagnosis_name}\n"
        f"Description: {report.description}\n"
        f"Actions: {[a.step for a in report.actions]}\n"
        f"Monitoring: {report.monitoring_guidance}\n"
        f"Limitations: {report.diagnostic_limitations}\n"
    )
    text = generate_text(
        "You write plain-language agricultural guidance for smallholder farmers.",
        prompt,
    )
    if text:
        report.initial_message = text
    return report
