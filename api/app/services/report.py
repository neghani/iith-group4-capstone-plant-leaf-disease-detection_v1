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
from app.vision.predictor import crop_from_class, display_name, is_healthy


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
        "low": "Weak match from the model; treat this as a tentative guess.",
        "inconclusive": "Not enough certainty for a firm diagnosis from this photo alone.",
    }[category]


MIN_ALTERNATIVE_SCORE = 0.01


def _alternatives_from_top_k(disease_id: str, top_k: list[dict]) -> list[str]:
    return [
        f"{t['display_name']} ({round(t['score'] * 100)}%)"
        for t in top_k
        if t["label"] != disease_id and t["score"] >= MIN_ALTERNATIVE_SCORE
    ]


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


def build_not_sure_report(
    top_k: list[dict] | None = None,
    confidence: float = 0.0,
    crop: str = "",
    *,
    reason: str = "",
) -> DiagnosisReport:
    """Low-confidence or OOD classifier output: do not write up a disease as if it were chosen."""
    top_k = top_k or []
    alternatives = [
        f"{t['display_name']} ({round(t['score'] * 100)}%)"
        for t in top_k
        if t.get("score", 0) >= MIN_ALTERNATIVE_SCORE
    ]
    ood = reason == "ood_margin"
    explanation = (
        "The top matches are too close together to trust the crop or disease."
        if ood
        else "The model score is below the confidence threshold."
    )
    description = (
        "The photo was readable, but the model cannot confirm the crop or disease with enough "
        "separation between its top guesses. Retake in daylight with one leaf filling the frame, "
        "or ask a local expert. The matches below are possibilities only."
        if ood
        else (
            "The photo was readable, but the model is not sure enough to name a disease. "
            "Retake in daylight with one leaf filling the frame, or ask a local expert. "
            "The matches below are possibilities only."
        )
    )
    message = (
        "I'm not confident — please retake the photo. "
        + (
            "The top matches are too close to trust."
            if ood
            else "The score is below the threshold, so this is not a diagnosis."
        )
    )
    return DiagnosisReport(
        status="not_sure",
        status_label="Not sure",
        diagnosis_name="I'm not confident — please retake the photo.",
        crop=crop or None,
        confidence=round(confidence or 0.0, 3),
        confidence_category="inconclusive",
        confidence_explanation=explanation,
        description=description,
        alternatives=alternatives,
        expert_confirmation_required=True,
        actions=[
            ActionItem(
                step="Retake the photo with the leaf centered, in daylight, and in focus.",
            ),
            ActionItem(
                step="If a clearer photo still comes back unsure, ask a local agricultural officer.",
            ),
        ],
        monitoring_guidance="Photograph the same leaf every few days if symptoms spread.",
        diagnostic_limitations="No disease was chosen; treatment steps for a guessed class are omitted.",
        initial_message=message,
    )


def build_error_report() -> DiagnosisReport:
    message = "The disease model is unavailable right now. Please try again in a minute."
    return DiagnosisReport(
        status="error",
        status_label="Analysis unavailable",
        diagnosis_name="Unable to analyze",
        confidence=0.0,
        confidence_category="inconclusive",
        confidence_explanation="The disease model did not return a prediction.",
        description=message,
        expert_confirmation_required=True,
        actions=[ActionItem(step="Upload the same photo again shortly.")],
        diagnostic_limitations="No disease prediction was produced for this image.",
        initial_message=message,
    )


def build_unsupported_crop_report(hint: str) -> DiagnosisReport:
    """The farmer named a crop the classifier was never trained on: do not present a disease."""
    from app.services.crops import supported_crops

    crops = ", ".join(supported_crops())
    message = (
        f"The disease model was not trained on {hint.strip()}, so it cannot give a reliable "
        "diagnosis for this plant, and any disease name it returned would be a guess."
    )
    return DiagnosisReport(
        status="inconclusive",
        status_label="Crop not supported",
        diagnosis_name=f"{hint.strip().title()} is not covered by this model",
        crop=hint.strip().title(),
        confidence=0.0,
        confidence_category="inconclusive",
        confidence_explanation="The classifier has never seen this crop, so its score is not meaningful.",
        description=message,
        expert_confirmation_required=True,
        actions=[
            ActionItem(step="Ask a local agricultural extension officer or a plant clinic to look at the leaf."),
            ActionItem(step="Isolate the plant and avoid spraying anything until the cause is confirmed."),
        ],
        monitoring_guidance="Photograph the same leaf every few days to see whether symptoms spread.",
        diagnostic_limitations=f"Supported crops: {crops}.",
        initial_message=message,
    )


def apply_crop_mismatch(report: DiagnosisReport, hint: str, predicted_crop: str) -> DiagnosisReport:
    """Farmer's crop and predicted crop disagree: keep the result, but make it tentative."""
    note = (
        f"You said this is {hint.strip()}, but the model matched the photo to {predicted_crop}. "
        "Treat this result as uncertain."
    )
    report.status, report.status_label = "possible", "Possible match"
    if report.confidence_category in ("high", "moderate"):
        report.confidence_category = "low"
        report.confidence_explanation = _confidence_explanation("low")
    report.expert_confirmation_required = True
    report.description = f"{note} {report.description}"
    report.initial_message = f"{note} {report.initial_message}".strip()
    return report


def _build_label_only_report(
    disease_id: str,
    confidence: float,
    crop: str,
    top_k: list[dict],
    status: DiagnosisStatus,
    status_label: str,
    category: ConfidenceCategory,
) -> DiagnosisReport:
    healthy = is_healthy(disease_id)
    return DiagnosisReport(
        status=status,
        status_label=status_label,
        disease_id=disease_id,
        diagnosis_name=display_name(disease_id),
        crop=crop,
        confidence=round(confidence, 3),
        confidence_category=category,
        confidence_explanation=_confidence_explanation(category),
        description=(
            f"The leaf looks healthy for {crop}."
            if healthy
            else f"The model matched this photo to {display_name(disease_id)}."
        ),
        alternatives=_alternatives_from_top_k(disease_id, top_k),
        expert_confirmation_required=not healthy,
        actions=[
            ActionItem(step="Isolate or remove clearly affected leaves to slow spread.")
            if not healthy
            else ActionItem(step="Keep regular watering, spacing, and field hygiene."),
            ActionItem(step="Confirm with your local agricultural extension office before any chemical treatment."),
        ],
        monitoring_guidance="Check the plant every few days and upload a new photo if symptoms change.",
        diagnostic_limitations="No curated guidance for this class yet; advice is general.",
    )


def build_report_from_knowledge(
    disease_id: str,
    confidence: float,
    crop: str,
    top_k: list[dict] | None = None,
) -> DiagnosisReport:
    settings = get_settings()
    top_k = top_k or []
    cat = _confidence_category(confidence, settings.confidence_threshold)
    low = confidence < settings.confidence_threshold
    healthy = is_healthy(disease_id)
    if low:
        status: DiagnosisStatus = "possible"
        status_label = "Possible match"
    elif healthy:
        status, status_label = "healthy", "Looks healthy"
    else:
        status, status_label = "confirmed_likely", "Likely disease"

    entry = get_entry(disease_id)
    if not entry:
        report = _build_label_only_report(
            disease_id, confidence, crop or crop_from_class(disease_id), top_k, status, status_label, cat
        )
        report.initial_message = _compose_initial_message(report)
        return report

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
        alternatives=_alternatives_from_top_k(disease_id, top_k) or entry.get("alternatives", []),
        expert_confirmation_required=not healthy and (low or cat in ("low", "inconclusive")),
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


REPORT_SYSTEM = (
    "You write plain-language agricultural guidance for smallholder farmers. "
    "You receive a disease classifier's text output and curated reference notes; you never see the photo. "
    "Use only the facts given. Do not name specific chemical products or doses. "
    "If the model is not confident, say clearly that the result is uncertain and mention the other possibilities."
)


def build_report_prompt(
    report: DiagnosisReport,
    reference: dict,
    top_k: list[dict],
    low_confidence: bool,
    crop_hint: str = "",
    additional_symptoms: str = "",
    crop_note: str = "",
) -> str:
    predictions = "\n".join(
        f"- {t['display_name']}: {round(t['score'] * 100, 1)}%" for t in top_k
    ) or f"- {report.diagnosis_name}: {round(report.confidence * 100, 1)}%"
    lines = [
        "Write a short farmer-friendly summary (3-5 sentences).",
        "",
        "Classifier output (top predictions):",
        predictions,
        f"Confidence level: {report.confidence_category}"
        + (" (below the confidence threshold)" if low_confidence else ""),
        f"Diagnosis shown to farmer: {report.diagnosis_name}",
        f"Crop: {report.crop or 'unknown'}",
    ]
    if crop_hint.strip():
        lines.append(f"Farmer says the crop is: {crop_hint.strip()}")
    if crop_note:
        lines.append(f"Warning: {crop_note}")
    if additional_symptoms.strip():
        lines.append(f"Farmer-reported symptoms: {additional_symptoms.strip()}")
    if reference:
        lines += [
            "",
            "Curated reference:",
            f"Description: {report.description}",
            f"Actions: {[a.step for a in report.actions]}",
            f"Monitoring: {report.monitoring_guidance}",
            f"Limitations: {report.diagnostic_limitations}",
        ]
    else:
        lines += [
            "",
            "No curated reference exists for this class. Give only general, low-risk "
            "integrated pest management or plant-care steps and advise confirming with a local expert.",
        ]
    return "\n".join(lines)


def enrich_report_with_llm(
    report: DiagnosisReport,
    reference: dict,
    top_k: list[dict] | None = None,
    low_confidence: bool = False,
    crop_hint: str = "",
    additional_symptoms: str = "",
    crop_note: str = "",
) -> DiagnosisReport:
    """Farmer-facing summary from the classifier output; structured fields stay as built."""
    prompt = build_report_prompt(
        report, reference, top_k or [], low_confidence, crop_hint, additional_symptoms, crop_note
    )
    text = generate_text(REPORT_SYSTEM, prompt)
    if text:
        report.initial_message = text
    return report
