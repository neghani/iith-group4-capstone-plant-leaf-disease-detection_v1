"""Declares, per response field, which component produced it.

The pipeline has four very different writers and tutors/reviewers could not tell
them apart from the JSON alone: OpenCV quality checks, the Hugging Face
classifier, curated knowledge-base notes, and the LLM (which only ever writes
the farmer summary). `build_provenance` turns that pipeline into an explicit
`provenance` list on /predict so each field names its source.
"""

from app.schemas import (
    DiagnosisReport,
    ImageQualityResult,
    InferenceInfo,
    ProvenanceEntry,
    TopKPrediction,
)

QUALITY_LABEL = "OpenCV image-quality checks"
CROP_LABEL = "Leaf detection & cropping (OpenCV)"
RULES_LABEL = "Threshold decision rules"
REFERENCE_LABEL = "Knowledge base"
FARMER_LABEL = "Farmer input (form fields)"
TEMPLATE_LABEL = "App text template"

QUALITY_DETAILS: dict[str, str] = {
    "brightness": "Mean pixel brightness (0-255) measured after decoding the photo.",
    "blur_score": "Laplacian-variance sharpness score, 0 (blurry) to 1 (sharp).",
    "leaf_found": "Green-leaf segmentation decided whether any leaf is visible.",
    "leaf_coverage": "Share of the frame covered by the detected leaf region.",
    "original_size": "Pixel dimensions of the decoded photo.",
    "reason": "Fixed rule (dark/small/blurry) that failed the photo.",
    "message": "Fixed rule (dark/small/blurry) that failed the photo.",
}


def _classifier_label(inference: InferenceInfo) -> str:
    verified = "" if inference.model_verified else " (unverified)"
    return f"Image classifier ({inference.model_id} {inference.model_revision}){verified}"


def build_provenance(
    report: DiagnosisReport,
    *,
    top_k: list[TopKPrediction] | None = None,
    inference: InferenceInfo | None = None,
    quality: ImageQualityResult | None = None,
    llm_used: bool = False,
    has_reference: bool = False,
    crop_hint: str = "",
    additional_symptoms: str = "",
) -> list[ProvenanceEntry]:
    top_k = top_k or []
    entries: list[ProvenanceEntry] = []

    # 1. What the OpenCV quality gate measured on the raw photo.
    if quality is not None:
        dumped = quality.model_dump()
        for key, detail in QUALITY_DETAILS.items():
            if dumped.get(key) is not None:
                entries.append(
                    ProvenanceEntry(
                        field=f"quality.{key}", source="image_quality_checks", label=QUALITY_LABEL, detail=detail
                    )
                )

    # 2. The leaf box / crop strategy that produced the classifier's input image.
    if inference is not None and inference.preprocessing:
        strategy = inference.preprocessing.get("crop_strategy") or {}
        n_crops = strategy.get("n_crops")
        detail = "Picked the sharpest green region and built the JPEG the classifier received."
        if n_crops:
            detail += f" Crop strategy '{strategy.get('strategy')}' scored {n_crops} candidate crop(s)."
        entries.append(
            ProvenanceEntry(
                field="inference.preprocessing", source="image_quality_checks", label=CROP_LABEL, detail=detail
            )
        )

    # 3. What the image classifier itself produced.
    if inference is not None:
        label = _classifier_label(inference)
        if report.disease_id:
            entries.append(
                ProvenanceEntry(
                    field="report.disease_id",
                    source="image_classifier",
                    label=label,
                    detail=f"Top-1 class label of {len(top_k)} scored classes ({round(report.confidence * 100, 1)}%).",
                )
            )
            entries.append(
                ProvenanceEntry(
                    field="report.confidence",
                    source="image_classifier",
                    label=label,
                    detail="Top-1 softmax probability, rounded for display.",
                )
            )
        if top_k:
            entries.append(
                ProvenanceEntry(
                    field="top_k",
                    source="image_classifier",
                    label=label,
                    detail="All class scores the model returned, highest first — not filtered or rewritten.",
                )
            )
        entries.append(
            ProvenanceEntry(
                field="inference",
                source="image_classifier",
                label=label,
                detail="Proof of which model served the request: model id/revision, latency, and the exact input image.",
            )
        )

    # 4. Deterministic rules that turn the score into farmer-facing status text.
    entries.append(
        ProvenanceEntry(
            field="report.status",
            source="decision_rules",
            label=RULES_LABEL,
            detail="Confidence threshold and top-2 margin mapped the score to a status; no model involved.",
        )
    )
    entries.append(
        ProvenanceEntry(
            field="report.confidence_category",
            source="decision_rules",
            label=RULES_LABEL,
            detail="Fixed thresholds bucket the classifier score into high/moderate/low/inconclusive.",
        )
    )
    entries.append(
        ProvenanceEntry(
            field="report.confidence_explanation",
            source="decision_rules",
            label=RULES_LABEL,
            detail="One canned sentence per confidence bucket, chosen by the same thresholds.",
        )
    )
    if report.alternatives:
        entries.append(
            ProvenanceEntry(
                field="report.alternatives",
                source="decision_rules",
                label=RULES_LABEL,
                detail="Runner-up classes above a minimum score, formatted as 'Name (N%)'.",
            )
        )
    entries.append(
        ProvenanceEntry(
            field="report.expert_confirmation_required",
            source="decision_rules",
            label=RULES_LABEL,
            detail="Safety rule: set when confidence is low or the farmer's crop disagrees with the model.",
        )
    )

    # 5. Curated reference notes — written ahead of time, never generated per photo.
    if has_reference:
        description_detail = "Knowledge-base entry for this disease, stored before any photo was uploaded."
        if additional_symptoms.strip():
            description_detail += " The farmer's extra symptoms are appended verbatim."
        entries.extend(
            [
                ProvenanceEntry(
                    field="report.description",
                    source="curated_reference",
                    label=REFERENCE_LABEL,
                    detail=description_detail,
                ),
                ProvenanceEntry(
                    field="report.symptoms",
                    source="curated_reference",
                    label=REFERENCE_LABEL,
                    detail="Each symptom row carries evidence_source='reference' for exactly this reason.",
                ),
                ProvenanceEntry(
                    field="report.actions",
                    source="curated_reference",
                    label=REFERENCE_LABEL,
                    detail="Treatment steps reviewed by the team; the LLM never writes them.",
                ),
            ]
        )
        if report.monitoring_guidance:
            entries.append(
                ProvenanceEntry(
                    field="report.monitoring_guidance",
                    source="curated_reference",
                    label=REFERENCE_LABEL,
                    detail="Knowledge-base entry for this disease.",
                )
            )
        if report.diagnostic_limitations:
            entries.append(
                ProvenanceEntry(
                    field="report.diagnostic_limitations",
                    source="curated_reference",
                    label=REFERENCE_LABEL,
                    detail="Knowledge-base limitations note, plus a fixed safety disclaimer appended by rules.",
                )
            )

    # 6. Optional text the farmer typed; the classifier never sees it.
    if crop_hint.strip():
        entries.append(
            ProvenanceEntry(
                field="request.crop_hint",
                source="farmer_input",
                label=FARMER_LABEL,
                detail="Used only to warn about crop mismatches; it cannot change the classifier's scores.",
            )
        )
    if additional_symptoms.strip():
        entries.append(
            ProvenanceEntry(
                field="request.additional_symptoms",
                source="farmer_input",
                label=FARMER_LABEL,
                detail="Appended to the description and shown to the LLM; not seen by the classifier.",
            )
        )

    # 7. The one field the LLM writes — or the template fallback when it did not run.
    if llm_used:
        entries.append(
            ProvenanceEntry(
                field="report.initial_message",
                source="llm_summary",
                label="LLM (text-only summary)",
                detail="Rewrites the classifier output and reference notes in plain language; the photo is never sent to the LLM.",
            )
        )
    else:
        entries.append(
            ProvenanceEntry(
                field="report.initial_message",
                source="template_text",
                label=TEMPLATE_LABEL,
                detail="Assembled from the status and actions by a fixed template (LLM unavailable or not needed).",
            )
        )
    if report.initial_message:
        entries.append(
            ProvenanceEntry(
                field="session_context.summary",
                source="llm_summary" if llm_used else "template_text",
                label="LLM (text-only summary)" if llm_used else TEMPLATE_LABEL,
                detail="The same text reused as the chat panel's opening context.",
            )
        )

    return entries