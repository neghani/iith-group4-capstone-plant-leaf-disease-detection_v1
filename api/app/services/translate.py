"""Translate a finished English diagnosis into Telugu. English stays the source of truth."""

import json
from functools import lru_cache
from pathlib import Path

from app.llm.provider import generate_text, llm_configured
from app.schemas import ActionItem, DiagnosisReport, SymptomItem

_GLOSSARY_PATH = Path(__file__).resolve().parents[2] / "data" / "glossary_en_te.json"

_STRING_FIELDS = (
    "status_label",
    "diagnosis_name",
    "confidence_explanation",
    "description",
    "monitoring_guidance",
    "diagnostic_limitations",
    "initial_message",
)

TRANSLATE_SYSTEM = (
    "You translate farmer-facing plant-disease guidance from English into Telugu. "
    "Return only a JSON object with the same keys and the same list lengths as the input. "
    "Do not add diseases, chemical products, doses, or advice that is not in the source. "
    "When an English glossary term appears, in any capitalisation, use its exact Telugu form."
)


def normalize_language(value: str | None) -> str:
    return "te" if (value or "").strip().lower() == "te" else "en"


@lru_cache
def load_glossary() -> dict[str, str]:
    with _GLOSSARY_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): str(text) for key, text in data.items()}


def glossary_prompt() -> str:
    lines = [f"{english} = {telugu}" for english, telugu in load_glossary().items()]
    return "Glossary (use these Telugu forms exactly):\n" + "\n".join(lines)


def telugu_reply_note() -> str:
    return (
        "Reply in Telugu. Do not invent chemical products or doses. "
        "When one of these English terms applies, use the Telugu form exactly.\n"
        + glossary_prompt()
    )


def _payload(report: DiagnosisReport) -> dict:
    return {
        "status_label": report.status_label,
        "diagnosis_name": report.diagnosis_name,
        "confidence_explanation": report.confidence_explanation,
        "description": report.description,
        "symptoms": [
            {"name": item.name, "explanation": item.explanation} for item in report.symptoms
        ],
        "alternatives": list(report.alternatives),
        "actions": [
            {"step": item.step, "safety_note": item.safety_note} for item in report.actions
        ],
        "monitoring_guidance": report.monitoring_guidance,
        "diagnostic_limitations": report.diagnostic_limitations,
        "initial_message": report.initial_message,
    }


def _parse_json(text: str) -> dict | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        fence = raw.rfind("```")
        if fence >= 0:
            raw = raw[:fence]
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _apply(report: DiagnosisReport, data: dict) -> DiagnosisReport | None:
    updates: dict = {}
    for key in _STRING_FIELDS:
        text = _text(data.get(key))
        if text is None:
            return None
        updates[key] = text

    symptoms = data.get("symptoms")
    if not isinstance(symptoms, list) or len(symptoms) != len(report.symptoms):
        return None
    new_symptoms: list[SymptomItem] = []
    for source, item in zip(report.symptoms, symptoms):
        if not isinstance(item, dict):
            return None
        name, explanation = _text(item.get("name")), _text(item.get("explanation"))
        if name is None or explanation is None:
            return None
        new_symptoms.append(source.model_copy(update={"name": name, "explanation": explanation}))
    updates["symptoms"] = new_symptoms

    alternatives = data.get("alternatives")
    if not isinstance(alternatives, list) or len(alternatives) != len(report.alternatives):
        return None
    if not all(isinstance(item, str) and item.strip() for item in alternatives):
        if alternatives:
            return None
    updates["alternatives"] = [item.strip() for item in alternatives]

    actions = data.get("actions")
    if not isinstance(actions, list) or len(actions) != len(report.actions):
        return None
    new_actions: list[ActionItem] = []
    for source, item in zip(report.actions, actions):
        if not isinstance(item, dict):
            return None
        step = _text(item.get("step"))
        if step is None:
            return None
        note = item.get("safety_note")
        if note is not None and not isinstance(note, str):
            return None
        new_actions.append(
            source.model_copy(
                update={"step": step, "safety_note": note.strip() if isinstance(note, str) and note.strip() else None}
            )
        )
    updates["actions"] = new_actions
    return report.model_copy(update=updates)


def translate_report(report: DiagnosisReport) -> tuple[DiagnosisReport, bool]:
    """Return the report in Telugu, or the original English report if translation cannot run."""
    if not llm_configured():
        return report, False
    user = f"{glossary_prompt()}\n\nJSON:\n{json.dumps(_payload(report), ensure_ascii=False)}"
    text = generate_text(TRANSLATE_SYSTEM, user)
    if not text:
        return report, False
    data = _parse_json(text)
    if data is None:
        return report, False
    translated = _apply(report, data)
    if translated is None:
        return report, False
    return translated, True
