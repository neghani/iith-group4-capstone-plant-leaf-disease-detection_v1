from app.llm.provider import chat_with_context
from app.schemas import ChatResponse


AGRICULTURE_KEYWORDS = (
    "plant",
    "crop",
    "leaf",
    "disease",
    "blight",
    "rust",
    "fung",
    "pest",
    "tomato",
    "potato",
    "corn",
    "apple",
    "grape",
    "peach",
    "pepper",
    "strawberr",
    "squash",
    "soybean",
    "cherry",
    "orange",
    "citrus",
    "blueberr",
    "raspberr",
    "maize",
    "mildew",
    "mold",
    "mite",
    "virus",
    "bacteri",
    "spot",
    "healthy",
    "spray",
    "fertil",
    "harvest",
    "field",
    "farm",
    "soil",
    "water",
    "irrigation",
    "treatment",
    "symptom",
)


def format_session_context(session_context: dict) -> str:
    """Plain-text diagnosis context for the LLM (classifier output only, never image data)."""
    if not session_context:
        return "No diagnosis yet."
    lines = []
    if session_context.get("diagnosis_name"):
        lines.append(f"Diagnosis: {session_context['diagnosis_name']}")
    if session_context.get("crop"):
        lines.append(f"Crop: {session_context['crop']}")
    if session_context.get("confidence") is not None:
        conf = float(session_context["confidence"])
        uncertain = " (low confidence, uncertain)" if session_context.get("low_confidence") else ""
        lines.append(f"Model confidence: {round(conf * 100, 1)}%{uncertain}")
    top_k = session_context.get("top_k") or []
    if top_k:
        preds = ", ".join(
            f"{t.get('display_name', t.get('label'))} {round(float(t.get('score', 0)) * 100, 1)}%"
            for t in top_k
        )
        lines.append(f"Top predictions: {preds}")
    if session_context.get("summary"):
        lines.append(f"Summary given to farmer: {session_context['summary']}")
    return "\n".join(lines)


def _is_agriculture_related(message: str) -> bool:
    text = message.lower()
    return any(k in text for k in AGRICULTURE_KEYWORDS)


def _fallback_chat(message: str, context_summary: str) -> str:
    if not _is_agriculture_related(message):
        return (
            "I can only help with crop and plant health questions. "
            "Ask me about symptoms, care steps, or what to watch for in your field."
        )
    return (
        f"Based on your current diagnosis ({context_summary}), "
        "continue monitoring affected leaves every few days, avoid working in wet conditions, "
        "and confirm any chemical treatment with your local agricultural extension office. "
        f"Regarding your question: {message.strip()} — "
        "for this demo, enable an LLM API key for richer follow-up answers."
    )


def handle_chat(message: str, session_context: dict) -> ChatResponse:
    msg = message.strip()
    if not msg:
        return ChatResponse(reply="Please type a question about your crop or diagnosis.")

    summary = session_context.get("summary") or session_context.get("diagnosis_name") or "recent upload"
    if not _is_agriculture_related(msg):
        return ChatResponse(
            reply=(
                "I focus on agriculture and plant health. "
                "Please ask about the disease, symptoms, or field actions."
            ),
            declined=True,
        )

    llm_reply = chat_with_context(msg, format_session_context(session_context))
    if llm_reply:
        return ChatResponse(reply=llm_reply)

    return ChatResponse(reply=_fallback_chat(msg, summary))
