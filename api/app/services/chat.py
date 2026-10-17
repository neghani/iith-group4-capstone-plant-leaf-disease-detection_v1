from app.llm.provider import chat_with_context
from app.schemas import ChatResponse
from app.services.translate import normalize_language, telugu_reply_note


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


def _clip(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def has_leaf_check(session_context: dict) -> bool:
    return bool(
        session_context.get("diagnosis_name")
        or session_context.get("disease_id")
        or session_context.get("crop")
    )


def format_session_context(session_context: dict) -> str:
    """Short leaf-check text for every chat prompt. Classifier output only, never image data."""
    if not session_context:
        return "No diagnosis yet."
    lines = []
    crop = session_context.get("crop")
    name = session_context.get("diagnosis_name")
    if crop or name:
        head = "Leaf check: "
        if crop:
            head += f"{crop} leaf"
        if name:
            head += f" — {name}" if crop else str(name)
        if session_context.get("confidence") is not None:
            conf = float(session_context["confidence"])
            uncertain = ", low confidence" if session_context.get("low_confidence") else ""
            head += f" ({round(conf * 100, 1)}%{uncertain})"
        lines.append(head)
    top_k = session_context.get("top_k") or []
    if top_k:
        preds = ", ".join(
            f"{t.get('display_name', t.get('label'))} {round(float(t.get('score', 0)) * 100, 1)}%"
            for t in top_k[:3]
        )
        lines.append(f"Top predictions: {preds}")
    if session_context.get("summary"):
        lines.append(f"Summary given to farmer: {_clip(session_context['summary'], 280)}")
    return "\n".join(lines) or "No diagnosis yet."


def format_history(history: list | None) -> str:
    """Last few turns so a follow-up is not treated as a new, context-free question."""
    if not history:
        return ""
    lines = []
    for turn in history[-6:]:
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content")
        else:
            role, content = getattr(turn, "role", None), getattr(turn, "content", None)
        if role not in ("user", "assistant") or not content:
            continue
        speaker = "Farmer" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {_clip(content, 400)}")
    return "\n".join(lines)


def _is_agriculture_related(message: str) -> bool:
    text = message.lower()
    return any(k in text for k in AGRICULTURE_KEYWORDS)


def _fallback_chat(message: str, context_summary: str, *, leaf_check: bool) -> str:
    if not leaf_check and not _is_agriculture_related(message):
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


def handle_chat(
    message: str,
    session_context: dict,
    history: list | None = None,
    language: str = "en",
) -> ChatResponse:
    msg = message.strip()
    if not msg:
        return ChatResponse(reply="Please type a question about your crop or diagnosis.")

    leaf_check = has_leaf_check(session_context)
    # A follow-up often omits the crop and disease ("Should I remove them?").
    # Decline only when this browser session has no leaf check to attach.
    if not leaf_check and not _is_agriculture_related(msg):
        return ChatResponse(
            reply=(
                "I focus on agriculture and plant health. "
                "Please ask about the disease, symptoms, or field actions."
            ),
            declined=True,
        )

    context = format_session_context(session_context)
    instruction = telugu_reply_note() if normalize_language(language) == "te" else ""
    llm_reply = chat_with_context(
        msg, context, format_history(history), reply_instruction=instruction
    )
    if llm_reply:
        return ChatResponse(reply=llm_reply)

    summary = session_context.get("summary") or session_context.get("diagnosis_name") or "recent upload"
    return ChatResponse(reply=_fallback_chat(msg, summary, leaf_check=leaf_check))
