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


def _diagnosis_label(session_context: dict) -> str:
    crop = str(session_context.get("crop") or "").strip()
    name = str(session_context.get("diagnosis_name") or "").strip()
    if crop and name:
        # diagnosis_name often already starts with the crop ("Soybean — Healthy").
        if name.lower().startswith(crop.lower()):
            return name
        return f"{crop} — {name}"
    return name or crop or "your recent leaf check"


def _looks_healthy(session_context: dict) -> bool:
    status = str(session_context.get("status") or "").lower()
    name = str(session_context.get("diagnosis_name") or "").lower()
    return status == "healthy" or "healthy" in name


def _fallback_chat(
    message: str,
    session_context: dict,
    *,
    leaf_check: bool,
    llm_error: str | None = None,
) -> str:
    """Readable markdown when the LLM is off or fails. Never dump the full farmer summary."""
    if not leaf_check and not _is_agriculture_related(message):
        return (
            "I can only help with crop and plant health questions. "
            "Ask me about symptoms, care steps, or what to watch for in your field."
        )

    # Billing / auth failures: say that up front. Do not pretend the key is missing.
    if llm_error and llm_error != "no_key":
        label = _diagnosis_label(session_context)
        return (
            f"{llm_error}\n\n"
            f"**This leaf check:** {label}\n\n"
            f"**Your question:** {message.strip()}\n\n"
            "Until the model is reachable again, use the diagnosis card on this page "
            "for monitoring and care notes."
        )

    label = _diagnosis_label(session_context)
    question = message.strip()
    q = question.lower()
    healthy = _looks_healthy(session_context)
    about_spray = any(
        w in q for w in ("fungicide", "pesticide", "herbicide", "spray", "chemical", "dose")
    )

    if healthy and about_spray:
        answer = (
            "No spray is needed for a healthy leaf reading. "
            "Keep checking a few plants across the field; if new spots appear, "
            "confirm any treatment with your local extension office before you spray."
        )
    elif about_spray:
        answer = (
            "Do not choose a product or dose from this demo alone. "
            "Match symptoms for a few days and confirm any chemical treatment "
            "with your local agricultural extension office."
        )
    elif healthy:
        answer = (
            "Keep good field habits: quality seed, rotation where you can, "
            "and check more than one plant. A single healthy leaf does not rule out "
            "problems elsewhere in the field."
        )
    else:
        answer = (
            "Continue monitoring affected leaves every few days, avoid working plants "
            "when wet, and confirm any chemical treatment with your local agricultural "
            "extension office."
        )

    footer = (
        "_Plain follow-up only — set `LLM_API_KEY` in `.env` for richer chat answers._"
        if llm_error == "no_key"
        else "_Plain follow-up only._"
    )
    return (
        f"**This leaf check:** {label}\n\n"
        f"**Your question:** {question}\n\n"
        f"{answer}\n\n"
        f"{footer}"
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
    llm_reply, llm_error = chat_with_context(
        msg, context, format_history(history), reply_instruction=instruction
    )
    if llm_reply:
        return ChatResponse(reply=llm_reply)

    return ChatResponse(
        reply=_fallback_chat(
            msg, session_context, leaf_check=leaf_check, llm_error=llm_error
        )
    )
