import logging
import traceback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings

_log = logging.getLogger("leafcheck.llm")


def llm_configured() -> bool:
    settings = get_settings()
    return bool(settings.llm_api_key.strip())


def get_chat_model(temperature: float = 0.2) -> ChatOpenAI | None:
    settings = get_settings()
    if not settings.llm_api_key.strip():
        return None

    base_url = settings.llm_base_url.strip() or None
    print(f"\n[LLM ROUTE CHECK] Model: {settings.llm_model} | Base URL: {base_url}", flush=True)

    kwargs: dict = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "temperature": temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["openai_api_base"] = base_url

    return ChatOpenAI(**kwargs)


AGRICULTURE_SYSTEM = (
    "You are a plant disease assistant for farmers, in one ongoing conversation about a single leaf check. "
    "Every question is about the crop and disease in Diagnosis context, including short follow-ups "
    "such as sunlight, water, weather, spread, or 'how will this disease react'. "
    "Read the earlier turns and answer the latest question as a continuation of them. "
    "Resolve 'it', 'that', 'the disease', and misspellings such as 'decease' from those turns and the diagnosis. "
    "Answer directly. Do not greet, do not introduce yourself, and do not ask the farmer to start over or restate the crop. "
    "Decline in one sentence only when the question is clearly unrelated to plants, crops, or this leaf check. "
    "Do not invent chemical products; prefer general integrated pest management guidance "
    "and remind users to confirm with local extension services for critical decisions. "
    "The diagnosis comes from an image classifier; if its confidence is low, say the result is uncertain."
)


def history_messages(history: str) -> list:
    """Prior turns as chat messages, so a follow-up is the next turn and not a new chat."""
    messages = []
    for line in history.splitlines():
        if line.startswith("Farmer: "):
            messages.append(HumanMessage(content=line.removeprefix("Farmer: ")))
        elif line.startswith("Assistant: "):
            messages.append(AIMessage(content=line.removeprefix("Assistant: ")))
    return messages


def build_chat_messages(
    user_message: str,
    context_summary: str,
    history: str = "",
    reply_instruction: str = "",
) -> list:
    system = AGRICULTURE_SYSTEM
    if reply_instruction.strip():
        system = f"{system}\n\n{reply_instruction.strip()}"
    # Language notes stay on the system message. They must not sit between
    # earlier turns and the latest question, or the model answers the note and drops the thread.
    current = (
        f"Diagnosis context:\n{context_summary}\n\n"
        "Continue this same conversation. Apply the latest question to this leaf check "
        "and to what was already said. Do not greet.\n\n"
        f"Farmer question:\n{user_message}"
    )
    return [
        SystemMessage(content=system),
        *history_messages(history),
        HumanMessage(content=current),
    ]


def _invoke(messages: list) -> str | None:
    for message in messages:
        if not isinstance(message.content, str):
            raise TypeError("LLM accepts text only; images go to the Hugging Face classifier.")
    model = get_chat_model()
    if model is None:
        print("\n>>> [LLM STATUS]: model is None (check .env LLM_API_KEY)\n", flush=True)
        return None
    try:
        response = model.invoke(messages)
        content = response.content
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
    except Exception as exc:
        print(f"\n{'='*50}", flush=True)
        print(f">>> [LLM CALL FAILED]: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        print(f"{'='*50}\n", flush=True)
        return None


def generate_text(system: str, user: str) -> str | None:
    if not isinstance(system, str) or not isinstance(user, str):
        raise TypeError("LLM accepts text only; images go to the Hugging Face classifier.")
    return _invoke([SystemMessage(content=system), HumanMessage(content=user)])


def chat_with_context(
    user_message: str,
    context_summary: str,
    history: str = "",
    reply_instruction: str = "",
) -> str | None:
    return _invoke(
        build_chat_messages(user_message, context_summary, history, reply_instruction)
    )
