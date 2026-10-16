import logging
import traceback
from langchain_core.messages import HumanMessage, SystemMessage
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
    "You are a plant disease assistant for farmers. "
    "Answer only agriculture, crop health, and plant disease questions. "
    "If the user asks about unrelated topics, politely decline in one sentence "
    "and invite an agriculture-related question. "
    "Do not invent chemical products; prefer general integrated pest management guidance "
    "and remind users to confirm with local extension services for critical decisions. "
    "The diagnosis comes from an image classifier; if its confidence is low, say the result is uncertain. "
    "Every question is about the same leaf check in Diagnosis context. "
    "Use that crop and disease on follow-ups even when the farmer does not repeat those names."
)


def generate_text(system: str, user: str) -> str | None:
    if not isinstance(system, str) or not isinstance(user, str):
        raise TypeError("LLM accepts text only; images go to the Hugging Face classifier.")
    model = get_chat_model()
    if model is None:
        print("\n>>> [LLM STATUS]: model is None (check .env LLM_API_KEY)\n", flush=True)
        return None
    try:
        response = model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
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


def chat_with_context(
    user_message: str, context_summary: str, history: str = ""
) -> str | None:
    parts = [f"Diagnosis context:\n{context_summary}"]
    if history.strip():
        parts.append(f"Recent conversation:\n{history.strip()}")
    parts.append(f"Farmer question:\n{user_message}")
    return generate_text(AGRICULTURE_SYSTEM, "\n\n".join(parts))
