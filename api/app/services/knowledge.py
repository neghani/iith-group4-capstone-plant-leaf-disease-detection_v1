import json
from functools import lru_cache
from pathlib import Path

from app.config import get_settings


def _resolve_knowledge_path() -> Path:
    settings = get_settings()
    # Prefer the release pin's knowledge_file when present.
    from shared.release import as_dict

    pin = as_dict()
    relative = pin.get("knowledge_file") or settings.knowledge_path
    path = Path(relative)
    if not path.is_absolute():
        # knowledge lives under api/data/... ; release paths are relative to api/.
        api_root = Path(__file__).resolve().parents[2]
        path = api_root / path
    return path


@lru_cache
def _load_knowledge() -> dict:
    path = _resolve_knowledge_path()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_entry(disease_id: str) -> dict | None:
    return _load_knowledge().get(disease_id)


def list_disease_ids() -> list[str]:
    return list(_load_knowledge().keys())


def assert_labels_match_knowledge(label_names: list[str]) -> None:
    """Refuse to start when the classifier label set and the knowledge file disagree."""
    knowledge_ids = set(list_disease_ids())
    model_ids = set(label_names)
    if knowledge_ids != model_ids:
        only_model = sorted(model_ids - knowledge_ids)[:5]
        only_knowledge = sorted(knowledge_ids - model_ids)[:5]
        raise RuntimeError(
            "Model labels do not match the knowledge base. "
            f"model={len(model_ids)} knowledge={len(knowledge_ids)}. "
            f"only_in_model={only_model} only_in_knowledge={only_knowledge}. "
            "Check shared/release.json knowledge_file against the Hub id2label."
        )


def curated_reference(disease_id: str) -> dict:
    """Entry for the LLM prompt. Stubs (curated=false) return empty so the prompt stays honest."""
    entry = get_entry(disease_id) or {}
    if not entry:
        return {}
    if entry.get("curated", True):
        return entry
    return {}
