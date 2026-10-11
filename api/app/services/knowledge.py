import json
from functools import lru_cache
from pathlib import Path

from app.config import get_settings


@lru_cache
def _load_knowledge() -> dict:
    settings = get_settings()
    path = Path(settings.knowledge_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / settings.knowledge_path
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_entry(disease_id: str) -> dict | None:
    return _load_knowledge().get(disease_id)


def list_disease_ids() -> list[str]:
    return list(_load_knowledge().keys())
