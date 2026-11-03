"""Knowledge file must match the pinned label list; rollback loads the 38-class file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import knowledge
from app.services.knowledge import assert_labels_match_knowledge


def test_live_knowledge_matches_model_labels():
    labels = json.loads((Path(__file__).parent / "model_labels.json").read_text(encoding="utf-8"))
    assert len(labels) == 38
    assert_labels_match_knowledge(labels)
    assert len(labels) == len(knowledge.list_disease_ids())
    entry = knowledge.get_entry("Tomato___Early_blight")
    assert entry and entry.get("curated") is True


def test_masterdataset_knowledge_matches_its_labels(monkeypatch):
    api_root = Path(__file__).resolve().parents[1]
    master_knowledge = api_root / "data" / "knowledge" / "diseases.json"
    master_labels = Path(__file__).parent / "model_labels.masterdataset.json"
    assert master_knowledge.is_file() and master_labels.is_file()
    labels = json.loads(master_labels.read_text(encoding="utf-8"))
    assert len(labels) == 86

    knowledge._load_knowledge.cache_clear()
    monkeypatch.setattr(
        knowledge,
        "_resolve_knowledge_path",
        lambda: master_knowledge,
    )
    knowledge._load_knowledge.cache_clear()
    assert_labels_match_knowledge(labels)
    knowledge._load_knowledge.cache_clear()


def test_guard_rejects_mismatched_labels():
    with pytest.raises(RuntimeError, match="do not match"):
        assert_labels_match_knowledge(["not_a_real_class"])
