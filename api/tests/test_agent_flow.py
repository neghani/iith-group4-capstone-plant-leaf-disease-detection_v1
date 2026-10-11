"""Proves the Hugging Face prediction is what drives the agentic flow, end to end.

predict (HF) -> knowledge lookup -> report -> LLM prompt -> safety -> chat context.
Every test changes only the classifier output (or crop hint) and checks the effect downstream.
"""

import json
from pathlib import Path

import pytest
from conftest import _upload

from app.services.crops import resolve_hint, same_crop
from app.services.knowledge import list_disease_ids
from app.vision.hf_client import LabelScore

# id2label of the Hub classifier (config.json; 38 classes, unchanged across v1.0/v1.1)
MODEL_LABELS = json.loads(
    (Path(__file__).parent / "model_labels.json").read_text(encoding="utf-8")
)


def test_every_model_label_has_curated_knowledge():
    assert len(MODEL_LABELS) == 38
    assert sorted(MODEL_LABELS) == sorted(list_disease_ids())


@pytest.mark.parametrize(
    "label,expected_name",
    [
        ("Tomato___Late_blight", "Tomato Late Blight"),
        ("Apple___Black_rot", "Apple Black Rot"),
        ("Potato___healthy", None),
    ],
)
def test_classifier_output_drives_report_and_llm(client, good_leaf, hf_calls, llm_prompts, label, expected_name):
    hf_calls.scores = [LabelScore(label, 0.93), LabelScore("Grape___healthy", 0.03)]
    body = client.post("/predict", files=_upload(good_leaf)).json()
    report = body["report"]

    assert report["disease_id"] == label
    assert report["crop"].lower().startswith(label.split("___")[0].split("_")[0].lower())
    if expected_name:
        assert report["diagnosis_name"].lower().replace("-", " ") == expected_name.lower()
    # knowledge entry for THIS label was used, and the LLM saw THIS label (text only)
    assert report["actions"], "curated actions missing"
    system, user = llm_prompts[0]
    assert report["diagnosis_name"] in user
    assert "93.0%" in user
    # follow-up chat is grounded in the same prediction
    assert body["session_context"]["disease_id"] == label
    assert body["session_context"]["top_k"][0]["label"] == label


def test_different_predictions_produce_different_guidance(client, good_leaf, hf_calls, llm_prompts):
    hf_calls.scores = [LabelScore("Tomato___Late_blight", 0.95)]
    first = client.post("/predict", files=_upload(good_leaf)).json()["report"]
    hf_calls.scores = [LabelScore("Corn_(maize)___Common_rust_", 0.95)]
    second = client.post("/predict", files=_upload(good_leaf)).json()["report"]
    assert first["diagnosis_name"] != second["diagnosis_name"]
    assert [a["step"] for a in first["actions"]] != [a["step"] for a in second["actions"]]


def test_response_proves_which_hf_model_answered(client, good_leaf, hf_calls, llm_prompts):
    info = client.post("/predict", files=_upload(good_leaf)).json()["inference"]
    from shared.release import HF_MODEL_ID, HF_MODEL_REVISION

    assert info["model_id"] == HF_MODEL_ID
    assert info["model_revision"] == HF_MODEL_REVISION
    assert info["model_verified"] is True  # reported by the serving side, not just our config
    assert info["input_bytes"] == len(hf_calls[0])


def test_no_hf_means_no_fabricated_diagnosis(client, good_leaf, monkeypatch, llm_prompts):
    from app.vision.hf_client import HFInferenceError

    def down(*_a, **_k):
        raise HFInferenceError("down")

    monkeypatch.setattr("app.vision.predictor.classify_image", down)
    body = client.post("/predict", files=_upload(good_leaf)).json()
    assert body["report"]["status"] == "error"
    assert body["report"]["disease_id"] is None
    assert body["inference"] is None
    assert llm_prompts == []


# ---- crop hint vs. prediction ------------------------------------------------------------


@pytest.mark.parametrize(
    "hint,crop",
    [
        ("corn", "Corn (maize)"),
        ("Maize", "Corn (maize)"),
        ("bell pepper", "Pepper bell"),
        ("capsicum", "Pepper bell"),
        ("tomatoes", "Tomato"),
        ("Potatoes", "Potato"),
    ],
)
def test_hint_resolution(hint, crop):
    assert resolve_hint(hint) == crop


@pytest.mark.parametrize("hint", ["hibiscus", "rose", "banana", "   "])
def test_hint_outside_model_is_unsupported(hint):
    assert resolve_hint(hint) is None


def test_same_crop_uses_aliases():
    assert same_crop("Corn (maize)", "Maize")
    assert not same_crop("Tomato", "Potato")


def test_matching_hint_keeps_confident_result(client, good_leaf, hf_calls, llm_prompts):
    body = client.post("/predict", files=_upload(good_leaf), data={"crop": "tomato"}).json()
    assert body["report"]["status"] == "confirmed_likely"
    assert body["low_confidence"] is False
    assert "Warning" not in llm_prompts[0][1]


def test_mismatching_hint_downgrades_and_tells_the_llm(client, good_leaf, hf_calls, llm_prompts):
    body = client.post("/predict", files=_upload(good_leaf), data={"crop": "potato"}).json()
    report = body["report"]
    assert report["disease_id"] == "Tomato___Early_blight"  # prediction still reported
    assert report["status"] == "possible"
    assert report["expert_confirmation_required"] is True
    assert "You said this is potato" in report["description"]
    assert body["low_confidence"] is True
    assert "the farmer says the crop is potato but the classifier matched Tomato" in llm_prompts[0][1]


def test_unsupported_crop_never_gets_a_disease_name(client, good_leaf, hf_calls, llm_prompts):
    body = client.post("/predict", files=_upload(good_leaf), data={"crop": "hibiscus"}).json()
    report = body["report"]
    assert hf_calls, "classifier is still called (its output is logged), but not trusted"
    assert report["status"] == "inconclusive"
    assert report["status_label"] == "Crop not supported"
    assert report["disease_id"] is None
    assert "Tomato" not in report["diagnosis_name"]
    assert report["alternatives"] == []
    assert body["low_confidence"] is True
    assert llm_prompts == []  # no LLM write-up of a guess
    assert body["session_context"]["disease_id"] is None
