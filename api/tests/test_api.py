from io import BytesIO

import pytest
from PIL import Image

from app.vision import hf_client
from app.vision.hf_client import HFInferenceError, LabelScore
from conftest import _upload


def test_health_reports_model(client):
    body = client.get("/health").json()
    assert body["hf_model_id"] == "neghani/plantvillage-mobilenetv2"
    assert body["hf_endpoint_configured"] is True
    assert body["hf_model_configured"] is True


def test_health_requires_endpoint(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("HF_ENDPOINT_URL", "")
    get_settings.cache_clear()
    body = client.get("/health").json()
    assert body["hf_model_configured"] is False
    assert body["hf_endpoint_configured"] is False


def test_quality_check_pass(client, good_leaf, hf_calls):
    body = client.post("/quality-check", files=_upload(good_leaf)).json()
    assert body["ok"] is True
    assert hf_calls == []


@pytest.mark.parametrize("fixture,needle", [("dark_leaf", "dark"), ("tiny_leaf", "small")])
def test_quality_check_fail(client, request, fixture, needle, hf_calls):
    body = client.post("/quality-check", files=_upload(request.getfixturevalue(fixture))).json()
    assert body["ok"] is False
    assert needle in body["message"].lower()
    assert hf_calls == []


def test_quality_check_rejects_bad_type(client):
    res = client.post("/quality-check", files={"file": ("a.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_predict_blocked_by_gate(client, dark_leaf, hf_calls, llm_prompts):
    body = client.post("/predict", files=_upload(dark_leaf)).json()
    assert body["quality"]["ok"] is False
    assert body["quality"]["reason"] == "too_dark"
    assert body["report"]["status"] == "quality_issue"
    assert hf_calls == []
    assert llm_prompts == []


def test_predict_high_confidence(client, good_leaf, hf_calls, llm_prompts):
    body = client.post(
        "/predict", files=_upload(good_leaf), data={"crop": "tomato", "additional_symptoms": "rings"}
    ).json()
    report = body["report"]
    assert report["disease_id"] == "Tomato___Early_blight"
    assert report["status"] == "confirmed_likely"
    assert body["low_confidence"] is False
    assert [t["label"] for t in body["top_k"]][0] == "Tomato___Early_blight"
    assert report["alternatives"][0].startswith("Tomato - Target Spot")
    assert report["initial_message"] == "LLM summary for the farmer."
    assert body["session_context"]["top_k"][0]["label"] == "Tomato___Early_blight"

    # Classifier receives a square 256px JPEG: the size its own preprocessing resizes to.
    sent = Image.open(BytesIO(hf_calls[0]))
    assert sent.format == "JPEG" and sent.size == (256, 256)


def test_llm_gets_prediction_as_text_only(client, good_leaf, hf_calls, llm_prompts):
    client.post("/predict", files=_upload(good_leaf), data={"additional_symptoms": "rings"})
    system, user = llm_prompts[0]
    assert isinstance(system, str) and isinstance(user, str)
    assert "Tomato - Early blight: 91.0%" in user
    assert "Tomato - Target Spot: 6.0%" in user
    assert "Farmer-reported symptoms: rings" in user
    assert "Curated reference" in user


def test_predict_low_confidence(client, good_leaf, hf_calls, llm_prompts):
    hf_calls.scores = [
        LabelScore("Potato___Late_blight", 0.41),
        LabelScore("Potato___Early_blight", 0.38),
    ]
    body = client.post("/predict", files=_upload(good_leaf)).json()
    report = body["report"]
    assert body["low_confidence"] is True
    assert report["status"] == "not_sure"
    assert report["status_label"] == "Not sure"
    assert report["disease_id"] is None
    assert "not confident" in report["diagnosis_name"].lower()
    assert report["expert_confirmation_required"] is True
    assert report["actions"]
    assert "retake" in report["actions"][0]["step"].lower()
    assert body["top_k"][0]["label"] == "Potato___Late_blight"
    assert llm_prompts == []


def test_predict_healthy(client, good_leaf, hf_calls, llm_prompts):
    hf_calls.scores = [LabelScore("Grape___healthy", 0.97)]
    report = client.post("/predict", files=_upload(good_leaf)).json()["report"]
    assert report["status"] == "healthy"
    assert report["expert_confirmation_required"] is False


def test_predict_hf_failure(client, good_leaf, monkeypatch, llm_prompts):
    def boom(*_args, **_kwargs):
        raise HFInferenceError("down")

    monkeypatch.setattr("app.vision.predictor.classify_image", boom)
    body = client.post("/predict", files=_upload(good_leaf)).json()
    assert body["report"]["status"] == "error"
    assert body["top_k"] == []
    assert llm_prompts == []


def test_hf_client_parses_and_sends_auth(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    from app.config import get_settings

    get_settings.cache_clear()
    seen = {}

    class Resp:
        status_code = 200
        text = ""
        headers = {"x-model-id": "neghani/plantvillage-mobilenetv2", "x-model-revision": "v1.0"}

        def json(self):
            return [[{"label": "Apple___healthy", "score": 0.2}, {"label": "Apple___Black_rot", "score": 0.8}]]

    def fake_post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return Resp()

    monkeypatch.setattr(hf_client.httpx, "post", fake_post)
    result = hf_client.classify_image(b"img")
    assert [r.label for r in result] == ["Apple___Black_rot", "Apple___healthy"]
    assert seen["url"] == "http://hf.test/"
    assert seen["headers"]["Authorization"] == "Bearer hf_test"


def test_generate_text_rejects_non_text():
    from app.llm.provider import generate_text

    with pytest.raises(TypeError):
        generate_text("system", b"image-bytes")  # type: ignore[arg-type]


def test_chat_context_includes_prediction(monkeypatch):
    from app.services import chat

    captured = {}

    def fake_chat(message, context, history="", **_kwargs):
        captured["message"] = message
        captured["context"] = context
        captured["history"] = history
        return "reply"

    monkeypatch.setattr(chat, "chat_with_context", fake_chat)
    ctx = {
        "diagnosis_name": "Tomato Early Blight",
        "crop": "tomato",
        "confidence": 0.91,
        "top_k": [{"label": "Tomato___Early_blight", "display_name": "Tomato - Early blight", "score": 0.91}],
    }
    chat.handle_chat("Should I remove tomato leaves?", ctx)
    assert "Leaf check: tomato leaf — Tomato Early Blight (91.0%)" in captured["context"]
    assert "Top predictions: Tomato - Early blight 91.0%" in captured["context"]

    captured.clear()
    chat.handle_chat(
        "Should I remove them?",
        ctx,
        history=[
            {"role": "user", "content": "Should I remove tomato leaves?"},
            {"role": "assistant", "content": "Remove badly spotted lower leaves and leave healthy ones."},
        ],
    )
    assert captured["message"] == "Should I remove them?"
    assert "Tomato Early Blight" in captured["context"]
    assert "Farmer: Should I remove tomato leaves?" in captured["history"]
    assert "Assistant: Remove badly spotted lower leaves" in captured["history"]


def test_followup_is_the_next_turn_not_a_new_chat():
    from app.llm.provider import build_chat_messages

    messages = build_chat_messages(
        "how the decease will help with sunlight",
        "Leaf check: Pepper bell leaf — Bell Pepper Bacterial Spot (54.0%)",
        history=(
            "Farmer: Effect of sunlight\n"
            "Assistant: Sunlight can dry the leaf surface and slow bacterial spot."
        ),
        reply_instruction="Reply in Telugu.",
    )
    assert [message.type for message in messages] == ["system", "human", "ai", "human"]
    assert "Reply in Telugu." in messages[0].content
    assert messages[1].content == "Effect of sunlight"
    assert "slow bacterial spot" in messages[2].content
    assert "Bell Pepper Bacterial Spot" in messages[3].content
    assert "how the decease will help with sunlight" in messages[3].content
    assert "Reply in Telugu" not in messages[3].content
