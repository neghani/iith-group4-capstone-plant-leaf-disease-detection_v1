import json

from app.config import get_settings
from app.services.translate import load_glossary
from conftest import _upload


def test_glossary_covers_farmer_terms():
    glossary = load_glossary()
    assert len(glossary) >= 60
    assert glossary["early blight"] == "తొలి ఎండుతెగులు"
    assert glossary["tomato"] == "టమాట"


def test_english_predict_skips_translation(client, good_leaf, hf_calls, llm_prompts, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("translation should not run for English")

    monkeypatch.setattr("app.services.translate.generate_text", boom)
    body = client.post("/predict", files=_upload(good_leaf)).json()
    assert body["language"] == "en"
    assert body["translation_applied"] is False
    assert body["report"]["initial_message"] == "LLM summary for the farmer."
    assert body["report"]["disease_id"] == "Tomato___Early_blight"


def test_telugu_without_llm_keeps_english(client, good_leaf, hf_calls, llm_prompts):
    body = client.post("/predict", files=_upload(good_leaf), data={"language": "te"}).json()
    assert body["language"] == "te"
    assert body["translation_applied"] is False
    assert body["report"]["initial_message"] == "LLM summary for the farmer."
    assert body["report"]["status"] == "confirmed_likely"
    assert body["report"]["crop"] == "Tomato"


def test_telugu_predict_translates_finished_report(client, good_leaf, hf_calls, llm_prompts, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    get_settings.cache_clear()
    captured = {}

    def fake_translate(system, user):
        captured["system"] = system
        captured["user"] = user
        payload = json.loads(user.split("JSON:\n", 1)[1])
        payload["initial_message"] = "తెలుగు సారాంశం"
        payload["diagnosis_name"] = "టమాట తొలి ఎండుతెగులు"
        payload["status_label"] = "సంభావ్య తెగులు"
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr("app.services.translate.generate_text", fake_translate)
    body = client.post("/predict", files=_upload(good_leaf), data={"language": "te"}).json()
    report = body["report"]

    assert body["translation_applied"] is True
    assert body["language"] == "te"
    assert report["initial_message"] == "తెలుగు సారాంశం"
    assert report["diagnosis_name"] == "టమాట తొలి ఎండుతెగులు"
    assert report["disease_id"] == "Tomato___Early_blight"
    assert report["status"] == "confirmed_likely"
    assert report["crop"] == "Tomato"
    assert body["session_context"]["summary"] == "తెలుగు సారాంశం"
    assert body["session_context"]["diagnosis_name"] == "టమాట తొలి ఎండుతెగులు"
    assert "early blight = తొలి ఎండుతెగులు" in captured["user"]
    assert "LLM summary for the farmer." in captured["user"]
    assert "Do not add diseases" in captured["system"]


def test_bad_telugu_translation_keeps_english(client, good_leaf, hf_calls, llm_prompts, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    get_settings.cache_clear()

    def fake_translate(_system, _user):
        return "not json"

    monkeypatch.setattr("app.services.translate.generate_text", fake_translate)
    body = client.post("/predict", files=_upload(good_leaf), data={"language": "te"}).json()
    assert body["translation_applied"] is False
    assert body["report"]["initial_message"] == "LLM summary for the farmer."
    assert body["report"]["disease_id"] == "Tomato___Early_blight"


def test_chat_requests_telugu(client, monkeypatch):
    captured = {}

    def fake_chat(message, context, history="", reply_instruction=""):
        captured["message"] = message
        captured["instruction"] = reply_instruction
        return "సమాధానం"

    monkeypatch.setattr("app.services.chat.chat_with_context", fake_chat)
    body = client.post(
        "/chat",
        json={
            "message": "ఆకులు తీసేయాలా?",
            "session_context": {"diagnosis_name": "Tomato Early Blight", "crop": "Tomato"},
            "language": "te",
        },
    ).json()
    assert body["declined"] is False
    assert body["reply"] == "సమాధానం"
    assert captured["message"] == "ఆకులు తీసేయాలా?"
    assert "early blight = తొలి ఎండుతెగులు" in captured["instruction"]
    assert "Reply in Telugu" in captured["instruction"]
