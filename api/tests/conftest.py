from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import get_settings
from app.vision.hf_client import Classification, LabelScore


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    for key in ("LLM_API_KEY", "HF_ENDPOINT_URL", "HF_TOKEN", "CONFIDENCE_THRESHOLD"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.5")
    monkeypatch.setenv("HF_ENDPOINT_URL", "http://hf.test/")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


def _jpeg(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def good_leaf() -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(40, 200, size=(600, 800, 3), dtype=np.uint8)
    return _jpeg(Image.fromarray(arr))


@pytest.fixture
def dark_leaf() -> bytes:
    return _jpeg(Image.new("RGB", (400, 400), (5, 5, 5)))


@pytest.fixture
def tiny_leaf() -> bytes:
    return _jpeg(Image.new("RGB", (50, 50), (120, 160, 90)))


def _upload(data: bytes):
    return {"file": ("leaf.jpg", data, "image/jpeg")}


class FakeClassifier(list):
    """Stands in for the remote classifier; the list holds the image payloads it received."""

    def __init__(self):
        super().__init__()
        self.scores = [
            LabelScore("Tomato___Early_blight", 0.91),
            LabelScore("Tomato___Target_Spot", 0.06),
            LabelScore("Tomato___Septoria_leaf_spot", 0.02),
        ]

    def __call__(self, image_bytes, content_type="image/jpeg", top_k=None):
        self.append(image_bytes)
        meta = {
            "source": "endpoint",
            "endpoint_host": "hf.test",
            "model_id": "neghani/plantvillage-mobilenetv2",
            "model_revision": "v1.0",
            "model_verified": True,
            "latency_ms": 12,
            "input_bytes": len(image_bytes),
        }
        return Classification(self.scores, meta)


@pytest.fixture
def hf_calls(monkeypatch):
    fake = FakeClassifier()
    monkeypatch.setattr("app.vision.predictor.classify_image", fake)
    return fake


@pytest.fixture
def llm_prompts(monkeypatch):
    prompts: list[tuple[str, str]] = []

    def fake_generate(system, user):
        prompts.append((system, user))
        return "LLM summary for the farmer."

    monkeypatch.setattr("app.services.report.generate_text", fake_generate)
    return prompts


