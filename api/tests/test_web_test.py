"""Web-test harness: class list + NDJSON stream wiring (search/classify mocked)."""

from io import BytesIO

from PIL import Image

from app.schemas import ImageQualityResult
from app.services import web_test
from app.vision.predictor import PredictionResult, TopKItem
from app.vision.quality import PreparedImage


def _jpeg() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (200, 200), color=(34, 120, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def _prepared(jpeg: bytes | None = None) -> PreparedImage:
    raw = jpeg or _jpeg()
    img = Image.open(BytesIO(raw)).convert("RGB")
    return PreparedImage(
        quality=ImageQualityResult(
            ok=True,
            reason=None,
            message="ok",
            brightness=120.0,
            blur_score=0.9,
            leaf_found=True,
            leaf_coverage=0.4,
            original_size=[img.size[0], img.size[1]],
        ),
        image=img,
        candidates=[],
        gate=None,
        trace={},
    )


def test_list_classes_includes_plantvillage_and_pin():
    payload = web_test.list_classes()
    assert payload["hf_configured"] in (True, False)
    assert payload["model_id"]
    assert payload["model_revision"]
    ids = {c["class_id"] for c in payload["all"]}
    assert "Tomato___Early_blight" in ids
    assert len(ids) == 38
    pool = {c["class_id"] for c in payload["demo_pool"]}
    assert "Pepper,_bell___Bacterial_spot" in pool
    assert "Raspberry___healthy" in pool


def test_run_web_test_unknown_class():
    events = list(web_test.run_web_test("NotAReal___Class", n=3))
    assert events[0]["status"] == "error"


def test_run_web_test_streams_rows(monkeypatch):
    monkeypatch.setattr(web_test, "hf_configured", lambda: True)
    jpeg = _jpeg()
    prepared = _prepared(jpeg)
    monkeypatch.setattr(
        web_test,
        "_one_image_for_class",
        lambda class_id: (f"https://example.com/{class_id}.jpg", jpeg),
    )
    monkeypatch.setattr(web_test, "prepare_image", lambda data: prepared)
    monkeypatch.setattr(
        "app.services.smoke_common.predict",
        lambda prepared: PredictionResult(
            disease_id="Tomato___Early_blight",
            confidence=0.91,
            crop="Tomato",
            status="ok",
            top_k=[
                TopKItem("Tomato___Early_blight", "Tomato - Early blight", 0.91),
                TopKItem("Tomato___Late_blight", "Tomato - Late blight", 0.05),
            ],
            inference={
                "model_id": "neghani/plantvillage-mobilenetv2",
                "model_revision": "v1.2",
                "latency_ms": 12,
            },
            strategy={"leaf_found": True, "strategy": "B"},
        ),
    )

    events = list(web_test.run_web_test("Tomato___Early_blight", n=3))
    assert events[0]["status"] == "searching"
    assert events[0]["model_id"]
    rows = [e for e in events if e["status"] == "row"]
    assert len(rows) == 3
    assert all(r["match"] for r in rows)
    assert all(r["verdict"] == "yes" for r in rows)
    assert all(r["model_revision"] == "v1.2" for r in rows)
    assert all(r["latency_ms"] == 12 for r in rows)
    done = events[-1]
    assert done["status"] == "done"
    assert done["correct"] == 3


def test_run_web_test_mixed_randomizes_pool(monkeypatch):
    monkeypatch.setattr(web_test, "hf_configured", lambda: True)
    jpeg = _jpeg()
    prepared = _prepared(jpeg)

    def fake_one(class_id: str):
        return (f"https://example.com/{class_id}.jpg", jpeg)

    monkeypatch.setattr(web_test, "_one_image_for_class", fake_one)
    monkeypatch.setattr(web_test, "prepare_image", lambda data: prepared)
    monkeypatch.setattr(
        "app.services.smoke_common.predict",
        lambda prepared: PredictionResult(
            disease_id="Pepper,_bell___Bacterial_spot",
            confidence=0.88,
            crop="Pepper bell",
            status="ok",
            top_k=[TopKItem("Pepper,_bell___Bacterial_spot", "Pepper bell - Bacterial spot", 0.88)],
            inference={
                "model_id": "neghani/plantvillage-mobilenetv2",
                "model_revision": "v1.2",
                "latency_ms": 9,
            },
            strategy={"leaf_found": True},
        ),
    )

    events = list(web_test.run_web_test(None, n=5))
    assert events[0]["status"] == "searching"
    assert events[0]["mode"] == "mixed"
    assert len(events[0]["schedule"]) == 5
    rows = [e for e in events if e["status"] == "row"]
    assert len(rows) == 5
    assert {r["gt_id"] for r in rows} <= set(web_test.DEMO_POOL)
    assert all(r["model_revision"] == "v1.2" for r in rows)
    assert all("verdict" in r for r in rows)
