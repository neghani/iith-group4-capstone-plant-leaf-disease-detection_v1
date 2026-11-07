"""Web-test harness: class list + NDJSON stream wiring (search/classify mocked)."""

from app.services import web_test


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
    monkeypatch.setattr(
        web_test,
        "_search_urls",
        lambda query, n: [f"https://example.com/{i}.jpg" for i in range(max(n, 3))],
    )

    from io import BytesIO
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (200, 200), color=(34, 120, 40)).save(buf, format="JPEG")
    jpeg = buf.getvalue()

    monkeypatch.setattr(web_test, "_fetch_jpeg", lambda url, timeout=20.0: jpeg)

    class _Score:
        def __init__(self, label, score):
            self.label = label
            self.score = score

    class _Cls(list):
        meta = {
            "model_id": "neghani/plantvillage-mobilenetv2",
            "model_revision": "v1.2",
            "latency_ms": 12,
        }

        def __init__(self, label="Tomato___Early_blight"):
            super().__init__([_Score(label, 0.91), _Score("Tomato___Late_blight", 0.05)])
            self._label = label

    monkeypatch.setattr(
        web_test,
        "classify_image",
        lambda *a, **k: _Cls("Tomato___Early_blight"),
    )

    events = list(web_test.run_web_test("Tomato___Early_blight", n=3))
    assert events[0]["status"] == "searching"
    assert events[0]["model_id"]
    rows = [e for e in events if e["status"] == "row"]
    assert len(rows) == 3
    assert all(r["match"] for r in rows)
    assert all(r["model_revision"] == "v1.2" for r in rows)
    assert all(r["latency_ms"] == 12 for r in rows)
    done = events[-1]
    assert done["status"] == "done"
    assert done["correct"] == 3


def test_run_web_test_mixed_randomizes_pool(monkeypatch):
    monkeypatch.setattr(web_test, "hf_configured", lambda: True)

    from io import BytesIO
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (200, 200), color=(34, 120, 40)).save(buf, format="JPEG")
    jpeg = buf.getvalue()

    def fake_one(class_id: str):
        return (f"https://example.com/{class_id}.jpg", jpeg)

    class _Score:
        def __init__(self, label, score):
            self.label = label
            self.score = score

    class _Cls(list):
        meta = {
            "model_id": "neghani/plantvillage-mobilenetv2",
            "model_revision": "v1.2",
            "latency_ms": 9,
        }

        def __init__(self):
            super().__init__([_Score("Pepper,_bell___Bacterial_spot", 0.88)])

    monkeypatch.setattr(web_test, "_one_image_for_class", fake_one)
    monkeypatch.setattr(web_test, "classify_image", lambda *a, **k: _Cls())

    events = list(web_test.run_web_test(None, n=5))
    assert events[0]["status"] == "searching"
    assert events[0]["mode"] == "mixed"
    assert len(events[0]["schedule"]) == 5
    rows = [e for e in events if e["status"] == "row"]
    assert len(rows) == 5
    assert {r["gt_id"] for r in rows} <= set(web_test.DEMO_POOL)
    assert all(r["model_revision"] == "v1.2" for r in rows)
    assert all(r["latency_ms"] == 9 for r in rows)
