"""Local real_world_set smoke harness."""

from io import BytesIO
from pathlib import Path

from PIL import Image

from app.schemas import ImageQualityResult
from app.services import local_test
from app.services.smoke_common import score_verdict
from app.vision.predictor import PredictionResult, TopKItem
from app.vision.quality import PreparedImage


def _jpeg(color=(34, 120, 40), size=400) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (size, size), color=color).save(buf, format="JPEG", quality=92)
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


def test_list_local_set_reports_pin():
    payload = local_test.list_local_set()
    assert payload["model_id"]
    assert payload["model_revision"]
    assert "available" in payload
    assert payload["n_labels_csv"] >= 0


def test_score_verdict_shared():
    out = score_verdict(
        pred_id="Soybean___healthy",
        gt_id="Grape___healthy",
        score=0.86,
        model_status="ok",
        use_gt_crop_hint=True,
        threshold=0.5,
    )
    assert out["verdict"] == "crop_mismatch"


def test_run_local_test_samples(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(local_test, "hf_configured", lambda: True)

    set_dir = tmp_path / "real_world_set"
    img_dir = set_dir / "images" / "Tomato___Early_blight"
    img_dir.mkdir(parents=True)
    files = []
    for i in range(5):
        rel = f"images/Tomato___Early_blight/rw_{i:03d}.jpg"
        path = set_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_jpeg())
        files.append((rel, "Tomato___Early_blight", "tune" if i % 2 == 0 else "final"))

    labels = set_dir / "labels.csv"
    labels.write_text(
        "file,label,plantdoc_class,split\n"
        + "\n".join(f"{rel},{lab},Tomato leaf,{spl}" for rel, lab, spl in files),
        encoding="utf-8",
    )

    monkeypatch.setattr(local_test, "resolve_set_dir", lambda: set_dir)
    jpeg = _jpeg()
    prepared = _prepared(jpeg)
    monkeypatch.setattr(local_test, "prepare_image", lambda data: prepared)
    monkeypatch.setattr(
        "app.services.smoke_common.predict",
        lambda prepared: PredictionResult(
            disease_id="Tomato___Early_blight",
            confidence=0.93,
            crop="Tomato",
            status="ok",
            top_k=[TopKItem("Tomato___Early_blight", "Tomato - Early blight", 0.93)],
            inference={
                "model_id": "neghani/plantvillage-mobilenetv2",
                "model_revision": "v1.2",
                "latency_ms": 11,
            },
            strategy={"leaf_found": True, "strategy": "B"},
        ),
    )

    events = list(local_test.run_local_test(n=3, use_gt_crop_hint=True))
    assert events[0]["status"] == "searching"
    assert events[0]["mode"] == "local"
    assert events[0]["n"] == 3
    rows = [e for e in events if e["status"] == "row"]
    assert len(rows) == 3
    assert all(r["verdict"] == "yes" for r in rows)
    assert all(r["match"] for r in rows)
    done = events[-1]
    assert done["status"] == "done"
    assert done["correct"] == 3
    assert done["abstain"] == 0


def test_run_local_test_missing_images(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(local_test, "hf_configured", lambda: True)
    set_dir = tmp_path / "real_world_set"
    set_dir.mkdir()
    (set_dir / "labels.csv").write_text(
        "file,label,plantdoc_class,split\n"
        "images/Tomato___healthy/missing.jpg,Tomato___healthy,Tomato leaf,tune\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(local_test, "resolve_set_dir", lambda: set_dir)
    events = list(local_test.run_local_test(n=5))
    assert events[0]["status"] == "error"
    assert "git-ignored" in events[0]["detail"] or "No image" in events[0]["detail"]
