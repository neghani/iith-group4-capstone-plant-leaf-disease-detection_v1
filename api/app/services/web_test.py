"""Live web-smoke harness: DuckDuckGo image search → classifier → GT vs prediction.

GT is the *intended* PlantVillage class used for the search query — not a human-verified
label. For tutor demos / smoke only; respect image-site terms and do not redistribute.
"""

from __future__ import annotations

import base64
import json
import logging
import random
from io import BytesIO
from pathlib import Path
from typing import Iterator

import httpx
from PIL import Image

from app.config import get_settings
from app.vision.hf_client import HFInferenceError, classify_image, hf_configured
from app.vision.predictor import display_name

_log = logging.getLogger("leafcheck.web_test")

# Curated “high-likelihood” demo pool (user-reviewed leaf photos + strong lookalikes).
# Clicking Web Test randomizes from this list — no manual GT picker.
DEMO_POOL: list[str] = [
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Potato___Late_blight",
    "Potato___Early_blight",
    "Tomato___Early_blight",
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    "Squash___Powdery_mildew",
    "Grape___healthy",
    "Grape___Black_rot",
    "Apple___Apple_scab",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Raspberry___healthy",
    "Strawberry___Leaf_scorch",
    "Cherry_(including_sour)___healthy",
]

# First query per class (same intent as model/download_web_test_set.py).
CLASS_QUERIES: dict[str, str] = {
    "Apple___Apple_scab": "apple scab leaf disease olive spots close up",
    "Apple___Black_rot": "apple black rot leaf frog eye spots close up",
    "Apple___Cedar_apple_rust": "apple cedar apple rust leaf orange spots close up",
    "Apple___healthy": "healthy apple tree leaf close up green single leaf",
    "Blueberry___healthy": "healthy blueberry leaf close up green",
    "Cherry_(including_sour)___Powdery_mildew": "cherry powdery mildew leaf white coating close up",
    "Cherry_(including_sour)___healthy": "healthy cherry leaf close up green",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn gray leaf spot rectangular lesions close up",
    "Corn_(maize)___Common_rust_": "corn maize common rust leaf pustules orange close up",
    "Corn_(maize)___Northern_Leaf_Blight": "corn northern leaf blight long gray lesions",
    "Corn_(maize)___healthy": "healthy corn maize leaf close up green",
    "Grape___Black_rot": "grape black rot leaf lesions brown close up",
    "Grape___Esca_(Black_Measles)": "grape esca black measles leaf stripes close up",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape isariopsis leaf blight dark spots close up",
    "Grape___healthy": "healthy grape vine leaf close up green",
    "Orange___Haunglongbing_(Citrus_greening)": "citrus greening HLB blotchy leaf yellow mottling close up",
    "Peach___Bacterial_spot": "peach bacterial spot leaf lesions close up",
    "Peach___healthy": "healthy peach leaf close up green",
    "Pepper,_bell___Bacterial_spot": "bell pepper bacterial spot leaf angular lesions close up",
    "Pepper,_bell___healthy": "healthy bell pepper leaf close up green",
    "Potato___Early_blight": "potato early blight leaf concentric rings close up",
    "Potato___Late_blight": "potato late blight leaf dark water soaked lesions",
    "Potato___healthy": "healthy potato plant leaf close up green",
    "Raspberry___healthy": "healthy raspberry leaf close up green",
    "Soybean___healthy": "healthy soybean leaf close up green trifoliate",
    "Squash___Powdery_mildew": "squash powdery mildew leaf white spots close up",
    "Strawberry___Leaf_scorch": "strawberry leaf scorch purple spots disease close up",
    "Strawberry___healthy": "healthy strawberry leaf close up green",
    "Tomato___Bacterial_spot": "tomato bacterial spot leaf small dark spots close up",
    "Tomato___Early_blight": "tomato early blight leaf concentric target spots close up",
    "Tomato___Late_blight": "tomato late blight leaf water soaked lesions close up",
    "Tomato___Leaf_Mold": "tomato leaf mold olive green patches underside close up",
    "Tomato___Septoria_leaf_spot": "tomato septoria leaf spot small circular lesions close up",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato spider mite damage leaf stippling yellow close up",
    "Tomato___Target_Spot": "tomato target spot leaf lesions concentric close up",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato yellow leaf curl virus curled yellow leaf close up",
    "Tomato___Tomato_mosaic_virus": "tomato mosaic virus mottled leaf close up",
    "Tomato___healthy": "healthy tomato leaf close up green",
}

def _manifest_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    # repo: …/api/app/services → parents[3]=repo; docker: /app/app/services → parents[2]=/app
    return [
        here.parents[3] / "data" / "web_test_set" / "manifest.json",
        here.parents[2] / "data" / "web_test_set" / "manifest.json",
        Path("/app/data/web_test_set/manifest.json"),
    ]


def _load_curated_classes() -> list[dict]:
    for path in _manifest_candidates():
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        seen: set[str] = set()
        out: list[dict] = []
        for item in payload.get("items") or []:
            cid = item.get("class_id")
            if not cid or cid in seen or cid not in CLASS_QUERIES:
                continue
            seen.add(cid)
            out.append({
                "class_id": cid,
                "display_name": display_name(cid),
                "query": item.get("query") or CLASS_QUERIES[cid],
                "curated": True,
            })
        if out:
            return out
    return []


def demo_pool_classes() -> list[dict]:
    """Classes used when the UI runs a randomized mixed smoke (no class picker)."""
    curated_ids = {c["class_id"] for c in _load_curated_classes()}
    out: list[dict] = []
    for cid in DEMO_POOL:
        if cid not in CLASS_QUERIES:
            continue
        out.append({
            "class_id": cid,
            "display_name": display_name(cid),
            "query": CLASS_QUERIES[cid],
            "curated": cid in curated_ids,
        })
    return out


def list_classes() -> dict:
    curated = _load_curated_classes()
    all_classes = [
        {
            "class_id": cid,
            "display_name": display_name(cid),
            "query": query,
            "curated": False,
        }
        for cid, query in CLASS_QUERIES.items()
    ]
    settings = get_settings()
    return {
        "model_id": settings.hf_model_id,
        "model_revision": settings.hf_model_revision,
        "hf_configured": hf_configured(),
        "demo_pool": demo_pool_classes(),
        "curated": curated,
        "all": all_classes,
        "note": (
            "One click randomizes GT classes from demo_pool, searches the web, "
            "and classifies with the live MobileNet pin. GT is the search target, "
            "not a human-verified label."
        ),
    }


def _search_urls(query: str, n: int) -> list[str]:
    from ddgs import DDGS

    results = list(DDGS().images(query, max_results=max(n * 4, 20)))
    urls: list[str] = []
    for item in results:
        url = item.get("image") or item.get("url")
        if url and url not in urls:
            urls.append(url)
        if len(urls) >= n * 3:
            break
    return urls


def _fetch_jpeg(url: str, timeout: float = 20.0) -> bytes | None:
    headers = {"User-Agent": "CropCareAI-web-test/1.0 (capstone local demo)"}
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            raw = resp.content
        if len(raw) < 6_000:
            return None
        img = Image.open(BytesIO(raw)).convert("RGB")
        w, h = img.size
        if min(w, h) < 160:
            return None
        img.thumbnail((1024, 1024))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as exc:
        _log.debug("fetch failed %s: %s", url[:80], exc)
        return None


def _thumb_data_url(jpeg: bytes) -> str:
    img = Image.open(BytesIO(jpeg)).convert("RGB")
    img.thumbnail((160, 160))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _classify_row(index: int, class_id: str, url: str, jpeg: bytes) -> dict:
    settings = get_settings()
    pin_id, pin_rev = settings.hf_model_id, settings.hf_model_revision
    try:
        scores = classify_image(jpeg, "image/jpeg", top_k=3)
    except HFInferenceError as exc:
        return {
            "status": "row",
            "index": index,
            "image": _thumb_data_url(jpeg),
            "source_url": url,
            "gt_id": class_id,
            "gt_display": display_name(class_id),
            "pred_id": None,
            "pred_display": None,
            "score": None,
            "match": False,
            "error": str(exc),
            "model_id": pin_id,
            "model_revision": pin_rev,
            "latency_ms": None,
        }

    top = scores[0]
    return {
        "status": "row",
        "index": index,
        "image": _thumb_data_url(jpeg),
        "source_url": url,
        "gt_id": class_id,
        "gt_display": display_name(class_id),
        "pred_id": top.label,
        "pred_display": display_name(top.label),
        "score": round(float(top.score), 4),
        "match": top.label == class_id,
        "top_k": [
            {
                "label": s.label,
                "display_name": display_name(s.label),
                "score": round(float(s.score), 4),
            }
            for s in scores[:3]
        ],
        "model_id": scores.meta.get("model_id") or pin_id,
        "model_revision": scores.meta.get("model_revision") or pin_rev,
        "latency_ms": scores.meta.get("latency_ms"),
    }


def _one_image_for_class(class_id: str) -> tuple[str, bytes] | None:
    query = CLASS_QUERIES[class_id]
    try:
        urls = _search_urls(query, 1)
    except Exception as exc:
        _log.warning("search failed for %s: %s", class_id, exc)
        return None
    for url in urls:
        jpeg = _fetch_jpeg(url)
        if jpeg is not None:
            return url, jpeg
    return None


def run_web_test(class_id: str | None = None, n: int = 10) -> Iterator[dict]:
    """Yield NDJSON events. If class_id is None, randomize a mix from DEMO_POOL."""
    if not hf_configured():
        yield {"status": "error", "detail": "HF_ENDPOINT_URL is not configured."}
        return

    n = max(1, min(int(n), 15))
    settings = get_settings()
    pin_id, pin_rev = settings.hf_model_id, settings.hf_model_revision

    if class_id:
        if class_id not in CLASS_QUERIES:
            yield {"status": "error", "detail": f"Unknown class_id: {class_id}"}
            return
        schedule = [class_id] * n
        mode = "single"
    else:
        pool = [c for c in DEMO_POOL if c in CLASS_QUERIES]
        if not pool:
            yield {"status": "error", "detail": "Demo pool is empty."}
            return
        schedule: list[str] = []
        while len(schedule) < n:
            batch = pool[:]
            random.shuffle(batch)
            schedule.extend(batch)
        schedule = schedule[:n]
        mode = "mixed"

    yield {
        "status": "searching",
        "mode": mode,
        "class_id": class_id,
        "gt_display": display_name(class_id) if class_id else "randomized demo pool",
        "query": CLASS_QUERIES[class_id] if class_id else None,
        "schedule": [display_name(c) for c in schedule],
        "n": n,
        "model_id": pin_id,
        "model_revision": pin_rev,
    }

    correct = 0
    rows = 0
    for class_target in schedule:
        found = _one_image_for_class(class_target)
        if found is None:
            continue
        url, jpeg = found
        if rows == 0:
            yield {"status": "found", "candidates": n}
        row = _classify_row(rows, class_target, url, jpeg)
        if row.get("match"):
            correct += 1
        yield row
        rows += 1

    if rows == 0:
        yield {"status": "error", "detail": "Could not download any web images for this run."}
        return

    yield {
        "status": "done",
        "mode": mode,
        "n": rows,
        "correct": correct,
        "accuracy": round(correct / rows, 3) if rows else 0.0,
        "class_id": class_id,
        "gt_display": display_name(class_id) if class_id else "randomized demo pool",
        "model_id": pin_id,
        "model_revision": pin_rev,
    }
