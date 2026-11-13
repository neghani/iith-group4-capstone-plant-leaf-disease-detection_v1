"""Inference server for the plant-disease classifier on a Hugging Face Space (Docker SDK).

Hub id/revision come from release.json (copy of shared/release.json). Same contract as a
HF Inference Endpoint for image-classification:
    POST /  (raw image bytes)  ?top_k=3  ->  [{"label": "...", "score": 0.97}, ...]
"""

import io
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response
from PIL import Image, ImageOps
from transformers import pipeline


def _release_pin() -> dict:
    here = Path(__file__).resolve().parent
    for path in (here / "release.json", here.parent / "shared" / "release.json"):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("shared/release.json missing (live Hub pin)")


_PIN = _release_pin()
MODEL_ID = os.environ.get("HF_MODEL_ID") or _PIN["hf_model_id"]
REVISION = os.environ.get("HF_MODEL_REVISION") or _PIN["hf_model_revision"]

classifier = pipeline("image-classification", model=MODEL_ID, revision=REVISION, device="cpu")
app = FastAPI(title="PlantVillage disease inference")

# Demo log colors (raw ANSI — `docker compose logs -f` renders them; NO_COLOR=1 to disable).
if os.environ.get("NO_COLOR") is None and os.environ.get("DEMO_LOG_COLOR", "1") != "0":
    _R, _DIM, _BOLD, _PURPLE, _GREEN, _RED, _CYAN = "\x1b[0m", "\x1b[2m", "\x1b[1m", "\x1b[95m", "\x1b[92m", "\x1b[91m", "\x1b[96m"
else:
    _R, _DIM, _BOLD, _PURPLE, _GREEN, _RED, _CYAN = "", "", "", "", "", "", ""


def _say(msg: str) -> None:
    print(f"🔮 {_PURPLE}{_BOLD}INFERENCE{_R} {msg}", flush=True)


_say(f"{_GREEN}✔{_R} model loaded — {MODEL_ID}@{REVISION} (device=cpu)")


@app.get("/health")
def health() -> dict:
    _say(f"{_CYAN}→{_R} GET /health probe")
    return {"status": "ok", "model": MODEL_ID, "revision": REVISION}


@app.post("/")
async def classify(request: Request, response: Response, top_k: int = Query(3, ge=1, le=38)) -> list[dict]:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty image body.")
    _say(f"{_CYAN}▶{_R} classify ▶ {len(body) / 1024:.0f} KB image · top_k={top_k}")
    started = time.perf_counter()
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(body))).convert("RGB")
    except Exception as exc:
        _say(f"{_RED}✖{_R} could not read image ({exc})")
        raise HTTPException(status_code=400, detail="Could not read image.") from exc
    # Lets the API prove which Hub model produced each prediction.
    response.headers["x-model-id"] = MODEL_ID
    response.headers["x-model-revision"] = REVISION
    results = classifier(img, top_k=top_k)
    latency_ms = (time.perf_counter() - started) * 1000
    top1 = results[0] if results else {"label": "(none)", "score": 0.0}
    runner_up = results[1] if len(results) > 1 else None
    margin = (
        f" · runner-up {runner_up['label'].split('___')[-1]} {runner_up['score']:.1%}"
        if runner_up
        else ""
    )
    _say(
        f"{_GREEN}✔{_R} top1 {_BOLD}'{top1['label']}'{_R} {top1['score']:.1%}{margin}"
        f" — {_DIM}{latency_ms:.0f} ms{_R} · {_DIM}{MODEL_ID}@{REVISION}{_R}"
    )
    return [{"label": r["label"], "score": float(r["score"])} for r in results]
