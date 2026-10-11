"""Inference server for neghani/plantvillage-mobilenetv2 on a Hugging Face Space (Docker SDK).

Same contract as a HF Inference Endpoint for image-classification:
    POST /  (raw image bytes)  ?top_k=3  ->  [{"label": "...", "score": 0.97}, ...]
"""

import io
import os

from fastapi import FastAPI, HTTPException, Query, Request, Response
from PIL import Image, ImageOps
from transformers import pipeline

MODEL_ID = os.environ.get("HF_MODEL_ID", "neghani/plantvillage-mobilenetv2")
REVISION = os.environ.get("HF_MODEL_REVISION", "v1.0")

classifier = pipeline("image-classification", model=MODEL_ID, revision=REVISION, device="cpu")
app = FastAPI(title="PlantVillage MobileNetV2 inference")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "revision": REVISION}


@app.post("/")
async def classify(request: Request, response: Response, top_k: int = Query(3, ge=1, le=38)) -> list[dict]:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty image body.")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(body))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read image.") from exc
    # Lets the API prove which Hub model produced each prediction.
    response.headers["x-model-id"] = MODEL_ID
    response.headers["x-model-revision"] = REVISION
    return [{"label": r["label"], "score": float(r["score"])} for r in classifier(img, top_k=top_k)]
