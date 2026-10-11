import logging
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.agent.graph import diagnosis_graph
from app.config import get_settings
from app.llm.provider import llm_configured
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DiagnosisReport,
    HealthResponse,
    ImageQualityResult,
    InferenceInfo,
    PredictResponse,
    TopKPrediction,
)
from app.services.chat import handle_chat
from app.services.report import build_quality_report
from app.vision.hf_client import hf_configured, probe_endpoint
from app.vision.quality import PreparedImage, prepare_image, validate_upload
from shared.vision.config import INFERENCE_SIZE

_log = logging.getLogger("leafcheck")
if not _log.handlers:  # uvicorn only configures its own loggers
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)

app = FastAPI(
    title="Plant Disease Detection API",
    description=(
        "Two-step flow: POST /quality-check gates the photo, POST /predict runs the "
        "Hugging Face classifier and a text-only LLM for farmer guidance. Large photos are "
        "downscaled and cropped to the best leaf before they reach the classifier."
    ),
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _gate_upload(file: UploadFile) -> PreparedImage:
    settings = get_settings()
    data = await file.read()
    err = validate_upload(file.content_type, len(data), settings.max_image_bytes)
    if err:
        raise HTTPException(status_code=400, detail=err)
    return await run_in_threadpool(prepare_image, data)  # CPU-bound; keep the event loop free


@app.get("/", include_in_schema=False)
def root():
    """Browsers open / by default; send people to interactive API docs."""
    return RedirectResponse(url="/docs", status_code=302)


@app.get("/health", response_model=HealthResponse)
def health(probe: bool = False) -> HealthResponse:
    """`?probe=true` also calls the inference server's /health to confirm which model it serves."""
    settings = get_settings()
    reachable, served = probe_endpoint() if probe else (None, None)
    return HealthResponse(
        status="ok",
        llm_configured=llm_configured(),
        hf_model_configured=hf_configured(),
        hf_model_id=settings.hf_model_id,
        hf_model_revision=settings.hf_model_revision,
        hf_endpoint_configured=bool(settings.hf_endpoint_url.strip()),
        hf_reachable=reachable,
        hf_served_model=served,
    )


@app.post("/quality-check", response_model=ImageQualityResult)
async def quality_check(file: Annotated[UploadFile, File(...)]) -> ImageQualityResult:
    """Step 1: cheap local checks only (type, size, dimensions, brightness, blur)."""
    return (await _gate_upload(file)).quality


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: Annotated[UploadFile, File(...)],
    crop: Annotated[str | None, Form()] = None,
    additional_symptoms: Annotated[str | None, Form()] = None,
) -> PredictResponse:
    """Step 2: re-runs the gate server-side, then HF classifier -> knowledge -> LLM -> safety."""
    prepared = await _gate_upload(file)
    quality = prepared.quality
    if not quality.ok or prepared.image is None:
        report = build_quality_report(quality.message or "Poor image quality.")
        return PredictResponse(
            quality=quality,
            low_confidence=True,
            report=report,
            session_context={},
        )

    result = await run_in_threadpool(
        diagnosis_graph.invoke,
        {
            "prepared": prepared,
            "crop_hint": crop or "",
            "additional_symptoms": additional_symptoms or "",
        },
    )
    report = DiagnosisReport.model_validate(result["report"])
    session_context = result.get("session_context", {})
    inference = result.get("inference")
    strategy = result.get("strategy") or {}
    _log.info(
        "predict: disease=%s conf=%.3f status=%s crop_flag=%r strategy=%s crops=%s agreement=%s model=%s",
        report.disease_id,
        report.confidence,
        report.status,
        result.get("crop_flag", ""),
        strategy.get("strategy"),
        strategy.get("n_crops"),
        strategy.get("agreement"),
        (inference or {}).get("model_id"),
    )

    return PredictResponse(
        quality=quality,
        low_confidence=bool(session_context.get("low_confidence", True)),
        report=report,
        top_k=[TopKPrediction.model_validate(t) for t in result.get("top_k", [])],
        session_context=session_context,
        inference=InferenceInfo(
            **inference,
            input_size=[INFERENCE_SIZE, INFERENCE_SIZE],
            preprocessing={**prepared.trace, "crop_strategy": strategy},
        )
        if inference
        else None,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    return handle_chat(
        body.message,
        body.session_context,
        [turn.model_dump() for turn in body.history],
    )
