from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.agent.graph import diagnosis_graph
from app.config import get_settings
from app.llm.provider import llm_configured
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DiagnosisReport,
    HealthResponse,
    ImageQualityResult,
    PredictResponse,
)
from app.services.chat import handle_chat
from app.vision.quality import check_image_quality, validate_upload

app = FastAPI(
    title="Plant Disease Detection API",
    description="Phase 2 demo: placeholder vision + LangGraph agent flow",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    """Browsers open / by default; send people to interactive API docs."""
    return RedirectResponse(url="/docs", status_code=302)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", llm_configured=llm_configured())


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: Annotated[UploadFile, File(...)],
    crop: Annotated[str | None, Form()] = None,
    additional_symptoms: Annotated[str | None, Form()] = None,
) -> PredictResponse:
    settings = get_settings()
    data = await file.read()
    err = validate_upload(file.content_type, len(data), settings.max_image_bytes)
    if err:
        raise HTTPException(status_code=400, detail=err)

    quality, _ = check_image_quality(data)
    if not quality.ok:
        from app.services.report import build_quality_report

        report = build_quality_report(quality.message or "Poor image quality.")
        return PredictResponse(
            quality=quality,
            low_confidence=True,
            report=report,
            session_context={},
        )

    result = diagnosis_graph.invoke(
        {
            "image_bytes": data,
            "crop_hint": crop or "",
            "additional_symptoms": additional_symptoms or "",
        }
    )
    report = DiagnosisReport.model_validate(result["report"])
    settings = get_settings()
    low = report.confidence < settings.confidence_threshold and report.status != "quality_issue"

    return PredictResponse(
        quality=ImageQualityResult(ok=True),
        low_confidence=low,
        report=report,
        session_context=result.get("session_context", {}),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    return handle_chat(body.message, body.session_context)
