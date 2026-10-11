from typing import Literal

from pydantic import BaseModel, Field


ConfidenceCategory = Literal["high", "moderate", "low", "inconclusive"]
DiagnosisStatus = Literal[
    "confirmed_likely",
    "possible",
    "inconclusive",
    "quality_issue",
    "error",
]


class ImageQualityResult(BaseModel):
    ok: bool
    message: str | None = None
    brightness: float | None = None
    blur_score: float | None = None


class SymptomItem(BaseModel):
    name: str
    explanation: str
    evidence_source: Literal["reference", "prediction", "general"] = "reference"


class ActionItem(BaseModel):
    step: str
    safety_note: str | None = None


class DiagnosisReport(BaseModel):
    status: DiagnosisStatus
    status_label: str
    disease_id: str | None = None
    diagnosis_name: str
    crop: str | None = None
    confidence: float
    confidence_category: ConfidenceCategory
    confidence_explanation: str
    description: str
    symptoms: list[SymptomItem] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    expert_confirmation_required: bool = False
    actions: list[ActionItem] = Field(default_factory=list)
    monitoring_guidance: str = ""
    diagnostic_limitations: str = ""
    initial_message: str = ""


class PredictResponse(BaseModel):
    quality: ImageQualityResult
    low_confidence: bool
    report: DiagnosisReport
    session_context: dict = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str
    session_context: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    declined: bool = False


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
