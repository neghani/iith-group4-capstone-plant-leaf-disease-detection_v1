from typing import Literal

from pydantic import BaseModel, Field


ConfidenceCategory = Literal["high", "moderate", "low", "inconclusive"]
DiagnosisStatus = Literal[
    "confirmed_likely",
    "healthy",
    "possible",
    "inconclusive",
    "not_sure",
    "quality_issue",
    "error",
]


class ImageQualityResult(BaseModel):
    ok: bool
    reason: str | None = None
    message: str | None = None
    brightness: float | None = None
    # Leaf sharpness 0 (blurry) .. 1 (sharp); resolution independent, see vision/leaf.py.
    blur_score: float | None = None
    leaf_found: bool | None = None
    leaf_coverage: float | None = None
    original_size: list[int] | None = None


class TopKPrediction(BaseModel):
    label: str
    display_name: str
    score: float


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


ProvenanceSourceId = Literal[
    "image_quality_checks",  # OpenCV gate: brightness, blur, leaf segmentation
    "image_classifier",  # the Hugging Face CNN that named the disease
    "decision_rules",  # fixed thresholds turning scores into status/confidence text
    "curated_reference",  # knowledge-base notes written ahead of time, not generated
    "farmer_input",  # optional crop hint / symptoms typed into the form
    "llm_summary",  # LLM-written farmer summary (text only, never sees the photo)
    "template_text",  # canned template used when the LLM is unavailable or skipped
]


class ProvenanceEntry(BaseModel):
    """Declares which component produced one field of the response.

    Tutors/reviewers reading the raw JSON can tell the classifier apart from the
    LLM without opening the frontend.
    """

    field: str  # dotted path into the response, e.g. "report.description"
    source: ProvenanceSourceId
    label: str  # human-readable component name, e.g. "Image classifier (org/model v1.1)"
    detail: str  # what exactly that component contributed to the field


class InferenceInfo(BaseModel):
    """Proof of what ran: which Hugging Face model answered and what image it received."""

    model_id: str
    model_revision: str
    model_verified: bool = False  # True when the serving side reported the model itself
    source: str  # "endpoint" (Space / Inference Endpoint) or "hf-router"
    endpoint_host: str
    latency_ms: int
    input_bytes: int
    input_size: list[int] | None = None  # w, h of the JPEG the classifier received
    preprocessing: dict = Field(default_factory=dict)  # leaf crop / scaling trace


class PredictResponse(BaseModel):
    quality: ImageQualityResult
    low_confidence: bool
    report: DiagnosisReport
    top_k: list[TopKPrediction] = Field(default_factory=list)
    session_context: dict = Field(default_factory=dict)
    inference: InferenceInfo | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
    language: str = "en"
    translation_applied: bool = False


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    session_context: dict = Field(default_factory=dict)
    # Prior turns only. The current question is `message`, sent again with the leaf check.
    history: list[ChatTurn] = Field(default_factory=list)
    language: str = "en"


class ChatResponse(BaseModel):
    reply: str
    declined: bool = False


class WebTestRunRequest(BaseModel):
    """Live web search + classifier smoke.

    Omit ``class_id`` to randomize a mix from the high-likelihood demo pool.
    """

    class_id: str | None = None
    n: int = Field(default=10, ge=1, le=15)


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    hf_model_configured: bool
    hf_model_id: str
    hf_model_revision: str
    hf_endpoint_configured: bool
    hf_reachable: bool | None = None  # None = not probed / server has no /health
    hf_served_model: str | None = None
