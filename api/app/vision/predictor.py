from dataclasses import dataclass, field

from PIL import Image

from app.config import get_settings
from app.vision.hf_client import classify_image
from app.vision.quality import PreparedImage
from shared.vision.calibration import apply_temperature_dict
from shared.vision.quality import encode_jpeg
from shared.vision.strategies import run_strategy

def num_classes() -> int:
    """Ask the classifier for every class. The count follows the knowledge base, not a fixed 38."""
    from app.services.knowledge import list_disease_ids

    return max(1, len(list_disease_ids()))


@dataclass
class TopKItem:
    label: str
    display_name: str
    score: float


@dataclass
class PredictionResult:
    disease_id: str
    confidence: float
    crop: str
    status: str = "ok"  # "ok" | "not_sure" (low confidence, or the crops disagree)
    top_k: list[TopKItem] = field(default_factory=list)
    inference: dict = field(default_factory=dict)  # provenance of the HF call(s)
    strategy: dict = field(default_factory=dict)  # crop strategy trace


_MASTER_CROPS = (
    "bell_pepper", "groundnut", "strawberry", "sugarcane",
    "apple", "bean", "cherry", "corn", "cotton", "cucumber", "grape", "guava",
    "lemon", "peach", "potato", "pumpkin", "rice", "tomato", "wheat",
)


def _master_parts(class_name: str) -> tuple[str, str] | None:
    low = class_name.lower()
    if low.startswith("healthy_"):
        return low[len("healthy_"):], "healthy"
    if low.startswith("diseased_"):
        return low[len("diseased_"):], "diseased"
    for crop in sorted(_MASTER_CROPS, key=len, reverse=True):
        prefix = crop + "_"
        if low.startswith(prefix):
            return crop, low[len(prefix):]
    return None


def crop_from_class(class_name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato'; 'tomato_early_blight' -> 'Tomato'; 'healthy_apple' -> 'Apple'."""
    if "___" not in class_name:
        parts = _master_parts(class_name)
        if parts:
            return parts[0].replace("_", " ").title()
    crop = class_name.split("___")[0]
    return crop.replace("_", " ").replace(",", "").strip()


def display_name(class_name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato - Early blight'; 'healthy_apple' -> 'Apple - Healthy'."""
    if "___" not in class_name:
        parts = _master_parts(class_name)
        if parts:
            cond = parts[1].replace("_", " ").strip()
            crop = parts[0].replace("_", " ").title()
            return f"{crop} - {cond[:1].upper()}{cond[1:]}" if cond else crop
    parts = class_name.split("___")
    crop = crop_from_class(class_name)
    if len(parts) == 1:
        return crop
    cond = parts[1].replace("_", " ").strip()
    return f"{crop} - {cond}"


def is_healthy(class_name: str) -> bool:
    low = class_name.lower()
    return low.endswith("___healthy") or low.startswith("healthy_") or low.endswith("_healthy")


class HFPredictor:
    """shared.vision.strategies.Predictor over the remote classifier; one HTTP call per crop."""

    def __init__(self):
        self.calls: list[dict] = []

    def predict_proba(self, images: list[Image.Image]) -> list[dict[str, float]]:
        out = []
        for img in images:
            scores = classify_image(encode_jpeg(img), "image/jpeg", top_k=num_classes())
            self.calls.append(dict(getattr(scores, "meta", {})))
            raw = {s.label: s.score for s in scores}
            t = get_settings().calibration_temperature
            out.append(apply_temperature_dict(raw, t) if t != 1.0 else raw)
        return out

    def provenance(self) -> dict:
        if not self.calls:
            return {}
        meta = dict(self.calls[0])
        meta["latency_ms"] = sum(c.get("latency_ms", 0) for c in self.calls)
        meta["model_verified"] = all(c.get("model_verified") for c in self.calls)
        meta["calls"] = len(self.calls)
        return meta


def predict(prepared: PreparedImage) -> PredictionResult:
    settings = get_settings()
    hf = HFPredictor()
    res = run_strategy(
        prepared.image,
        settings.crop_strategy,
        hf,
        threshold=settings.confidence_threshold,
        ood_threshold=settings.ood_threshold,
        top_k=settings.hf_top_k,
        candidates=prepared.candidates,
        gate=prepared.gate,
        tta=settings.crop_tta,
    )
    disease = res.get("disease") or ""
    trace = {
        k: res[k]
        for k in (
            "strategy",
            "n_crops",
            "agreement",
            "crop_boxes",
            "crop_tops",
            "tta",
            "margin",
            "ood_threshold",
        )
        if k in res
    }
    if res.get("message"):
        trace["message"] = res["message"]
    if res.get("gate_reason"):
        trace["gate_reason"] = res["gate_reason"]
    return PredictionResult(
        disease_id=disease,
        confidence=float(res["confidence"] or 0.0),
        crop=crop_from_class(disease) if disease else "",
        status=res["status"],
        top_k=[
            TopKItem(label=t["disease"], display_name=display_name(t["disease"]), score=round(t["confidence"], 4))
            for t in res.get("top_k") or []
        ],
        inference=hf.provenance(),
        strategy=trace,
    )
