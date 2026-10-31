from dataclasses import dataclass, field

from PIL import Image

from app.config import get_settings
from app.vision.hf_client import classify_image
from app.vision.quality import PreparedImage
from shared.vision.calibration import apply_temperature_dict
from shared.vision.quality import encode_jpeg
from shared.vision.strategies import run_strategy

NUM_CLASSES = 38  # multi-crop strategies average full softmax vectors, so ask for every class


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


def crop_from_class(class_name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato'; 'Corn_(maize)___Common_rust_' -> 'Corn (maize)'."""
    crop = class_name.split("___")[0]
    return crop.replace("_", " ").replace(",", "").strip()


def display_name(class_name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato - Early blight'."""
    parts = class_name.split("___")
    crop = crop_from_class(class_name)
    if len(parts) == 1:
        return crop
    cond = parts[1].replace("_", " ").strip()
    return f"{crop} - {cond}"


def is_healthy(class_name: str) -> bool:
    return class_name.lower().endswith("___healthy")


class HFPredictor:
    """shared.vision.strategies.Predictor over the remote classifier; one HTTP call per crop."""

    def __init__(self):
        self.calls: list[dict] = []

    def predict_proba(self, images: list[Image.Image]) -> list[dict[str, float]]:
        out = []
        for img in images:
            scores = classify_image(encode_jpeg(img), "image/jpeg", top_k=NUM_CLASSES)
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
