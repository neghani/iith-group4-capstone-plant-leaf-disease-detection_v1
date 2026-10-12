import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from app.config import get_settings


logger = logging.getLogger("leafcheck.hf")


class HFInferenceError(RuntimeError):
    """Raised when the remote Hugging Face classifier cannot return a prediction."""


@dataclass
class LabelScore:
    label: str
    score: float


class Classification(list):
    """Sorted `LabelScore`s plus provenance (`meta`) proving which HF model answered."""

    meta: dict

    def __init__(self, scores: list[LabelScore], meta: dict):
        super().__init__(scores)
        self.meta = meta


def hf_configured() -> bool:
    """True only when a dedicated Space / Inference Endpoint is set.

    Serverless HF inference does not host this fine-tune; an empty URL used to
    look healthy while predict failed against the router.
    """
    return bool(get_settings().hf_endpoint_url.strip())


def _target_url() -> str:
    settings = get_settings()
    endpoint = settings.hf_endpoint_url.strip()
    if not endpoint:
        raise HFInferenceError(
            "HF_ENDPOINT_URL is empty. Point it at hf_space "
            "(Docker: http://inference-service:7860/; local: http://localhost:7860/). "
            "Serverless HF inference does not host this fine-tune."
        )
    return endpoint


def _provenance(response: httpx.Response, latency_ms: int, image_bytes: int) -> dict:
    settings = get_settings()
    endpoint = settings.hf_endpoint_url.strip()
    # Our Space (hf_space/app.py) stamps every answer with the model it actually loaded.
    served_id = response.headers.get("x-model-id")
    served_rev = response.headers.get("x-model-revision")
    return {
        "source": "endpoint" if endpoint else "hf-router",
        "endpoint_host": urlparse(endpoint).netloc if endpoint else "router.huggingface.co",
        "model_id": served_id or settings.hf_model_id,
        "model_revision": served_rev or settings.hf_model_revision,
        "model_verified": bool(served_id),  # True only when the server reported it itself
        "latency_ms": latency_ms,
        "input_bytes": image_bytes,
    }


def _parse(payload: object) -> list[LabelScore]:
    # Endpoints return [{"label", "score"}]; some wrap it once more as [[...]].
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        payload = payload[0]
    if not isinstance(payload, list):
        raise HFInferenceError(f"Unexpected response from classifier: {str(payload)[:200]}")
    results = [
        LabelScore(label=str(item["label"]), score=float(item["score"]))
        for item in payload
        if isinstance(item, dict) and "label" in item and "score" in item
    ]
    if not results:
        raise HFInferenceError("Classifier returned no labels.")
    return sorted(results, key=lambda r: r.score, reverse=True)


def classify_image(
    image_bytes: bytes, content_type: str = "image/jpeg", top_k: int | None = None
) -> Classification:
    settings = get_settings()
    top_k = top_k or settings.hf_top_k
    headers = {"Content-Type": content_type, "Accept": "application/json"}
    if settings.hf_token.strip():
        headers["Authorization"] = f"Bearer {settings.hf_token.strip()}"
    params = {"top_k": top_k}

    last_error: Exception | None = None
    for attempt in range(2):
        if attempt:
            time.sleep(2)  # a cold Space answers 503 for a moment; retrying instantly is useless
        started = time.perf_counter()
        try:
            response = httpx.post(
                _target_url(),
                content=image_bytes,
                headers=headers,
                params=params,
                timeout=settings.hf_timeout_seconds,
            )
        except httpx.HTTPError as exc:
            last_error = exc
            continue
        if response.status_code == 503:
            last_error = HFInferenceError("Classifier is starting up (503).")
            continue
        if response.status_code >= 400:
            raise HFInferenceError(
                f"Classifier request failed ({response.status_code}): {response.text[:200]}"
            )
        latency_ms = round((time.perf_counter() - started) * 1000)
        scores = _parse(response.json())[:top_k]
        meta = _provenance(response, latency_ms, len(image_bytes))
        logger.info(
            "hf classify ok: model=%s@%s via %s in %d ms, top1=%s %.3f (attempt %d)",
            meta["model_id"], meta["model_revision"], meta["endpoint_host"], latency_ms,
            scores[0].label, scores[0].score, attempt + 1,
        )
        return Classification(scores, meta)

    logger.error("hf classify failed: %s", last_error)
    raise HFInferenceError(f"Classifier unavailable: {last_error}")


def probe_endpoint() -> tuple[bool | None, str | None]:
    """(reachable, "model@revision") from our Space's /health. (None, None) if not probe-able."""
    settings = get_settings()
    endpoint = settings.hf_endpoint_url.strip()
    if not endpoint:
        return None, None
    try:
        response = httpx.get(endpoint.rstrip("/") + "/health", timeout=3.0)
    except httpx.HTTPError:
        return False, None
    if response.status_code != 200:
        return False, None
    try:
        body = response.json()
        return True, f"{body['model']}@{body['revision']}"
    except Exception:
        return True, None
