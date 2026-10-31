from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.release import (
    CALIBRATION_TEMPERATURE,
    CONFIDENCE_THRESHOLD,
    CROP_STRATEGY,
    HF_MODEL_ID,
    HF_MODEL_REVISION,
    OOD_THRESHOLD,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    # Always the repo-root .env, even when the process cwd is api/.
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), extra="ignore")

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""

    # Defaults come from shared/release.json. Env / .env overrides that pin.
    hf_model_id: str = HF_MODEL_ID
    hf_model_revision: str = HF_MODEL_REVISION
    hf_token: str = ""
    # Dedicated Inference Endpoint or Space URL; serverless providers do not host this fine-tune.
    hf_endpoint_url: str = ""
    hf_timeout_seconds: float = 30.0
    hf_top_k: int = 3

    crop_strategy: str = CROP_STRATEGY
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    ood_threshold: float = OOD_THRESHOLD
    crop_tta: bool = False  # also classify each crop mirrored and average
    calibration_temperature: float = CALIBRATION_TEMPERATURE
    default_language: str = "en"

    # Phone photos are 3-15 MB; they are decoded with JPEG draft mode and downscaled.
    max_image_bytes: int = 10 * 1024 * 1024
    knowledge_path: str = "data/knowledge/diseases.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
