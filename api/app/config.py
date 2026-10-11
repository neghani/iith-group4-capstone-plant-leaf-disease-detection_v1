from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""

    hf_model_id: str = "neghani/plantvillage-mobilenetv2"
    hf_model_revision: str = "v1.0"
    hf_token: str = ""
    # Dedicated Inference Endpoint or Space URL; serverless providers do not host this fine-tune.
    hf_endpoint_url: str = ""
    hf_timeout_seconds: float = 30.0
    hf_top_k: int = 3

    # Crop strategy (shared/vision/strategies.py, A-E) and the confidence threshold chosen with
    # it on the tune split by model/eval_strategies.py; see model/outputs/final_report.md.
    # No threshold reached 70% correct "ok" answers on tune; 0.85 had the highest precision.
    crop_strategy: str = "B"
    confidence_threshold: float = 0.85
    crop_tta: bool = False  # also classify each crop mirrored and average
    default_language: str = "en"

    # Phone photos are 3-15 MB; they are decoded with JPEG draft mode and downscaled.
    max_image_bytes: int = 25 * 1024 * 1024
    knowledge_path: str = "data/knowledge/diseases.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
