from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""

    confidence_threshold: float = 0.6
    default_language: str = "en"

    max_image_bytes: int = 10 * 1024 * 1024
    knowledge_path: str = "data/knowledge/diseases.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
