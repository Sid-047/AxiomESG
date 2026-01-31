from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AxiomESG"
    log_level: str = "INFO"

    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")

    max_file_mb: int = Field(default=25, alias="MAX_FILE_MB")
    max_total_mb: int = Field(default=50, alias="MAX_TOTAL_MB")
    job_poll_ttl_seconds: int = Field(default=3600, alias="JOB_POLL_TTL_SECONDS")

    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="openrouter/auto", alias="OPENROUTER_MODEL")

    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(default="2024-02-15-preview", alias="AZURE_OPENAI_API_VERSION")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")

    azure_docintel_endpoint: str = Field(default="", alias="AZURE_DOCINTEL_ENDPOINT")
    azure_docintel_key: str = Field(default="", alias="AZURE_DOCINTEL_KEY")

    redis_url: str = Field(default="", alias="REDIS_URL")

    preview_chars: int = Field(default=2000, alias="RAW_TEXT_PREVIEW_CHARS")

    esg_keywords_env: str = Field(default="", alias="ESG_KEYWORDS_E")
    esg_keywords_soc: str = Field(default="", alias="ESG_KEYWORDS_S")
    esg_keywords_gov: str = Field(default="", alias="ESG_KEYWORDS_G")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    def max_total_bytes(self) -> int:
        return self.max_total_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
