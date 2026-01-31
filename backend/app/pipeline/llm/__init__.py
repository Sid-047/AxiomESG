from __future__ import annotations

from app.core.config import Settings
from app.pipeline.llm.azure_openai import AzureOpenAIClient
from app.pipeline.llm.gemini import GeminiClient
from app.pipeline.llm.openrouter import OpenRouterClient


def get_llm_client(settings: Settings):
    provider = settings.llm_provider.lower()
    if provider == "openrouter":
        return OpenRouterClient(settings)
    if provider == "azure_openai":
        return AzureOpenAIClient(settings)
    if provider == "gemini":
        return GeminiClient(settings)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
