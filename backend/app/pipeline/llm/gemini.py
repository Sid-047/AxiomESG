from __future__ import annotations

from typing import Any, Dict

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.pipeline.llm.base import LLMResult


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    def generate(self, prompt: str, request_id: str) -> LLMResult:
        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        model = self.settings.gemini_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": self.settings.gemini_api_key}
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        timeout = httpx.Timeout(45.0)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, params=params, json=payload)
            resp.raise_for_status()
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            usage = data.get("usageMetadata", {})
            return LLMResult(text=text, usage=usage, model_name=model)
