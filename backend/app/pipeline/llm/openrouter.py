from __future__ import annotations

from typing import Any, Dict

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.pipeline.llm.base import LLMResult


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    def generate(self, prompt: str, request_id: str) -> LLMResult:
        if not self.settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": "AxiomESG",
        }
        payload: Dict[str, Any] = {
            "model": self.settings.openrouter_model,
            "messages": [
                {"role": "system", "content": "You are a strict JSON generator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        timeout = httpx.Timeout(45.0)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            model = data.get("model", self.settings.openrouter_model)
            return LLMResult(text=text, usage=usage, model_name=model)
