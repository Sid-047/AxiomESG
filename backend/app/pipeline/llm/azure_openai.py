from __future__ import annotations

from typing import Any, Dict

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.pipeline.llm.base import LLMResult


class AzureOpenAIClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    def generate(self, prompt: str, request_id: str) -> LLMResult:
        if not self.settings.azure_openai_endpoint or not self.settings.azure_openai_api_key:
            raise ValueError("Azure OpenAI is not configured.")
        if not self.settings.azure_openai_deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT is not configured.")
        endpoint = self.settings.azure_openai_endpoint.rstrip("/")
        url = f"{endpoint}/openai/deployments/{self.settings.azure_openai_deployment}/chat/completions"
        params = {"api-version": self.settings.azure_openai_api_version}
        headers = {"api-key": self.settings.azure_openai_api_key, "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "You are a strict JSON generator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        timeout = httpx.Timeout(45.0)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, params=params, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            model = data.get("model", self.settings.azure_openai_deployment)
            return LLMResult(text=text, usage=usage, model_name=model)
