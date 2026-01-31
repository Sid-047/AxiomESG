from __future__ import annotations

import time
from typing import Any, Dict

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import Settings


AZURE_API_VERSION = "2024-02-29-preview"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def azure_read_document(data: bytes, content_type: str, settings: Settings) -> str:
    endpoint = settings.azure_docintel_endpoint.rstrip("/")
    if not endpoint or not settings.azure_docintel_key:
        raise ValueError("Azure Document Intelligence not configured.")

    url = f"{endpoint}/documentintelligence/documentModels/prebuilt-read:analyze"
    params = {"api-version": AZURE_API_VERSION}
    headers = {
        "Ocp-Apim-Subscription-Key": settings.azure_docintel_key,
        "Content-Type": content_type,
    }
    timeout = httpx.Timeout(30.0)

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, params=params, headers=headers, content=data)
        resp.raise_for_status()
        operation = resp.headers.get("operation-location")
        if not operation:
            raise RuntimeError("OCR operation location missing.")

        for _ in range(20):
            poll = client.get(operation, headers={"Ocp-Apim-Subscription-Key": settings.azure_docintel_key})
            poll.raise_for_status()
            payload: Dict[str, Any] = poll.json()
            status = payload.get("status", "").lower()
            if status == "succeeded":
                analyze_result = payload.get("analyzeResult", {})
                content = analyze_result.get("content", "")
                return content.strip()
            if status == "failed":
                raise RuntimeError("OCR failed in Azure Document Intelligence.")
            time.sleep(1.5)
    raise RuntimeError("OCR polling timed out.")
