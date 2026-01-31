from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple


@dataclass
class LLMResult:
    text: str
    usage: Dict[str, Any]
    model_name: str


class LLMClient(Protocol):
    def generate(self, prompt: str, request_id: str) -> LLMResult:
        ...
