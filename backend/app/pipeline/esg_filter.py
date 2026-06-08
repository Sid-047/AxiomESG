from __future__ import annotations

import re
from typing import Dict, List, Tuple

from app.core.config import Settings

DEFAULT_E = [
    "emission",
    "carbon",
    "climate",
    "energy",
    "renewable",
    "water",
    "waste",
    "biodiversity",
    "pollution",
    "recycling",
]
DEFAULT_S = [
    "diversity",
    "inclusion",
    "labor",
    "health",
    "safety",
    "community",
    "human rights",
    "training",
    "employee",
    "privacy",
]
DEFAULT_G = [
    "board",
    "governance",
    "ethics",
    "compliance",
    "risk",
    "audit",
    "shareholder",
    "transparency",
    "anti-corruption",
    "policy",
]


def _load_keywords(settings: Settings) -> Dict[str, List[str]]:
    def parse(value: str, fallback: List[str]) -> List[str]:
        if not value.strip():
            return fallback
        return [v.strip().lower() for v in value.split(",") if v.strip()]

    return {
        "E": parse(settings.esg_keywords_env, DEFAULT_E),
        "S": parse(settings.esg_keywords_soc, DEFAULT_S),
        "G": parse(settings.esg_keywords_gov, DEFAULT_G),
    }


def _split_sentences(text: str) -> List[str]:
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split by sentence boundaries if long, else keep as is
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p = p.strip()
            if p:
                chunks.append(p)
    return chunks


def filter_esg_sentences(text: str, settings: Settings) -> Dict[str, List[str]]:
    keywords = _load_keywords(settings)
    sentences = _split_sentences(text)
    result: Dict[str, List[str]] = {"E": [], "S": [], "G": [], "UNKNOWN": []}
    
    for sentence in sentences:
        lowered = sentence.lower()
        matched = False
        for category in ("E", "S", "G"):
            if any(k in lowered for k in keywords[category]):
                result[category].append(sentence)
                matched = True
        
        if not matched and getattr(settings, "disable_pre_algorithm_filter", False):
            result["UNKNOWN"].append(sentence)
            
    return result


def flatten_esg(result: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    flat: List[Tuple[str, str]] = []
    for category in ("E", "S", "G", "UNKNOWN"):
        for sentence in result.get(category, []):
            flat.append((category, sentence))
    return flat
