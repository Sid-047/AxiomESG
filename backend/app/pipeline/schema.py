from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class EvidenceSpan(BaseModel):
    text: str
    weight: float
    category: Literal["E", "S", "G"]
    source_file: str


class Metric(BaseModel):
    name: str
    value: str
    unit: Optional[str] = None
    year: Optional[str] = None
    source_text: str


class ESGSection(BaseModel):
    narrative: str
    metrics: List[Metric]
    confidence_score: float = Field(ge=0.0, le=1.0)
    top_evidence: List[EvidenceSpan]

    @field_validator("narrative")
    @classmethod
    def narrative_not_empty(cls, value: str) -> str:
        return value.strip()


class ESGOutputMetadata(BaseModel):
    source_files: List[str]
    extraction_date: str
    model_provider: str
    model_name: str
    awfa_weights_preserved: bool


class ESGAggregation(BaseModel):
    total_documents: int
    total_esg_sentences: int
    total_weighted_blocks: int
    ocr_used: bool


class ESGOutput(BaseModel):
    metadata: ESGOutputMetadata
    aggregation: ESGAggregation
    environmental: ESGSection
    social: ESGSection
    governance: ESGSection
