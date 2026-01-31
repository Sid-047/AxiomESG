"""JSON schema validation for ESG output."""
from typing import Dict, List, Any
from pydantic import BaseModel, Field, validator

class Metric(BaseModel):
    """Metric schema."""
    name: str
    value: str
    unit: str = ""
    year: str = None
    source_text: str = None

class ESGSection(BaseModel):
    """ESG section schema."""
    narrative: str = ""
    metrics: List[Metric] = []
    confidence_score: float = Field(ge=0.0, le=1.0)

class Metadata(BaseModel):
    """Metadata schema."""
    source_files: List[str] = []
    extraction_date: str = ""
    awfa_weights_preserved: bool = True

class AggregationInfo(BaseModel):
    """Aggregation info schema."""
    total_documents: int = 0
    awfa_applied: bool = True
    weighted_content_blocks: int = 0

class ESGOutput(BaseModel):
    """Complete ESG output schema."""
    metadata: Metadata
    environmental: ESGSection
    social: ESGSection
    governance: ESGSection
    aggregation_info: AggregationInfo

def validate_esg_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate ESG JSON against schema.
    Returns validated and normalized data.
    """
    try:
        validated = ESGOutput(**data)
        # Pydantic v2 uses model_dump(), v1 uses dict()
        if hasattr(validated, 'model_dump'):
            return validated.model_dump()
        else:
            return validated.dict()
    except Exception as e:
        raise ValueError(f"Schema validation failed: {str(e)}")
