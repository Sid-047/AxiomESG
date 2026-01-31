from app.pipeline.schema import ESGOutput


def test_schema_validation():
    payload = {
        "metadata": {
            "source_files": ["a.pdf"],
            "extraction_date": "2025-01-01T00:00:00Z",
            "model_provider": "openrouter",
            "model_name": "test",
            "awfa_weights_preserved": True,
        },
        "aggregation": {
            "total_documents": 1,
            "total_esg_sentences": 2,
            "total_weighted_blocks": 2,
            "ocr_used": False,
        },
        "environmental": {
            "narrative": "Not found in provided documents.",
            "metrics": [],
            "confidence_score": 0.0,
            "top_evidence": [],
        },
        "social": {
            "narrative": "We track safety outcomes.",
            "metrics": [
                {
                    "name": "Injury rate",
                    "value": "2.1",
                    "unit": "per 200k hours",
                    "year": "2024",
                    "source_text": "Injury rate was 2.1 per 200k hours in 2024.",
                }
            ],
            "confidence_score": 0.6,
            "top_evidence": [
                {"text": "Injury rate was 2.1 per 200k hours in 2024.", "weight": 0.7, "category": "S", "source_file": "a.pdf"}
            ],
        },
        "governance": {
            "narrative": "Not found in provided documents.",
            "metrics": [],
            "confidence_score": 0.0,
            "top_evidence": [],
        },
    }
    model = ESGOutput.model_validate(payload)
    assert model.aggregation.total_documents == 1
