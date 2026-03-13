"""Tests for the ESG report generator."""

from app.pipeline.report import generate_report


def _sample_result():
    return {
        "metadata": {
            "source_files": ["test.pdf"],
            "extraction_date": "2025-01-01T00:00:00Z",
            "model_provider": "openrouter",
            "model_name": "test-model",
            "awfa_weights_preserved": True,
            "algorithm_used": "heuristic",
        },
        "aggregation": {
            "total_documents": 1,
            "total_esg_sentences": 5,
            "total_weighted_blocks": 3,
            "ocr_used": False,
        },
        "environmental": {
            "narrative": "Carbon emissions were reduced by 15%.",
            "metrics": [
                {"name": "CO2 Reduction", "value": "15", "unit": "%", "year": "2024", "source_text": "CO2 reduced by 15% in 2024."}
            ],
            "confidence_score": 0.7,
            "top_evidence": [
                {"text": "CO2 reduced by 15%.", "weight": 0.85, "category": "E", "source_file": "test.pdf"}
            ],
        },
        "social": {
            "narrative": "Employee safety improved.",
            "metrics": [],
            "confidence_score": 0.4,
            "top_evidence": [],
        },
        "governance": {
            "narrative": "Not found in provided documents.",
            "metrics": [],
            "confidence_score": 0.0,
            "top_evidence": [],
        },
    }


def test_report_not_empty():
    report = generate_report(_sample_result())
    assert len(report) > 0


def test_report_contains_header():
    report = generate_report(_sample_result())
    assert "# AxiomESG" in report


def test_report_contains_algorithm():
    report = generate_report(_sample_result())
    assert "Heuristic AWFA" in report


def test_report_contains_sections():
    report = generate_report(_sample_result())
    assert "## Environmental" in report
    assert "## Social" in report
    assert "## Governance" in report


def test_report_contains_metrics():
    report = generate_report(_sample_result())
    assert "CO2 Reduction" in report
    assert "15" in report
