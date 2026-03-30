"""
Pytest test suite for AxiomESG benchmarking harness.

Tests:
1. Text normalization and matching (strict/relaxed)
2. Real AWFA deduplication correctness
3. Evaluation metric sanity checks
4. Jaccard similarity
5. Bootstrap CI
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure imports
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pytest

from benchmarks.src.utils import (
    bootstrap_ci,
    jaccard_similarity,
    normalize_case,
    normalize_number_str,
    normalize_unit,
    normalize_whitespace,
    strict_match_metric,
    relaxed_match_metric,
    text_hash,
)
from benchmarks.src.real_awfa import (
    apply_real_awfa,
    _jaccard_dedup,
    _compute_uniqueness_scores,
    _compute_confidence_scores,
)
from benchmarks.src.eval import (
    compute_metric_scores,
    compute_per_section_scores,
    compute_groundedness,
    validate_esg_schema,
    evaluate_single_run,
)


# ===========================================================================
# 1. Normalization tests
# ===========================================================================

class TestNormalization:
    def test_normalize_whitespace(self):
        assert normalize_whitespace("  hello   world  ") == "hello world"
        assert normalize_whitespace("\n\thello\n") == "hello"

    def test_normalize_case(self):
        assert normalize_case("Hello WORLD") == "hello world"

    def test_normalize_number_commas(self):
        assert normalize_number_str("245,000") == "245000"
        assert normalize_number_str("1,234,567") == "1234567"

    def test_normalize_number_decimal_commas(self):
        assert normalize_number_str("1,234.56") == "1234.56"

    def test_normalize_number_plain(self):
        assert normalize_number_str("42") == "42"
        assert normalize_number_str("3.14") == "3.14"

    def test_normalize_unit_equivalents(self):
        equivs = [["tCO2e", "tonnes CO2e", "metric tons CO2e"]]
        assert normalize_unit("tonnes CO2e", equivs) == "tco2e"
        assert normalize_unit("tCO2e", equivs) == "tco2e"
        assert normalize_unit("metric tons CO2e", equivs) == "tco2e"

    def test_normalize_unit_no_match(self):
        equivs = [["tCO2e", "tonnes CO2e"]]
        assert normalize_unit("MWh", equivs) == "mwh"

    def test_text_hash_deterministic(self):
        h1 = text_hash("hello world")
        h2 = text_hash("hello world")
        assert h1 == h2
        assert len(h1) == 16


# ===========================================================================
# 2. Metric matching tests
# ===========================================================================

class TestMatching:
    def test_strict_match_exact(self):
        pred = {"name": "Scope 1 GHG Emissions", "value": "50000", "unit": "tCO2e", "year": "2023"}
        gt = {"name": "Scope 1 GHG Emissions", "value": "50000", "unit": "tCO2e", "year": "2023"}
        assert strict_match_metric(pred, gt) is True

    def test_strict_match_case_sensitive(self):
        pred = {"name": "scope 1 ghg emissions", "value": "50000", "unit": "tCO2e", "year": "2023"}
        gt = {"name": "Scope 1 GHG Emissions", "value": "50000", "unit": "tCO2e", "year": "2023"}
        assert strict_match_metric(pred, gt) is False

    def test_strict_match_different_value(self):
        pred = {"name": "Scope 1 GHG Emissions", "value": "50001", "unit": "tCO2e", "year": "2023"}
        gt = {"name": "Scope 1 GHG Emissions", "value": "50000", "unit": "tCO2e", "year": "2023"}
        assert strict_match_metric(pred, gt) is False

    def test_relaxed_match_case_insensitive(self):
        pred = {"name": "scope 1 ghg emissions", "value": "50000", "unit": "tCO2e", "year": "2023"}
        gt = {"name": "Scope 1 GHG Emissions", "value": "50000", "unit": "tCO2e", "year": "2023"}
        assert relaxed_match_metric(pred, gt) is True

    def test_relaxed_match_number_normalization(self):
        pred = {"name": "Scope 1 GHG Emissions", "value": "245,000", "unit": "tCO2e", "year": "2023"}
        gt = {"name": "Scope 1 GHG Emissions", "value": "245000", "unit": "tCO2e", "year": "2023"}
        assert relaxed_match_metric(pred, gt) is True

    def test_relaxed_match_unit_equivalents(self):
        equivs = [["tCO2e", "tonnes CO2e"]]
        pred = {"name": "Scope 1 GHG Emissions", "value": "50000", "unit": "tonnes CO2e", "year": "2023"}
        gt = {"name": "Scope 1 GHG Emissions", "value": "50000", "unit": "tCO2e", "year": "2023"}
        assert relaxed_match_metric(pred, gt, equivs) is True

    def test_relaxed_match_missing_year(self):
        pred = {"name": "Scope 1 GHG Emissions", "value": "50000", "unit": "tCO2e", "year": ""}
        gt = {"name": "Scope 1 GHG Emissions", "value": "50000", "unit": "tCO2e", "year": ""}
        assert relaxed_match_metric(pred, gt) is True


# ===========================================================================
# 3. Jaccard similarity tests
# ===========================================================================

class TestJaccard:
    def test_identical(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert jaccard_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        sim = jaccard_similarity("the quick brown fox", "the quick red dog")
        assert 0.0 < sim < 1.0
        # "the", "quick" in common, out of {"the","quick","brown","fox","red","dog"}
        assert abs(sim - 2/6) < 0.01

    def test_empty(self):
        assert jaccard_similarity("", "") == 1.0
        assert jaccard_similarity("hello", "") == 0.0


# ===========================================================================
# 4. Real AWFA tests
# ===========================================================================

class TestRealAWFA:
    def test_basic_scoring(self):
        sentences = {
            "E": [
                "Total carbon emissions were 50000 tCO2e in 2023.",
                "The company sells widgets and gadgets globally.",
            ],
            "S": [
                "Employee training hours averaged 40 per person in 2023.",
            ],
            "G": [],
        }
        result = apply_real_awfa(sentences)
        assert len(result) > 0
        # All results should be (category, sentence, weight)
        for cat, sent, w in result:
            assert cat in ("E", "S", "G")
            assert isinstance(w, float)
            assert 0.0 <= w <= 1.0

    def test_dedup_removes_near_duplicates(self):
        items = [
            ("E", "carbon emissions were very high in the reporting period", 0.9),
            ("E", "carbon emissions were very high in the reporting period today", 0.8),
            ("S", "employee safety remains a priority", 0.7),
        ]
        deduped, removed = _jaccard_dedup(items, threshold=0.8)
        # First two are very similar, one should be removed
        assert removed >= 1
        assert len(deduped) <= 2

    def test_dedup_keeps_dissimilar(self):
        items = [
            ("E", "carbon emissions data for scope one", 0.9),
            ("S", "employee diversity metrics for the year", 0.8),
            ("G", "board governance independence percentage", 0.7),
        ]
        deduped, removed = _jaccard_dedup(items, threshold=0.8)
        assert removed == 0
        assert len(deduped) == 3

    def test_uniqueness_scores(self):
        sentences = [
            "carbon emissions data report",
            "carbon emissions data update",
            "completely different topic about quantum computing",
        ]
        scores = _compute_uniqueness_scores(sentences)
        assert len(scores) == 3
        # The unique sentence should have higher score
        assert scores[2] > scores[0]  # quantum computing sentence is more unique

    def test_confidence_scores_esg(self):
        sentences = [
            "Total carbon emissions were 50000 tCO2e.",
            "The weather was nice today.",
        ]
        categories = ["E", "E"]
        scores = _compute_confidence_scores(sentences, categories)
        assert len(scores) == 2
        assert scores[0] > scores[1]  # ESG sentence should score higher


# ===========================================================================
# 5. Evaluation metric tests
# ===========================================================================

class TestEvalMetrics:
    def test_perfect_scores(self):
        pred = [
            {"name": "Scope 1", "value": "50000", "unit": "tCO2e", "year": "2023"},
            {"name": "Water", "value": "100", "unit": "ML", "year": "2023"},
        ]
        gt = [
            {"name": "Scope 1", "value": "50000", "unit": "tCO2e", "year": "2023"},
            {"name": "Water", "value": "100", "unit": "ML", "year": "2023"},
        ]
        result = compute_metric_scores(pred, gt, mode="strict")
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_zero_scores(self):
        pred = [
            {"name": "Wrong Metric", "value": "999", "unit": "kg", "year": "2020"},
        ]
        gt = [
            {"name": "Scope 1", "value": "50000", "unit": "tCO2e", "year": "2023"},
        ]
        result = compute_metric_scores(pred, gt, mode="strict")
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_partial_scores(self):
        pred = [
            {"name": "Scope 1", "value": "50000", "unit": "tCO2e", "year": "2023"},
            {"name": "Wrong", "value": "0", "unit": "", "year": ""},
        ]
        gt = [
            {"name": "Scope 1", "value": "50000", "unit": "tCO2e", "year": "2023"},
            {"name": "Water", "value": "100", "unit": "ML", "year": "2023"},
        ]
        result = compute_metric_scores(pred, gt, mode="strict")
        assert result["precision"] == 0.5  # 1 TP, 1 FP
        assert result["recall"] == 0.5  # 1 TP, 1 FN
        assert abs(result["f1"] - 0.5) < 0.01

    def test_schema_validation_valid(self):
        parsed = {
            "metadata": {"source_files": [], "extraction_date": "", "model_provider": "",
                         "model_name": "", "awfa_weights_preserved": True, "algorithm_used": ""},
            "aggregation": {"total_documents": 1, "total_esg_sentences": 10,
                            "total_weighted_blocks": 5, "ocr_used": False},
            "environmental": {
                "narrative": "Test", "metrics": [], "confidence_score": 0.5, "top_evidence": []
            },
            "social": {
                "narrative": "Test", "metrics": [], "confidence_score": 0.5, "top_evidence": []
            },
            "governance": {
                "narrative": "Test", "metrics": [], "confidence_score": 0.5, "top_evidence": []
            },
        }
        valid, err = validate_esg_schema(parsed)
        assert valid is True

    def test_schema_validation_missing_section(self):
        parsed = {
            "metadata": {},
            "aggregation": {},
            "environmental": {"narrative": "", "metrics": [], "confidence_score": 0.5, "top_evidence": []},
            "social": {"narrative": "", "metrics": [], "confidence_score": 0.5, "top_evidence": []},
            # missing governance
        }
        valid, err = validate_esg_schema(parsed)
        assert valid is False
        assert "governance" in err

    def test_groundedness_exact_match(self):
        pred = [
            {"name": "Scope 1", "value": "50000", "unit": "tCO2e", "year": "2023",
             "source_text": "Total emissions were 50000 tCO2e."},
        ]
        raw_text = "In 2023, total emissions were 50000 tCO2e. The company is headquartered in London."
        result = compute_groundedness(pred, raw_text)
        assert result["grounded_metric_rate"] == 1.0

    def test_groundedness_no_match(self):
        pred = [
            {"name": "Scope 1", "value": "50000", "unit": "tCO2e", "year": "2023",
             "source_text": "This text does not appear anywhere in the document."},
        ]
        raw_text = "Completely different content about financial markets and stock trading."
        result = compute_groundedness(pred, raw_text, jaccard_threshold=0.95)
        assert result["grounded_metric_rate"] < 1.0


# ===========================================================================
# 6. Bootstrap CI tests
# ===========================================================================

class TestBootstrapCI:
    def test_constant_values(self):
        values = [0.5] * 100
        mean, lower, upper = bootstrap_ci(values, n_bootstrap=500)
        assert abs(mean - 0.5) < 0.01
        assert abs(lower - 0.5) < 0.01
        assert abs(upper - 0.5) < 0.01

    def test_varied_values(self):
        values = list(range(100))
        mean, lower, upper = bootstrap_ci(values, n_bootstrap=1000)
        assert lower < mean < upper
        assert abs(mean - 49.5) < 2.0

    def test_empty(self):
        mean, lower, upper = bootstrap_ci([])
        assert mean == 0.0
        assert lower == 0.0
        assert upper == 0.0
