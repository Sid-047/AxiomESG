"""
Utility functions for the AxiomESG benchmarking harness.

Provides:
- Deterministic hashing of text content
- Text normalization for matching
- Bootstrap confidence interval calculations
- Numeric normalization
- Jaccard similarity
- Logging setup
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_benchmark_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(f"axiomesg.bench.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def text_hash(text: str) -> str:
    """SHA-256 of UTF-8-encoded text, truncated to 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# IDs and metadata
# ---------------------------------------------------------------------------

def make_run_id() -> str:
    return str(uuid.uuid4())


def get_git_commit() -> str:
    """Best-effort retrieval of current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_case(text: str) -> str:
    return text.lower()


def normalize_number_str(val: str) -> str:
    """
    Normalize numeric strings for comparison.
    '245,000' -> '245000'
    '1,234.56' -> '1234.56'
    Strip leading/trailing whitespace.
    """
    val = val.strip()
    # Remove commas used as thousands separators (not decimal commas)
    # Heuristic: if there's a dot after the last comma, commas are thousands seps
    if "," in val and "." in val:
        if val.rfind(",") < val.rfind("."):
            val = val.replace(",", "")
    elif "," in val:
        # No dot present — try to determine if comma is thousands or decimal
        # If comma-separated groups of 3 digits, it's thousands
        if re.match(r"^\d{1,3}(,\d{3})+$", val):
            val = val.replace(",", "")
    return val


def normalize_unit(unit: str, equivalents: List[List[str]] | None = None) -> str:
    """Normalize a unit string using known equivalence lists."""
    if not unit:
        return ""
    unit_stripped = normalize_whitespace(normalize_case(unit))
    if equivalents:
        for group in equivalents:
            normed_group = [normalize_whitespace(normalize_case(u)) for u in group]
            if unit_stripped in normed_group:
                return normed_group[0]  # canonical form
    return unit_stripped


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def strict_match_metric(
    pred: Dict[str, str], gt: Dict[str, str]
) -> bool:
    """Exact match on (name, value, unit, year)."""
    return (
        pred.get("name", "").strip() == gt.get("name", "").strip()
        and pred.get("value", "").strip() == gt.get("value", "").strip()
        and (pred.get("unit") or "").strip() == (gt.get("unit") or "").strip()
        and (pred.get("year") or "").strip() == (gt.get("year") or "").strip()
    )


def relaxed_match_metric(
    pred: Dict[str, str],
    gt: Dict[str, str],
    unit_equivalents: List[List[str]] | None = None,
) -> bool:
    """Relaxed match: case-insensitive, whitespace-normalized, numeric normalization."""
    def _norm(v: str) -> str:
        return normalize_case(normalize_whitespace(normalize_number_str(v)))

    name_match = _norm(pred.get("name", "")) == _norm(gt.get("name", ""))
    value_match = _norm(pred.get("value", "")) == _norm(gt.get("value", ""))
    unit_pred = normalize_unit(pred.get("unit") or "", unit_equivalents)
    unit_gt = normalize_unit(gt.get("unit") or "", unit_equivalents)
    unit_match = unit_pred == unit_gt
    year_pred = _norm(pred.get("year") or "")
    year_gt = _norm(gt.get("year") or "")
    year_match = year_pred == year_gt
    return name_match and value_match and unit_match and year_match


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(normalize_case(text_a).split())
    tokens_b = set(normalize_case(text_b).split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap (ci)% confidence interval.

    Returns: (mean, lower_bound, upper_bound)
    """
    if not values:
        return (0.0, 0.0, 0.0)

    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean_val = float(arr.mean())
    return (mean_val, lower, upper)


def format_ci(mean: float, lower: float, upper: float, decimals: int = 3) -> str:
    """Format a value with CI as 'mean [lower, upper]'."""
    fmt = f".{decimals}f"
    return f"{mean:{fmt}} [{lower:{fmt}}, {upper:{fmt}}]"


# ---------------------------------------------------------------------------
# CSV column schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "run_id",
    "git_commit",
    "timestamp",
    "seed",
    "variant_id",
    "variant_label",
    "algorithm_used",
    "filter_on",
    "weight_on",
    "ocr_mode",
    "awfa_mode",
    "bert_mode",
    "llm_provider",
    "llm_model_name",
    "doc_id",
    "doc_path",
    "doc_type",
    "is_synthetic",
    "is_real",
    "is_scanned",
    "sector",
    "ground_truth_available",
    "augmentation_round",
    "benchmark_mode",
    "mock_llm_used",
    # Stage timings (ms)
    "extract_ms",
    "filter_ms",
    "weight_ms",
    "ocr_ms",
    "intelligence_ms",
    "validate_ms",
    "total_latency_ms",
    # Outputs
    "output_json_path",
    "raw_text_hash",
    "raw_text_preview",
    "extracted_char_count",
    "evidence_char_count",
    # Validity
    "json_parse_success",
    "schema_valid",
    # Per-section strict
    "env_strict_precision",
    "env_strict_recall",
    "env_strict_f1",
    "soc_strict_precision",
    "soc_strict_recall",
    "soc_strict_f1",
    "gov_strict_precision",
    "gov_strict_recall",
    "gov_strict_f1",
    "overall_strict_precision",
    "overall_strict_recall",
    "overall_strict_f1",
    # Per-section relaxed
    "env_relaxed_precision",
    "env_relaxed_recall",
    "env_relaxed_f1",
    "soc_relaxed_precision",
    "soc_relaxed_recall",
    "soc_relaxed_f1",
    "gov_relaxed_precision",
    "gov_relaxed_recall",
    "gov_relaxed_f1",
    "overall_relaxed_precision",
    "overall_relaxed_recall",
    "overall_relaxed_f1",
    # Environmental-specific
    "environmental_precision",
    "environmental_recall",
    "environmental_f1",
    "environmental_miss_rate",
    "environmental_ERRS",
    "emissions_ERRS",
    "energy_ERRS",
    "water_ERRS",
    "waste_ERRS",
    "pollution_ERRS",
    "biodiversity_ERRS",
    "circularity_ERRS",
    "compliance_ERRS",
    # Rates
    "missing_year_rate",
    "wrong_year_rate",
    "unit_missing_rate",
    # Groundedness
    "grounded_metric_rate",
    "unsupported_metric_rate",
    "narrative_grounded_rate",
    # Evidence quality
    "evidence_hit_rate",
    "recall_at_10",
    "recall_at_30",
    "recall_at_60",
    "dedup_rate",
    "compression_ratio",
    # Efficiency
    "llm_prompt_chars",
    "llm_output_chars",
    "cost_proxy",
    "ocr_called",
    # Errors
    "variant_skipped_reason",
    "error_message",
]

