"""
Evaluation Metrics for AxiomESG Benchmark
==========================================

Implements all evaluation metrics:
1. Validity: JSON parse success, schema validation
2. Metric extraction quality: strict/relaxed precision, recall, F1
3. Groundedness / hallucination control
4. Evidence quality (AWFA novelty metrics)
5. Efficiency metrics

All metrics are computed per-run and aggregated in the CSV.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.src.utils import (
    jaccard_similarity,
    normalize_case,
    normalize_number_str,
    normalize_unit,
    normalize_whitespace,
    strict_match_metric,
    relaxed_match_metric,
)


# ---------------------------------------------------------------------------
# Pydantic schema for quick validation (mirrors backend schema)
# ---------------------------------------------------------------------------

REQUIRED_TOP_KEYS = {"metadata", "aggregation", "environmental", "social", "governance"}
REQUIRED_SECTION_KEYS = {"narrative", "metrics", "confidence_score", "top_evidence"}
REQUIRED_METRIC_KEYS = {"name", "value", "source_text"}


def validate_esg_schema(parsed: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a parsed ESG JSON against the expected schema.
    Returns (is_valid, error_message).
    """
    if not isinstance(parsed, dict):
        return False, "Root is not a dict"

    missing_top = REQUIRED_TOP_KEYS - set(parsed.keys())
    if missing_top:
        return False, f"Missing top-level keys: {missing_top}"

    for section_name in ("environmental", "social", "governance"):
        section = parsed.get(section_name, {})
        if not isinstance(section, dict):
            return False, f"{section_name} is not a dict"
        missing_section = REQUIRED_SECTION_KEYS - set(section.keys())
        if missing_section:
            return False, f"{section_name} missing keys: {missing_section}"
        metrics = section.get("metrics", [])
        if not isinstance(metrics, list):
            return False, f"{section_name}.metrics is not a list"
        for i, m in enumerate(metrics):
            if not isinstance(m, dict):
                return False, f"{section_name}.metrics[{i}] is not a dict"
            missing_m = REQUIRED_METRIC_KEYS - set(m.keys())
            if missing_m:
                return False, f"{section_name}.metrics[{i}] missing: {missing_m}"

    return True, ""


# ---------------------------------------------------------------------------
# Metric extraction scoring
# ---------------------------------------------------------------------------

def _compute_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_metric_scores(
    predicted_metrics: List[Dict[str, str]],
    gt_metrics: List[Dict[str, str]],
    unit_equivalents: List[List[str]] | None = None,
    mode: str = "strict",
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for metric extraction.

    Args:
        predicted_metrics: List of predicted metric dicts
        gt_metrics: List of ground truth metric dicts
        unit_equivalents: List of unit equivalence groups for relaxed matching
        mode: "strict" or "relaxed"

    Returns:
        Dict with precision, recall, f1
    """
    if mode == "strict":
        match_fn = strict_match_metric
    else:
        match_fn = lambda p, g: relaxed_match_metric(p, g, unit_equivalents)

    # Greedy matching: each GT metric matched at most once
    gt_matched = [False] * len(gt_metrics)
    tp = 0
    fp = 0

    for pred in predicted_metrics:
        found = False
        for j, gt in enumerate(gt_metrics):
            if not gt_matched[j] and match_fn(pred, gt):
                gt_matched[j] = True
                tp += 1
                found = True
                break
        if not found:
            fp += 1

    fn = sum(1 for matched in gt_matched if not matched)
    precision, recall, f1 = _compute_prf(tp, fp, fn)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_per_section_scores(
    parsed: Dict[str, Any],
    gt_metrics: List[Dict[str, Any]],
    unit_equivalents: List[List[str]] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute strict and relaxed scores per section (E/S/G) and overall.

    Args:
        parsed: Parsed ESG JSON output
        gt_metrics: Ground truth metrics with 'category' field

    Returns:
        Nested dict: {section: {strict_precision, ..., relaxed_f1, ...}}
    """
    section_map = {"E": "environmental", "S": "social", "G": "governance"}
    results = {}

    # Separate GT by category
    gt_by_cat = {"E": [], "S": [], "G": []}
    for m in gt_metrics:
        cat = m.get("category", "E")
        gt_by_cat[cat].append(m)

    all_pred = []
    all_gt = []

    for cat, section_name in section_map.items():
        section = parsed.get(section_name, {})
        pred_metrics = section.get("metrics", [])
        gt_list = gt_by_cat[cat]
        all_pred.extend(pred_metrics)
        all_gt.extend(gt_list)

        prefix = {"E": "env", "S": "soc", "G": "gov"}[cat]
        strict = compute_metric_scores(pred_metrics, gt_list, unit_equivalents, mode="strict")
        relaxed = compute_metric_scores(pred_metrics, gt_list, unit_equivalents, mode="relaxed")

        results[f"{prefix}_strict_precision"] = strict["precision"]
        results[f"{prefix}_strict_recall"] = strict["recall"]
        results[f"{prefix}_strict_f1"] = strict["f1"]
        results[f"{prefix}_relaxed_precision"] = relaxed["precision"]
        results[f"{prefix}_relaxed_recall"] = relaxed["recall"]
        results[f"{prefix}_relaxed_f1"] = relaxed["f1"]

    # Overall
    strict_overall = compute_metric_scores(all_pred, all_gt, unit_equivalents, mode="strict")
    relaxed_overall = compute_metric_scores(all_pred, all_gt, unit_equivalents, mode="relaxed")
    results["overall_strict_precision"] = strict_overall["precision"]
    results["overall_strict_recall"] = strict_overall["recall"]
    results["overall_strict_f1"] = strict_overall["f1"]
    results["overall_relaxed_precision"] = relaxed_overall["precision"]
    results["overall_relaxed_recall"] = relaxed_overall["recall"]
    results["overall_relaxed_f1"] = relaxed_overall["f1"]

    return results


# ---------------------------------------------------------------------------
# Year and unit quality rates
# ---------------------------------------------------------------------------

def compute_year_unit_rates(
    predicted_metrics: List[Dict[str, str]],
    gt_metrics: List[Dict[str, str]],
    unit_equivalents: List[List[str]] | None = None,
) -> Dict[str, float]:
    """
    Compute rates for missing/wrong years and missing units.
    Uses relaxed matching on (name, value) to pair metrics, then checks year/unit.
    """
    if not predicted_metrics:
        return {
            "missing_year_rate": 1.0,
            "wrong_year_rate": 0.0,
            "unit_missing_rate": 1.0,
        }

    missing_year = 0
    wrong_year = 0
    unit_missing = 0
    total = len(predicted_metrics)

    # Match predicted to GT for year/unit comparison
    gt_matched = [False] * len(gt_metrics)
    for pred in predicted_metrics:
        pred_year = (pred.get("year") or "").strip()
        pred_unit = (pred.get("unit") or "").strip()

        if not pred_year:
            missing_year += 1
        if not pred_unit:
            unit_missing += 1

        # Try to find a matching GT to check year correctness
        for j, gt in enumerate(gt_metrics):
            if gt_matched[j]:
                continue
            # Match on name+value (relaxed)
            if (normalize_case(normalize_whitespace(pred.get("name", "")))
                == normalize_case(normalize_whitespace(gt.get("name", "")))):
                gt_matched[j] = True
                gt_year = (gt.get("year") or "").strip()
                if pred_year and gt_year and pred_year != gt_year:
                    wrong_year += 1
                break

    return {
        "missing_year_rate": round(missing_year / total, 4) if total else 0.0,
        "wrong_year_rate": round(wrong_year / total, 4) if total else 0.0,
        "unit_missing_rate": round(unit_missing / total, 4) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Groundedness / Hallucination detection
# ---------------------------------------------------------------------------

def compute_groundedness(
    predicted_metrics: List[Dict[str, str]],
    raw_text: str,
    jaccard_threshold: float = 0.9,
) -> Dict[str, float]:
    """
    Compute groundedness metrics.

    grounded_metric_rate: fraction of predicted metrics whose source_text
    appears in (or highly overlaps with) the raw extracted text.
    """
    if not predicted_metrics:
        return {
            "grounded_metric_rate": 0.0,
            "unsupported_metric_rate": 1.0,
        }

    grounded = 0
    raw_lower = raw_text.lower()

    for m in predicted_metrics:
        source = (m.get("source_text") or "").strip()
        if not source:
            continue

        # Exact substring check
        if source.lower() in raw_lower:
            grounded += 1
            continue

        # Fuzzy: token Jaccard >= threshold
        sim = jaccard_similarity(source, raw_text)
        if sim >= jaccard_threshold:
            grounded += 1
            continue

        # Check shorter windows of raw_text
        # Sliding window of similar length to source_text
        source_tokens = set(source.lower().split())
        if len(source_tokens) > 2:
            raw_sentences = re.split(r'[.!?\n]+', raw_text)
            for rs in raw_sentences:
                if jaccard_similarity(source, rs) >= jaccard_threshold:
                    grounded += 1
                    break

    total = len(predicted_metrics)
    grounded_rate = round(grounded / total, 4)
    return {
        "grounded_metric_rate": grounded_rate,
        "unsupported_metric_rate": round(1.0 - grounded_rate, 4),
    }


def compute_narrative_groundedness(
    parsed: Dict[str, Any],
    raw_text: str,
    n_samples: int = 5,
) -> float:
    """
    Sample sentences from narratives and check overlap with raw text.
    Returns narrative_grounded_rate.
    """
    narratives = []
    for section in ("environmental", "social", "governance"):
        narrative = parsed.get(section, {}).get("narrative", "")
        if narrative and narrative != "Not found in provided documents.":
            # Split into sentences
            sents = re.split(r'(?<=[.!?])\s+', narrative)
            narratives.extend(s.strip() for s in sents if len(s.strip()) > 10)

    if not narratives:
        return 0.0

    # Sample
    samples = narratives[:n_samples] if len(narratives) <= n_samples else narratives[:n_samples]
    raw_lower = raw_text.lower()

    grounded = 0
    for sent in samples:
        sent_lower = sent.lower()
        # Check if any significant phrase from the narrative exists in raw text
        words = sent_lower.split()
        if len(words) >= 4:
            # Check 4-grams
            for i in range(len(words) - 3):
                phrase = " ".join(words[i:i+4])
                if phrase in raw_lower:
                    grounded += 1
                    break
            else:
                # Jaccard with raw text sentences
                raw_sents = re.split(r'[.!?\n]+', raw_text)
                for rs in raw_sents:
                    if jaccard_similarity(sent, rs) >= 0.6:
                        grounded += 1
                        break

    return round(grounded / len(samples), 4) if samples else 0.0


# ---------------------------------------------------------------------------
# Evidence quality metrics
# ---------------------------------------------------------------------------

def compute_evidence_quality(
    evidence_spans: List[Dict[str, Any]],
    gt_metrics: List[Dict[str, Any]],
    raw_text: str,
    k_values: List[int] = [10, 30, 60],
) -> Dict[str, float]:
    """
    Compute evidence hit rate and recall@K.

    evidence_hit_rate: fraction of GT metric source lines present in evidence spans.
    recall_at_k: fraction of GT source lines found in top-K evidence spans.
    """
    gt_source_texts = [
        m.get("source_text", "").strip().lower()
        for m in gt_metrics
        if m.get("source_text", "").strip()
    ]

    if not gt_source_texts:
        result = {"evidence_hit_rate": 0.0}
        for k in k_values:
            result[f"recall_at_{k}"] = 0.0
        return result

    evidence_texts = [
        (e.get("text", "") or "").strip().lower()
        for e in evidence_spans
    ]

    def _hits_at_k(k: int) -> int:
        top_k = evidence_texts[:k]
        hits = 0
        for gt_src in gt_source_texts:
            for ev in top_k:
                if gt_src in ev or ev in gt_src:
                    hits += 1
                    break
                elif jaccard_similarity(gt_src, ev) >= 0.7:
                    hits += 1
                    break
        return hits

    result = {}
    total_gt = len(gt_source_texts)

    # Overall hit rate (all evidence)
    all_hits = _hits_at_k(len(evidence_texts))
    result["evidence_hit_rate"] = round(all_hits / total_gt, 4)

    for k in k_values:
        hits = _hits_at_k(k)
        result[f"recall_at_{k}"] = round(hits / total_gt, 4)

    return result


def compute_compression_stats(
    evidence_spans: List[Dict[str, Any]],
    raw_text: str,
    pre_dedup_count: int = 0,
    post_dedup_count: int = 0,
) -> Dict[str, float]:
    """
    Compute compression and dedup statistics.

    compression_ratio = evidence_chars / extracted_chars
    dedup_rate = % of candidates removed by dedup
    """
    evidence_chars = sum(len(e.get("text", "")) for e in evidence_spans)
    extracted_chars = len(raw_text)

    compression_ratio = round(evidence_chars / max(extracted_chars, 1), 4)
    dedup_rate = 0.0
    if pre_dedup_count > 0:
        dedup_rate = round((pre_dedup_count - post_dedup_count) / pre_dedup_count, 4)

    return {
        "evidence_char_count": evidence_chars,
        "extracted_char_count": extracted_chars,
        "compression_ratio": compression_ratio,
        "dedup_rate": dedup_rate,
    }


# ---------------------------------------------------------------------------
# ERRS (Environmental Reporting Readiness Score)
# ---------------------------------------------------------------------------

# Environmental subcategory keyword mappings
ENV_SUBCATEGORY_KEYWORDS = {
    "emissions": ["emission", "ghg", "carbon", "co2", "scope 1", "scope 2", "scope 3", "greenhouse", "carbon intensity"],
    "energy": ["energy", "renewable", "electricity", "mwh", "megawatt", "solar", "wind", "power consumption"],
    "water": ["water", "withdrawal", "consumption", "discharge", "megalitre", "kilolitre", "freshwater"],
    "waste": ["waste", "recycling", "landfill", "hazardous", "non-hazardous", "diversion", "recovery"],
    "pollution": ["nox", "sox", "pm", "voc", "nitrogen oxide", "pollution", "particulate", "air emission"],
    "biodiversity": ["biodiversity", "habitat", "species", "deforestation", "land use", "ecosystem"],
    "circularity": ["circular", "recycled content", "material recovery", "reuse", "packaging"],
    "compliance": ["compliance", "fine", "penalty", "violation", "non-compliance", "iso 14001", "environmental management", "audit"],
}


def _classify_env_subcategory(metric_name: str) -> str:
    """Classify an environmental metric into a subcategory."""
    name_lower = metric_name.lower()
    for subcat, keywords in ENV_SUBCATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return subcat
    return "other"


def compute_errs(
    predicted_metrics: List[Dict[str, Any]],
    raw_text: str,
    schema_valid: bool,
    jaccard_threshold: float = 0.8,
) -> Dict[str, float]:
    """
    Compute Environmental Reporting Readiness Score (ERRS).

    ERRS = 0.20(value extracted)
         + 0.15(unit extracted)
         + 0.15(year extracted)
         + 0.25(source evidence linked)
         + 0.10(source file identified)
         + 0.10(schema validity)
         + 0.05(confidence score present)

    Returns overall ERRS and per-subcategory ERRS.
    """
    # Filter to environmental metrics only
    env_metrics = [m for m in predicted_metrics
                   if m.get("category", "E") == "E"
                   or _classify_env_subcategory(m.get("name", "")) != "other"]

    if not env_metrics:
        result = {"environmental_ERRS": 0.0}
        for subcat in ENV_SUBCATEGORY_KEYWORDS:
            result[f"{subcat}_ERRS"] = 0.0
        return result

    raw_lower = raw_text.lower()

    def _score_metric(m: Dict[str, Any]) -> float:
        score = 0.0
        # Value extracted (0.20)
        if m.get("value", "").strip():
            score += 0.20
        # Unit extracted (0.15)
        if m.get("unit", "").strip():
            score += 0.15
        # Year extracted (0.15)
        if m.get("year", "").strip():
            score += 0.15
        # Source evidence linked (0.25)
        source = m.get("source_text", "").strip()
        if source:
            if source.lower() in raw_lower:
                score += 0.25
            else:
                # Jaccard check
                from benchmarks.src.utils import jaccard_similarity
                raw_sents = re.split(r'[.!?\n]+', raw_text)
                for rs in raw_sents:
                    if jaccard_similarity(source, rs) >= jaccard_threshold:
                        score += 0.25
                        break
                else:
                    score += 0.10  # Partial credit for having source_text
        # Source file identified (0.10)
        if m.get("source_file", "").strip() or m.get("source_document", "").strip():
            score += 0.10
        # Schema validity (0.10)
        if schema_valid:
            score += 0.10
        # Confidence score present (0.05)
        if m.get("confidence_score") is not None or m.get("confidence") is not None:
            score += 0.05
        return min(score, 1.0)

    # Overall ERRS
    all_scores = [_score_metric(m) for m in env_metrics]
    overall_errs = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    result = {"environmental_ERRS": overall_errs}

    # Per-subcategory ERRS
    subcat_scores: Dict[str, List[float]] = {k: [] for k in ENV_SUBCATEGORY_KEYWORDS}
    for m in env_metrics:
        subcat = _classify_env_subcategory(m.get("name", ""))
        if subcat in subcat_scores:
            subcat_scores[subcat].append(_score_metric(m))

    for subcat, scores in subcat_scores.items():
        result[f"{subcat}_ERRS"] = round(sum(scores) / len(scores), 4) if scores else 0.0

    return result


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_single_run(
    output_json: Dict[str, Any] | None,
    raw_text: str,
    gt_metrics: List[Dict[str, Any]],
    json_parse_success: bool,
    evidence_spans: List[Dict[str, Any]] | None = None,
    unit_equivalents: List[List[str]] | None = None,
    k_values: List[int] = [10, 30, 60],
    jaccard_threshold: float = 0.9,
    pre_dedup_count: int = 0,
    post_dedup_count: int = 0,
) -> Dict[str, Any]:
    """
    Run all evaluation metrics for a single benchmark run.

    Returns a flat dict of all metric values.
    """
    result: Dict[str, Any] = {}

    # 1. Validity
    result["json_parse_success"] = 1 if json_parse_success else 0
    if output_json and json_parse_success:
        schema_ok, schema_err = validate_esg_schema(output_json)
        result["schema_valid"] = 1 if schema_ok else 0
    else:
        result["schema_valid"] = 0

    # If no valid output, set all metrics to 0
    if not output_json or not json_parse_success:
        _set_zero_metrics(result, k_values)
        return result

    # 2. Metric extraction quality
    section_scores = compute_per_section_scores(output_json, gt_metrics, unit_equivalents)
    result.update(section_scores)

    # Get all predicted metrics
    all_pred_metrics = []
    for section in ("environmental", "social", "governance"):
        all_pred_metrics.extend(output_json.get(section, {}).get("metrics", []))

    # Tag environmental metrics with category
    env_pred_metrics = output_json.get("environmental", {}).get("metrics", [])
    for m in env_pred_metrics:
        m["category"] = "E"

    # 3. Year/unit rates
    yu_rates = compute_year_unit_rates(all_pred_metrics, gt_metrics, unit_equivalents)
    result.update(yu_rates)
    result["unit_accuracy"] = round(1.0 - result.get("unit_missing_rate", 1.0), 4)
    result["year_accuracy"] = round(1.0 - (result.get("missing_year_rate", 1.0) + result.get("wrong_year_rate", 0.0)), 4)
    result["unit_year_accuracy"] = round((result["unit_accuracy"] + result["year_accuracy"]) / 2, 4)

    # 4. Groundedness
    grounded = compute_groundedness(all_pred_metrics, raw_text, jaccard_threshold)
    result.update(grounded)

    narr_grounded = compute_narrative_groundedness(output_json, raw_text)
    result["narrative_grounded_rate"] = narr_grounded

    # 5. Evidence quality
    if evidence_spans is None:
        evidence_spans = []
        for section in ("environmental", "social", "governance"):
            evidence_spans.extend(output_json.get(section, {}).get("top_evidence", []))

    ev_quality = compute_evidence_quality(evidence_spans, gt_metrics, raw_text, k_values)
    result.update(ev_quality)

    # 6. Compression/dedup stats
    comp_stats = compute_compression_stats(evidence_spans, raw_text, pre_dedup_count, post_dedup_count)
    result.update(comp_stats)

    # 7. ERRS (Environmental Reporting Readiness Score)
    schema_valid = result.get("schema_valid", 0) == 1
    errs = compute_errs(all_pred_metrics, raw_text, schema_valid, jaccard_threshold=0.8)
    result.update(errs)

    # 8. Environmental-specific precision/recall/F1
    env_gt = [m for m in gt_metrics if m.get("category") == "E"]
    env_pred = output_json.get("environmental", {}).get("metrics", [])
    if env_gt:
        env_strict = compute_metric_scores(env_pred, env_gt, unit_equivalents, mode="strict")
        env_relaxed = compute_metric_scores(env_pred, env_gt, unit_equivalents, mode="relaxed")
        result["environmental_precision"] = env_relaxed["precision"]
        result["environmental_recall"] = env_relaxed["recall"]
        result["environmental_f1"] = env_relaxed["f1"]
        result["environmental_miss_rate"] = round(1.0 - env_relaxed["recall"], 4)
    else:
        result["environmental_precision"] = 0.0
        result["environmental_recall"] = 0.0
        result["environmental_f1"] = 0.0
        result["environmental_miss_rate"] = 1.0

    result["social_f1"] = result.get("soc_relaxed_f1", 0.0)
    result["governance_f1"] = result.get("gov_relaxed_f1", 0.0)
    result["evidence_alignment"] = result.get("evidence_hit_rate", 0.0)  # alias for alignment

    return result


def _set_zero_metrics(result: Dict[str, Any], k_values: List[int]) -> None:
    """Set all evaluation metrics to 0 when output is invalid."""
    for prefix in ("env", "soc", "gov", "overall"):
        for mode in ("strict", "relaxed"):
            for metric in ("precision", "recall", "f1"):
                result[f"{prefix}_{mode}_{metric}"] = 0.0

    result["missing_year_rate"] = 1.0
    result["wrong_year_rate"] = 0.0
    result["unit_missing_rate"] = 1.0
    result["grounded_metric_rate"] = 0.0
    result["unsupported_metric_rate"] = 1.0
    result["narrative_grounded_rate"] = 0.0
    result["evidence_hit_rate"] = 0.0
    for k in k_values:
        result[f"recall_at_{k}"] = 0.0
    result["evidence_char_count"] = 0
    result["extracted_char_count"] = 0
    result["compression_ratio"] = 0.0
    result["dedup_rate"] = 0.0
    # Environmental metrics
    result["environmental_ERRS"] = 0.0
    result["environmental_precision"] = 0.0
    result["environmental_recall"] = 0.0
    result["environmental_f1"] = 0.0
    result["environmental_miss_rate"] = 1.0
    for subcat in ENV_SUBCATEGORY_KEYWORDS:
        result[f"{subcat}_ERRS"] = 0.0

    result["social_f1"] = 0.0
    result["governance_f1"] = 0.0
    result["evidence_alignment"] = 0.0
    result["unit_accuracy"] = 0.0
    result["year_accuracy"] = 0.0
    result["unit_year_accuracy"] = 0.0

