"""
AxiomESG Benchmark Runner
==========================

Orchestrates the full experiment matrix:
1. Loads synthetic dataset & ground truth
2. Loads variant configurations
3. Runs each (doc, variant) combination through the pipeline
4. Evaluates outputs against ground truth
5. Writes results to CSV
6. Triggers resolution plan if metrics are weak

Supports two modes:
- Import mode (default): directly calls pipeline functions
- Mock-LLM mode: when no LLM API keys are configured, uses a deterministic
  mock LLM that produces schema-compliant output from extracted evidence

Usage:
    python -m benchmarks.src.run_benchmarks --config benchmarks/config/benchmark.yaml
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Ensure backend is importable
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_backend_root = os.path.join(_repo_root, "backend")
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from benchmarks.src.eval import evaluate_single_run, validate_esg_schema
from benchmarks.src.real_awfa import apply_real_awfa, get_dedup_stats
from benchmarks.src.utils import (
    CSV_COLUMNS,
    get_benchmark_logger,
    get_git_commit,
    make_run_id,
    text_hash,
    utc_timestamp,
)
from benchmarks.src.variants import (
    VariantConfig,
    compute_run_plan,
    get_all_variants,
)

logger = get_benchmark_logger("runner")


# ---------------------------------------------------------------------------
# Mock LLM for offline / no-API-key mode
# ---------------------------------------------------------------------------

class MockLLMResult:
    def __init__(self, text: str, model_name: str = "mock-llm"):
        self.text = text
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.model_name = model_name


def _build_mock_esg_json(
    evidence: List[Dict[str, Any]],
    filenames: List[str],
    algorithm: str,
) -> Dict[str, Any]:
    """
    Build a deterministic ESG JSON from evidence spans.
    This is used when no LLM API is available.
    Extracts metrics directly from evidence text using regex patterns.
    """
    import re
    from datetime import datetime, timezone

    sections = {
        "environmental": {"cat": "E", "metrics": [], "evidence": [], "narrative_parts": []},
        "social": {"cat": "S", "metrics": [], "evidence": [], "narrative_parts": []},
        "governance": {"cat": "G", "metrics": [], "evidence": [], "narrative_parts": []},
    }

    cat_to_section = {"E": "environmental", "S": "social", "G": "governance"}

    for ev in evidence:
        cat = ev.get("category", "E")
        section_name = cat_to_section.get(cat, "environmental")
        section = sections[section_name]

        section["evidence"].append({
            "text": ev.get("text", ""),
            "weight": ev.get("weight", 0.0),
            "category": cat,
            "source_file": ev.get("source_file", "unknown"),
        })
        section["narrative_parts"].append(ev.get("text", ""))

        # Try to extract metrics from evidence text
        text = ev.get("text", "")
        # Pattern: look for numbers with units
        number_pattern = r'(\d[\d,]*\.?\d*)\s*(%|tCO2e|tonnes?\s*CO2e?|MWh|megawatt\s*hours?|megalitres?|metric\s*tons?|hours?|cases?|incidents?|meetings?|per\s*[\d,]+\s*hours?\s*worked|per\s*million\s*hours?\s*worked|USD)'
        matches = re.findall(number_pattern, text, re.IGNORECASE)

        # Year pattern
        year_match = re.search(r'\b(20[12]\d)\b', text)
        year = year_match.group(1) if year_match else ""

        if matches:
            value, unit = matches[0]
            # Derive metric name from surrounding text
            name = _extract_metric_name(text)
            section["metrics"].append({
                "name": name,
                "value": value.strip(),
                "unit": unit.strip(),
                "year": year,
                "source_text": text[:200],
            })

    output = {
        "metadata": {
            "source_files": filenames,
            "extraction_date": datetime.now(timezone.utc).isoformat(),
            "model_provider": "mock",
            "model_name": "mock-deterministic",
            "awfa_weights_preserved": True,
            "algorithm_used": algorithm,
        },
        "aggregation": {
            "total_documents": len(filenames),
            "total_esg_sentences": len(evidence),
            "total_weighted_blocks": len(evidence),
            "ocr_used": False,
        },
    }

    for section_name, data in sections.items():
        narrative = " ".join(data["narrative_parts"][:3]) if data["narrative_parts"] else "Not found in provided documents."
        conf = min(0.3 + len(data["metrics"]) * 0.1, 1.0)
        output[section_name] = {
            "narrative": narrative[:500],
            "metrics": data["metrics"],
            "confidence_score": round(conf, 2),
            "top_evidence": data["evidence"][:10],
        }

    return output


def _extract_metric_name(text: str) -> str:
    """Heuristically extract a metric name from evidence text."""
    import re
    # Try to find common ESG metric names in the text
    patterns = [
        (r"Scope\s*[123]\s*(?:GHG\s*)?[Ee]missions?", "Scope GHG Emissions"),
        (r"[Tt]otal\s+[Ee]nergy\s+[Cc]onsumption", "Total Energy Consumption"),
        (r"[Rr]enewable\s+[Ee]nergy", "Renewable Energy Share"),
        (r"[Ww]ater\s+[Ww]ithdrawal", "Total Water Withdrawal"),
        (r"[Ww]aste\s+[Gg]enerat", "Total Waste Generated"),
        (r"[Rr]ecycling\s+[Rr]ate", "Waste Recycling Rate"),
        (r"[Cc]arbon\s+[Ii]ntensity", "Carbon Intensity"),
        (r"NOx|[Nn]itrogen\s*[Oo]xide", "NOx Emissions"),
        (r"TRIR|[Tt]otal\s+[Rr]ecordable", "Total Recordable Incident Rate"),
        (r"LTIFR|[Ll]ost\s+[Tt]ime", "Lost Time Injury Frequency Rate"),
        (r"[Gg]ender\s+[Dd]iversity|[Ff]emale\s+represent", "Gender Diversity Ratio (Female)"),
        (r"[Bb]oard\s+[Gg]ender|[Ww]omen.*[Bb]oard", "Board Gender Diversity (Female)"),
        (r"[Tt]raining\s+[Hh]ours", "Average Training Hours per Employee"),
        (r"[Tt]urnover\s+[Rr]ate", "Employee Turnover Rate"),
        (r"[Cc]ommunity\s+[Ii]nvest", "Community Investment"),
        (r"[Ee]ngagement\s+[Ss]core", "Employee Engagement Score"),
        (r"[Bb]oard\s+[Ii]ndependenc", "Board Independence"),
        (r"[Aa]nti-[Cc]orruption", "Anti-Corruption Training Completion"),
        (r"[Bb]oard\s+[Mm]eeting\s+[Aa]ttendanc", "Board Meeting Attendance Rate"),
        (r"[Ww]histleblower", "Whistleblower Reports Received"),
        (r"[Dd]ata\s+[Pp]rivacy", "Data Privacy Incidents"),
        (r"[Aa]udit\s+[Cc]ommittee", "Audit Committee Meetings"),
        (r"ESG.*[Cc]ompensation", "ESG-Linked Executive Compensation"),
    ]
    for pattern, name in patterns:
        if re.search(pattern, text):
            return name
    # Fallback: first few words
    words = text.split()[:5]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Pipeline execution wrapper
# ---------------------------------------------------------------------------

def _check_llm_available() -> bool:
    """Check if any LLM provider API keys are configured."""
    return bool(
        os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )


def _run_single(
    file_data: bytes,
    file_name: str,
    content_type: str,
    variant: VariantConfig,
    seed: int,
    run_id: str,
    doc_type: str,
    use_real_llm: bool = False,
) -> Dict[str, Any]:
    """
    Execute a single pipeline run for one document with one variant.

    Returns dict with:
        output_json, raw_text, evidence, timings, usage, error
    """
    result = {
        "output_json": None,
        "raw_text": "",
        "evidence": [],
        "timings": {},
        "usage": {},
        "error": None,
        "json_parse_success": False,
        "ocr_called": False,
        "pre_dedup_count": 0,
        "post_dedup_count": 0,
    }

    try:
        # ------ EXTRACT ------
        t0 = time.perf_counter()
        from app.pipeline.extractor import extract_documents
        from app.core.config import Settings

        # Build settings with controlled OCR behavior
        env_overrides = {}
        if variant.ocr_mode == "force_off":
            env_overrides["AZURE_DOCINTEL_ENDPOINT"] = ""
            env_overrides["AZURE_DOCINTEL_KEY"] = ""
        # force_on: keep env as-is (if configured) — OCR triggered by short text

        # Create settings (loads from env)
        settings = Settings()

        # For force_off, override settings attributes
        if variant.ocr_mode == "force_off":
            settings.azure_docintel_endpoint = ""
            settings.azure_docintel_key = ""

        files = [(file_name, file_data, content_type)]

        try:
            extracted, ocr_used = extract_documents(files, settings)
        except Exception as e:
            # For scanned PDFs without OCR configured, we get an error
            # Use empty text instead
            if "OCR not configured" in str(e) or doc_type == "pdf_scanned":
                extracted = {file_name: ""}
                ocr_used = False
            else:
                raise

        t_extract = time.perf_counter() - t0
        raw_text = "\n\n".join(extracted.values()).strip()
        result["raw_text"] = raw_text
        result["ocr_called"] = ocr_used

        # ------ FILTER ------
        t1 = time.perf_counter()
        if variant.filter_on:
            from app.pipeline.esg_filter import filter_esg_sentences
            esg_filtered: Dict[str, List[str]] = {"E": [], "S": [], "G": []}
            for fname, text in extracted.items():
                filtered = filter_esg_sentences(text, settings)
                for key in ("E", "S", "G"):
                    esg_filtered[key].extend(filtered[key])
        else:
            # No filtering: split all text into sentences and assign to categories
            import re
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text) if s.strip()]
            esg_filtered = {"E": sentences[:], "S": sentences[:], "G": sentences[:]}
        t_filter = time.perf_counter() - t1

        # ------ WEIGHT ------
        t2 = time.perf_counter()
        if variant.weight_on:
            if variant.awfa_mode == "real":
                # Use Real AWFA
                weighted = apply_real_awfa(esg_filtered)
                stats = get_dedup_stats(esg_filtered)
                result["pre_dedup_count"] = stats["total_input"]
                result["post_dedup_count"] = stats["total_after_dedup"]
            elif variant.algorithm in ("bert_mean", "bert_static", "bert_awfa_v1", "bert_awfa_v2"):
                # BERT-based strategies
                from app.pipeline.strategies import get_strategy
                weight_fn = get_strategy(variant.algorithm)
                weighted = weight_fn(esg_filtered)
            else:
                # Heuristic AWFA
                from app.pipeline.awfa import apply_awfa
                weighted = apply_awfa(esg_filtered)
        else:
            # No weighting: pass through with uniform weights
            weighted = []
            for cat in ("E", "S", "G"):
                for sent in esg_filtered.get(cat, []):
                    weighted.append((cat, sent, 0.5))
        t_weight = time.perf_counter() - t2

        # Build evidence
        evidence = []
        for category, sentence, weight in weighted[:60]:
            source_file = file_name
            evidence.append({
                "text": sentence,
                "weight": weight,
                "category": category,
                "source_file": source_file,
            })
        result["evidence"] = evidence

        # ------ INTELLIGENCE (LLM) ------
        t3 = time.perf_counter()
        if use_real_llm:
            from app.pipeline.orchestrator import _prompt, _parse_json, _repair_prompt
            from app.pipeline.llm import get_llm_client

            prompt = _prompt(evidence)
            llm = get_llm_client(settings)
            llm_result = llm.generate(prompt, run_id)

            try:
                parsed = _parse_json(llm_result.text)
                result["json_parse_success"] = True
            except Exception:
                try:
                    repair = llm.generate(_repair_prompt(llm_result.text), run_id)
                    parsed = _parse_json(repair.text)
                    result["json_parse_success"] = True
                except Exception:
                    parsed = None
                    result["json_parse_success"] = False

            result["usage"] = llm_result.usage or {}
            llm_model = llm_result.model_name
        else:
            # Mock LLM mode
            parsed = _build_mock_esg_json(
                evidence,
                list(extracted.keys()),
                variant.algorithm,
            )
            result["json_parse_success"] = True
            result["usage"] = {"prompt_tokens": 0, "completion_tokens": 0}
            llm_model = "mock-deterministic"

        t_intelligence = time.perf_counter() - t3

        # ------ VALIDATE ------
        t4 = time.perf_counter()
        if parsed:
            # Inject metadata
            from datetime import datetime, timezone
            if "metadata" not in parsed:
                parsed["metadata"] = {}
            parsed["metadata"]["source_files"] = list(extracted.keys())
            parsed["metadata"]["extraction_date"] = datetime.now(timezone.utc).isoformat()
            parsed["metadata"]["model_provider"] = settings.llm_provider if use_real_llm else "mock"
            parsed["metadata"]["model_name"] = llm_model
            parsed["metadata"]["awfa_weights_preserved"] = True
            parsed["metadata"]["algorithm_used"] = variant.algorithm
            if "aggregation" not in parsed:
                parsed["aggregation"] = {}
            parsed["aggregation"]["total_documents"] = len(extracted)
            parsed["aggregation"]["total_esg_sentences"] = sum(len(v) for v in esg_filtered.values())
            parsed["aggregation"]["total_weighted_blocks"] = len(weighted)
            parsed["aggregation"]["ocr_used"] = ocr_used

            result["output_json"] = parsed
        t_validate = time.perf_counter() - t4

        result["timings"] = {
            "extract_ms": round(t_extract * 1000, 1),
            "filter_ms": round(t_filter * 1000, 1),
            "weight_ms": round(t_weight * 1000, 1),
            "ocr_ms": 0.0,  # OCR timing is embedded in extract
            "intelligence_ms": round(t_intelligence * 1000, 1),
            "validate_ms": round(t_validate * 1000, 1),
            "total_latency_ms": round((t_extract + t_filter + t_weight + t_intelligence + t_validate) * 1000, 1),
        }

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Run {run_id} failed: {result['error']}")
        if not result["timings"]:
            result["timings"] = {
                "extract_ms": 0, "filter_ms": 0, "weight_ms": 0,
                "ocr_ms": 0, "intelligence_ms": 0, "validate_ms": 0,
                "total_latency_ms": 0,
            }

    return result


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv_row(csv_path: str, row: Dict[str, Any], write_header: bool = False) -> None:
    """Append a single row to the CSV file."""
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    mode = "a"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header or not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main benchmark execution
# ---------------------------------------------------------------------------

def run_benchmarks(
    config_path: str = "benchmarks/config/benchmark.yaml",
    augmentation_round: int = 0,
    augmentation_tag: str = "",
) -> str:
    """
    Execute the full benchmark matrix.

    Returns path to the CSV file.
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_dir = config.get("dataset", {}).get("output_dir", "benchmarks/dataset")
    results_dir = config.get("output", {}).get("results_dir", "benchmarks/results")
    artifacts_dir = config.get("output", {}).get("artifacts_dir", "benchmarks/artifacts")
    csv_filename = config.get("output", {}).get("csv_file", "axiomesg_benchmark_runs.csv")
    csv_path = os.path.join(results_dir, csv_filename)
    min_runs = config.get("execution", {}).get("min_scored_runs", 500)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Load variants
    all_variants = get_all_variants(config_path)
    available_variants = [v for v in all_variants if v.available]
    unavailable_variants = [v for v in all_variants if not v.available]

    # Load dataset manifest
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Dataset manifest not found at {manifest_path}. Run generate_synthetic_dataset first.")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    docs = manifest.get("documents", [])
    if augmentation_tag:
        # Filter to only augmentation docs
        docs = [d for d in docs if d["doc_id"].startswith(augmentation_tag)]

    # Compute run plan
    plan = compute_run_plan(len(docs), all_variants, min_runs)

    # Determine LLM availability
    use_real_llm = _check_llm_available()
    if use_real_llm:
        logger.info("LLM API keys detected. Using real LLM for intelligence stage.")
    else:
        logger.info("No LLM API keys found. Using mock LLM (deterministic extraction).")

    # Load evaluation config
    eval_config = config.get("evaluation", {})
    unit_equivalents = eval_config.get("relaxed_match", {}).get("unit_equivalents", [])
    k_values = eval_config.get("evidence_k_values", [10, 30, 60])
    jaccard_threshold = eval_config.get("jaccard_grounding_threshold", 0.9)

    git_commit = get_git_commit()
    base_seed = config.get("dataset", {}).get("seed", 42)
    replicas = plan["total_replicas"]

    total_runs = 0
    scored_runs = 0
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    logger.info(f"Starting benchmark: {len(docs)} docs × {len(available_variants)} variants × {replicas} replicas")
    logger.info(f"Total planned runs: {plan['total_runs']}")

    # Write rows for unavailable variants
    for uv in unavailable_variants:
        for doc in docs:
            row = {col: "" for col in CSV_COLUMNS}
            row["run_id"] = make_run_id()
            row["git_commit"] = git_commit
            row["timestamp"] = utc_timestamp()
            row["seed"] = base_seed
            row["variant_id"] = uv.variant_id
            row["variant_label"] = uv.label
            row["algorithm_used"] = uv.algorithm
            row["filter_on"] = uv.filter_on
            row["weight_on"] = uv.weight_on
            row["ocr_mode"] = uv.ocr_mode
            row["awfa_mode"] = uv.awfa_mode
            row["bert_mode"] = uv.bert_mode
            row["doc_id"] = doc["doc_id"]
            row["doc_path"] = doc["file_path"]
            row["doc_type"] = doc["doc_type"]
            row["augmentation_round"] = augmentation_round
            row["variant_skipped_reason"] = uv.skip_reason
            _write_csv_row(csv_path, row, write_header)
            write_header = False
            total_runs += 1

    # Execute available variants
    for replica in range(replicas):
        for vi, variant in enumerate(available_variants):
            for di, doc in enumerate(docs):
                run_id = make_run_id()
                seed = base_seed + replica * 1000 + di

                logger.info(
                    f"[{total_runs + 1}/{plan['total_runs']}] "
                    f"Variant={variant.variant_id} Doc={doc['doc_id']} Replica={replica}"
                )

                # Load document file
                doc_path = os.path.join(
                    os.path.dirname(os.path.dirname(dataset_dir)),
                    doc["file_path"],
                )
                # Try alternative path resolution
                if not os.path.exists(doc_path):
                    doc_path = os.path.join(dataset_dir, "synthetic_docs",
                                            os.path.basename(doc["file_path"]))
                if not os.path.exists(doc_path):
                    # Search for the file
                    for root, dirs, files in os.walk(os.path.join(dataset_dir, "synthetic_docs")):
                        for fname in files:
                            if doc["doc_id"] in fname:
                                doc_path = os.path.join(root, fname)
                                break

                if not os.path.exists(doc_path):
                    logger.warning(f"Document not found: {doc_path}, skipping")
                    continue

                with open(doc_path, "rb") as f:
                    file_data = f.read()

                # Determine content type
                ext = os.path.splitext(doc_path)[1].lower()
                content_type_map = {
                    ".pdf": "application/pdf",
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ".csv": "text/csv",
                    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                }
                content_type = content_type_map.get(ext, "application/octet-stream")

                # Load ground truth
                gt_path = os.path.join(dataset_dir, "ground_truth", f"{doc['doc_id']}.json")
                gt_metrics = []
                is_scanned = False
                if os.path.exists(gt_path):
                    with open(gt_path, "r") as f:
                        gt = json.load(f)
                    gt_metrics = gt.get("metrics", [])
                    is_scanned = gt.get("is_scanned", False)

                # Execute pipeline
                run_result = _run_single(
                    file_data=file_data,
                    file_name=os.path.basename(doc_path),
                    content_type=content_type,
                    variant=variant,
                    seed=seed,
                    run_id=run_id,
                    doc_type=doc["doc_type"],
                    use_real_llm=use_real_llm,
                )

                # Save output artifact
                artifact_path = ""
                if run_result["output_json"]:
                    artifact_dir = os.path.join(artifacts_dir, run_id[:8])
                    os.makedirs(artifact_dir, exist_ok=True)
                    artifact_path = os.path.join(artifact_dir, "output.json")
                    with open(artifact_path, "w") as f:
                        json.dump(run_result["output_json"], f, indent=2)

                # Evaluate
                eval_result = evaluate_single_run(
                    output_json=run_result["output_json"],
                    raw_text=run_result["raw_text"],
                    gt_metrics=gt_metrics,
                    json_parse_success=run_result["json_parse_success"],
                    evidence_spans=run_result["evidence"],
                    unit_equivalents=unit_equivalents,
                    k_values=k_values,
                    jaccard_threshold=jaccard_threshold,
                    pre_dedup_count=run_result["pre_dedup_count"],
                    post_dedup_count=run_result["post_dedup_count"],
                )

                # Build CSV row
                timings = run_result["timings"]
                usage = run_result["usage"]
                prompt_chars = usage.get("prompt_tokens", 0) * 4  # estimate
                output_chars = usage.get("completion_tokens", 0) * 4

                row = {
                    "run_id": run_id,
                    "git_commit": git_commit,
                    "timestamp": utc_timestamp(),
                    "seed": seed,
                    "variant_id": variant.variant_id,
                    "variant_label": variant.label,
                    "algorithm_used": variant.algorithm,
                    "filter_on": variant.filter_on,
                    "weight_on": variant.weight_on,
                    "ocr_mode": variant.ocr_mode,
                    "awfa_mode": variant.awfa_mode,
                    "bert_mode": variant.bert_mode,
                    "llm_provider": "mock" if not use_real_llm else os.environ.get("LLM_PROVIDER", "unknown"),
                    "llm_model_name": "mock-deterministic" if not use_real_llm else "",
                    "doc_id": doc["doc_id"],
                    "doc_path": doc.get("file_path", ""),
                    "doc_type": doc["doc_type"],
                    "is_scanned": is_scanned,
                    "augmentation_round": augmentation_round,
                    # Timings
                    "extract_ms": timings.get("extract_ms", 0),
                    "filter_ms": timings.get("filter_ms", 0),
                    "weight_ms": timings.get("weight_ms", 0),
                    "ocr_ms": timings.get("ocr_ms", 0),
                    "intelligence_ms": timings.get("intelligence_ms", 0),
                    "validate_ms": timings.get("validate_ms", 0),
                    "total_latency_ms": timings.get("total_latency_ms", 0),
                    # Output info
                    "output_json_path": artifact_path,
                    "raw_text_hash": text_hash(run_result["raw_text"]),
                    "raw_text_preview": run_result["raw_text"][:200].replace("\n", " "),
                    "extracted_char_count": eval_result.get("extracted_char_count", len(run_result["raw_text"])),
                    "evidence_char_count": eval_result.get("evidence_char_count", 0),
                    # LLM usage
                    "llm_prompt_chars": prompt_chars,
                    "llm_output_chars": output_chars,
                    "cost_proxy": prompt_chars + output_chars,
                    "ocr_called": 1 if run_result["ocr_called"] else 0,
                    # Variant skip info
                    "variant_skipped_reason": "",
                    "error_message": run_result["error"] or "",
                }

                # Merge eval metrics
                for key in eval_result:
                    if key in CSV_COLUMNS:
                        row[key] = eval_result[key]

                _write_csv_row(csv_path, row, write_header)
                write_header = False
                total_runs += 1
                if not run_result["error"]:
                    scored_runs += 1

    logger.info(f"Benchmark complete. Total runs: {total_runs}, Scored runs: {scored_runs}")
    logger.info(f"Results saved to: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Resolution plan: diagnose and augment
# ---------------------------------------------------------------------------

def diagnose_and_augment(
    csv_path: str,
    config_path: str = "benchmarks/config/benchmark.yaml",
    max_rounds: int = 3,
) -> None:
    """
    Check if metrics are below thresholds and automatically generate
    targeted augmentation documents + rerun selected variants.
    """
    import pandas as pd

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    resolution = config.get("resolution", {})
    strict_f1_min = resolution.get("strict_f1_min", 0.55)
    grounded_min = resolution.get("grounded_metric_rate_min", 0.85)
    schema_valid_min = resolution.get("schema_valid_min", 0.95)
    augment_count = resolution.get("augment_doc_count", 20)
    dataset_dir = config.get("dataset", {}).get("output_dir", "benchmarks/dataset")

    df = pd.read_csv(csv_path)
    # Filter to scored runs only (no skipped variants)
    scored = df[df["variant_skipped_reason"].isna() | (df["variant_skipped_reason"] == "")]
    scored = scored[scored["error_message"].isna() | (scored["error_message"] == "")]

    if scored.empty:
        logger.warning("No scored runs found. Cannot diagnose.")
        return

    # Check thresholds
    avg_strict_f1 = scored["overall_strict_f1"].mean()
    avg_grounded = scored["grounded_metric_rate"].mean()
    avg_schema = scored["schema_valid"].mean()

    logger.info(f"Current metrics — strict_f1: {avg_strict_f1:.3f}, "
                f"grounded: {avg_grounded:.3f}, schema_valid: {avg_schema:.3f}")

    needs_augmentation = False
    stress_focus = ""
    diagnosis = ""

    if avg_strict_f1 < strict_f1_min:
        needs_augmentation = True
        # Diagnose: which stage is failing?
        avg_evidence_hit = scored["evidence_hit_rate"].mean()
        avg_recall_10 = scored["recall_at_10"].mean()
        avg_json_parse = scored["json_parse_success"].mean()

        if avg_json_parse < 0.9:
            stress_focus = "llm"
            diagnosis = f"LLM formatting issues (json_parse={avg_json_parse:.3f})"
        elif avg_evidence_hit < 0.5:
            stress_focus = "filter"
            diagnosis = f"Filtering misses (evidence_hit={avg_evidence_hit:.3f})"
        elif avg_recall_10 < 0.3:
            stress_focus = "weight"
            diagnosis = f"Weighting misses (recall@10={avg_recall_10:.3f})"
        else:
            stress_focus = "filter"
            diagnosis = f"General extraction weakness (strict_f1={avg_strict_f1:.3f})"

    if avg_grounded < grounded_min:
        needs_augmentation = True
        if not stress_focus:
            stress_focus = "llm"
            diagnosis = f"Low groundedness ({avg_grounded:.3f})"

    if avg_schema < schema_valid_min:
        needs_augmentation = True
        if not stress_focus:
            stress_focus = "llm"
            diagnosis = f"Schema validation failures ({avg_schema:.3f})"

    if not needs_augmentation:
        logger.info("All metrics above thresholds. No augmentation needed.")
        return

    # Check augmentation round
    current_round = int(scored["augmentation_round"].max()) if "augmentation_round" in scored.columns else 0
    next_round = current_round + 1

    if next_round > max_rounds:
        logger.warning(f"Max augmentation rounds ({max_rounds}) reached. Stopping.")
        return

    logger.info(f"Diagnosis: {diagnosis}")
    logger.info(f"Generating augmentation round {next_round} with stress_focus={stress_focus}")

    # Generate augmented dataset
    from benchmarks.src.generate_synthetic_dataset import generate_dataset
    aug_tag = f"aug{next_round}"
    aug_seed = 42 + next_round * 1000

    generate_dataset(
        output_dir=dataset_dir,
        n_docs=augment_count,
        seed=aug_seed,
        stress_focus=stress_focus,
        augmentation_tag=aug_tag,
    )

    logger.info(f"Generated {augment_count} augmentation docs with tag={aug_tag}")

    # Rerun selected variants on augmented docs
    run_benchmarks(
        config_path=config_path,
        augmentation_round=next_round,
        augmentation_tag=aug_tag,
    )

    logger.info(f"Augmentation round {next_round} complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run AxiomESG benchmarks")
    parser.add_argument("--config", default="benchmarks/config/benchmark.yaml",
                        help="Path to benchmark config")
    parser.add_argument("--augment", action="store_true",
                        help="Run resolution plan (diagnose + augment) after benchmarking")
    parser.add_argument("--augment-only", action="store_true",
                        help="Only run augmentation (skip main benchmark)")
    parser.add_argument("--report", action="store_true",
                        help="Generate report after benchmarking")

    args = parser.parse_args()

    if not args.augment_only:
        csv_path = run_benchmarks(config_path=args.config)
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        csv_path = os.path.join(
            config.get("output", {}).get("results_dir", "benchmarks/results"),
            config.get("output", {}).get("csv_file", "axiomesg_benchmark_runs.csv"),
        )

    if args.augment or args.augment_only:
        try:
            diagnose_and_augment(csv_path, args.config)
        except Exception as e:
            logger.error(f"Augmentation failed: {e}")

    if args.report:
        from benchmarks.src.report import generate_report
        generate_report(csv_path, args.config)


if __name__ == "__main__":
    main()
