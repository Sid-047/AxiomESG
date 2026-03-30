"""
Variant Configuration & Availability Detection
=================================================

Defines all benchmark experiment variants, checks BERT weight availability,
and yields executable pipeline configurations for the benchmark runner.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from benchmarks.src.utils import get_benchmark_logger

logger = get_benchmark_logger("variants")


@dataclass
class VariantConfig:
    """Configuration for a single benchmark variant."""
    variant_id: str
    label: str
    filter_on: bool
    weight_on: bool
    ocr_mode: str          # auto, force_on, force_off
    awfa_mode: str          # none, heuristic, real
    bert_mode: str          # none, mean, static, awfa_v1, awfa_v2
    algorithm: str          # heuristic, real_awfa, bert_mean, bert_static, etc.
    requires_bert: bool = False
    available: bool = True
    skip_reason: str = ""

    @property
    def algorithm_for_pipeline(self) -> str:
        """Map variant algorithm names to pipeline strategy keys."""
        mapping = {
            "passthrough": "heuristic",   # Will be bypassed in runner
            "heuristic": "heuristic",
            "real_awfa": "real_awfa",     # Custom implementation
            "bert_mean": "bert_mean",
            "bert_static": "bert_static",
            "bert_awfa_v1": "bert_awfa_v1",
            "bert_awfa_v2": "bert_awfa_v2",
        }
        return mapping.get(self.algorithm, self.algorithm)


def check_bert_availability() -> bool:
    """
    Check if BERT ESG classifier weights are available on disk.
    Returns True if model directory exists and contains weight files.
    """
    backend_root = Path(__file__).resolve().parents[2] / "backend"
    bert_dirs = [
        backend_root / "app" / "bert_esg_classifier" / "content" / "bert_esg_classifier_v2",
        backend_root / "app" / "bert_esg_classifier" / "content" / "bert_esg_classifier",
    ]

    for d in bert_dirs:
        if d.exists():
            # Check for actual weight files
            has_weights = any(
                f.suffix in (".bin", ".safetensors", ".pt", ".pth")
                for f in d.iterdir() if f.is_file()
            )
            has_config = (d / "config.json").exists()
            if has_weights or has_config:
                return True
    return False


def load_variants(config_path: str | None = None) -> List[VariantConfig]:
    """
    Load variant definitions from benchmark.yaml and check availability.

    Returns a list of VariantConfig, with unavailable variants marked.
    """
    if config_path is None:
        config_path = str(
            Path(__file__).resolve().parents[1] / "config" / "benchmark.yaml"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    bert_available = check_bert_availability()
    if not bert_available:
        logger.warning("BERT ESG model weights NOT found. BERT variants will be marked UNAVAILABLE.")
    else:
        logger.info("BERT ESG model weights detected. All variants available.")

    variants_raw = config.get("variants", {})
    variants: List[VariantConfig] = []

    for vid, vdef in variants_raw.items():
        requires_bert = vdef.get("requires_bert", False)
        available = True
        skip_reason = ""

        if requires_bert and not bert_available:
            available = False
            skip_reason = "missing_bert_weights"

        vc = VariantConfig(
            variant_id=vid,
            label=vdef.get("label", vid),
            filter_on=vdef.get("filter_on", True),
            weight_on=vdef.get("weight_on", True),
            ocr_mode=vdef.get("ocr_mode", "auto"),
            awfa_mode=vdef.get("awfa_mode", "none"),
            bert_mode=vdef.get("bert_mode", "none"),
            algorithm=vdef.get("algorithm", "heuristic"),
            requires_bert=requires_bert,
            available=available,
            skip_reason=skip_reason,
        )
        variants.append(vc)

    available_count = sum(1 for v in variants if v.available)
    total_count = len(variants)
    logger.info(
        f"Loaded {total_count} variants, {available_count} available, "
        f"{total_count - available_count} unavailable."
    )

    return variants


def get_available_variants(config_path: str | None = None) -> List[VariantConfig]:
    """Return only available variants."""
    return [v for v in load_variants(config_path) if v.available]


def get_all_variants(config_path: str | None = None) -> List[VariantConfig]:
    """Return all variants including unavailable ones."""
    return load_variants(config_path)


def compute_run_plan(
    n_docs: int,
    variants: List[VariantConfig],
    min_runs: int = 500,
    replicas: int = 1,
) -> Dict[str, Any]:
    """
    Compute the run plan to achieve at least min_runs scored runs.

    If not enough runs from available variants, increase doc count or replicas.
    """
    available = [v for v in variants if v.available]
    unavailable = [v for v in variants if not v.available]

    base_runs = n_docs * len(available) * replicas
    extra_docs = 0
    extra_replicas = 0

    if base_runs < min_runs and available:
        needed = min_runs - base_runs
        # First try adding replicas
        extra_replicas_needed = -(-needed // (n_docs * len(available)))  # ceil div
        if extra_replicas_needed <= 3:
            extra_replicas = extra_replicas_needed
        else:
            # Add more docs instead
            extra_docs_needed = -(-needed // len(available))
            extra_docs = extra_docs_needed

    total_runs = (n_docs + extra_docs) * len(available) * (replicas + extra_replicas)

    plan = {
        "n_docs": n_docs,
        "extra_docs": extra_docs,
        "total_docs": n_docs + extra_docs,
        "available_variants": len(available),
        "unavailable_variants": len(unavailable),
        "base_replicas": replicas,
        "extra_replicas": extra_replicas,
        "total_replicas": replicas + extra_replicas,
        "total_runs": total_runs,
        "meets_minimum": total_runs >= min_runs,
        "unavailable_variant_ids": [v.variant_id for v in unavailable],
        "unavailable_reasons": {v.variant_id: v.skip_reason for v in unavailable},
    }

    logger.info(
        f"Run plan: {total_runs} total runs "
        f"({n_docs + extra_docs} docs × {len(available)} variants × "
        f"{replicas + extra_replicas} replicas). "
        f"Meets minimum={plan['meets_minimum']}"
    )

    return plan
