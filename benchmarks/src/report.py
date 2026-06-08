"""
AxiomESG Benchmark Report Generator (No-Filter Version)
=========================================================
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

# Ensure imports work
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from benchmarks.src.utils import bootstrap_ci, format_ci, get_benchmark_logger

logger = get_benchmark_logger("report")

def _safe_mean(series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def generate_report(csv_path: str, config_path: str = "benchmarks/config/benchmark.yaml") -> str:
    results_dir = os.path.dirname(os.path.abspath(csv_path))
    report_path = os.path.join(results_dir, "benchmark_REPORT.md")

    df = pd.read_csv(csv_path)
    total_runs = len(df)

    scored = df[
        (df["variant_skipped_reason"].isna() | (df["variant_skipped_reason"] == ""))
        & (df["error_message"].isna() | (df["error_message"] == ""))
    ].copy()
    skipped = df[df["variant_skipped_reason"].notna() & (df["variant_skipped_reason"] != "")].copy()
    errored = df[df["error_message"].notna() & (df["error_message"] != "")].copy()

    lines: List[str] = []

    # ---- 1. Executive Summary ----
    lines.append("# AxiomESG Benchmark Report")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append(f"- **Total runs**: {total_runs}")
    lines.append(f"- **Scored runs**: {len(scored)}")
    lines.append(f"- **Skipped runs**: {len(skipped)}")
    lines.append(f"- **Errored runs**: {len(errored)}")
    if not scored.empty:
        best_variant = scored.groupby("variant_id")["overall_relaxed_f1"].mean().idxmax()
        best_f1 = scored.groupby("variant_id")["overall_relaxed_f1"].mean().max()
        lines.append(f"- **Best Variant**: `{best_variant}` (F1 = {best_f1:.3f})")
    lines.append("")

    # ---- 2. Algorithm Efficacy (F1) ----
    lines.append("## 2. Algorithm Efficacy (F1)")
    lines.append("| Variant | Strict P | Strict R | Strict F1 | Relaxed P | Relaxed R | Relaxed F1 |")
    lines.append("|---------|----------|----------|-----------|-----------|-----------|------------|")
    if not scored.empty:
        variant_groups = scored.groupby("variant_id")
        for vid, group in sorted(variant_groups, key=lambda x: x[0]):
            sp = _safe_mean(group["overall_strict_precision"])
            sr = _safe_mean(group["overall_strict_recall"])
            sf = _safe_mean(group["overall_strict_f1"])
            rp = _safe_mean(group["overall_relaxed_precision"])
            rr = _safe_mean(group["overall_relaxed_recall"])
            rf = _safe_mean(group["overall_relaxed_f1"])
            lines.append(f"| {vid} | {sp:.3f} | {sr:.3f} | {sf:.3f} | {rp:.3f} | {rr:.3f} | {rf:.3f} |")
    lines.append("")

    # ---- 3. Algorithm Evidence Quality (Alignment) ----
    lines.append("## 3. Algorithm Evidence Quality (Alignment)")
    lines.append("| Variant | Evidence Hit Rate | Grounded % | Unsupported % | Narrative Grounded % |")
    lines.append("|---------|-------------------|------------|---------------|----------------------|")
    if not scored.empty:
        for vid, group in sorted(variant_groups, key=lambda x: x[0]):
            ehr = _safe_mean(group["evidence_hit_rate"]) if "evidence_hit_rate" in group.columns else 0.0
            gmr = _safe_mean(group["grounded_metric_rate"]) if "grounded_metric_rate" in group.columns else 0.0
            umr = _safe_mean(group["unsupported_metric_rate"]) if "unsupported_metric_rate" in group.columns else 0.0
            ngr = _safe_mean(group["narrative_grounded_rate"]) if "narrative_grounded_rate" in group.columns else 0.0
            lines.append(f"| {vid} | {ehr:.3f} | {gmr:.3f} | {umr:.3f} | {ngr:.3f} |")
    lines.append("")

    # ---- 4. OCR Ablation ----
    lines.append("## 4. OCR Ablation")
    scanned = scored[scored["is_scanned"] == True] if not scored.empty and "is_scanned" in scored.columns else pd.DataFrame()
    if not scanned.empty:
        lines.append("| Variant | OCR Mode | Scanned Doc F1 |")
        lines.append("|---------|----------|----------------|")
        for (vid, ocr), group in scanned.groupby(["variant_id", "ocr_mode"]):
            f1 = _safe_mean(group["overall_relaxed_f1"])
            lines.append(f"| {vid} | {ocr} | {f1:.3f} |")
    else:
        lines.append("> No scanned documents or OCR ablation data available.")
    lines.append("")

    # ---- 5. Latency & Cost ----
    lines.append("## 5. Latency & Cost")
    lines.append("| Variant | Extract (ms) | Filter (ms) | Weight (ms) | Validate (ms) | Total (ms) |")
    lines.append("|---------|--------------|-------------|-------------|---------------|------------|")
    if not scored.empty:
        for vid, group in sorted(variant_groups, key=lambda x: x[0]):
            ex = _safe_mean(group["extract_ms"])
            fi = _safe_mean(group["filter_ms"]) if "filter_ms" in group.columns else _safe_mean(group["classify_ms"]) if "classify_ms" in group.columns else 0.0
            we = _safe_mean(group["weight_ms"])
            va = _safe_mean(group["validate_ms"])
            to = _safe_mean(group["total_latency_ms"])
            lines.append(f"| {vid} | {ex:.0f} | {fi:.0f} | {we:.0f} | {va:.0f} | {to:.0f} |")
    lines.append("")

    # ---- 6. Component Failure Analysis ----
    lines.append("## 6. Component Failure Analysis")
    if not errored.empty:
        lines.append("| Variant | Error Message | Count |")
        lines.append("|---------|---------------|-------|")
        err_counts = errored.groupby(["variant_id", "error_message"]).size().reset_index(name="count")
        for _, row in err_counts.iterrows():
            lines.append(f"| {row['variant_id']} | {row['error_message']} | {row['count']} |")
    else:
        lines.append("> No components failed during this benchmark run.")
    lines.append("")

    # ---- 7. Component Resolution Deltas (if applicable) ----
    lines.append("## 7. Component Resolution Deltas (if applicable)")
    if not scored.empty and "augmentation_round" in scored.columns:
        aug_rounds = sorted(scored["augmentation_round"].dropna().unique())
        if len(aug_rounds) > 1:
            base_f1 = _safe_mean(scored[scored["augmentation_round"] == 0]["overall_relaxed_f1"])
            for r in aug_rounds[1:]:
                aug_f1 = _safe_mean(scored[scored["augmentation_round"] == r]["overall_relaxed_f1"])
                lines.append(f"- **Round {int(r)} vs Base**: {aug_f1 - base_f1:+.3f} F1")
        else:
            lines.append("> No multi-round augmentations present.")
    else:
        lines.append("> Not applicable.")
    lines.append("")

    # ---- 8. BERT AWFA v1 vs v2 ----
    lines.append("## 8. BERT AWFA v1 vs v2")
    if not scored.empty:
        v1_runs = scored[scored["variant_id"].str.contains("AWFA_V1")]
        v2_runs = scored[scored["variant_id"].str.contains("AWFA_V2")]
        if not v1_runs.empty and not v2_runs.empty:
            v1_f1 = _safe_mean(v1_runs["overall_relaxed_f1"])
            v2_f1 = _safe_mean(v2_runs["overall_relaxed_f1"])
            lines.append(f"- **AWFA v1 mean F1**: {v1_f1:.3f}")
            lines.append(f"- **AWFA v2 mean F1**: {v2_f1:.3f}")
            lines.append(f"- **Delta**: {v2_f1 - v1_f1:+.3f}")
        else:
            lines.append("> AWFA v1 and/or v2 not tested.")
    lines.append("")

    # ---- 9. ERRS Deep Dive ----
    lines.append("## 9. ERRS Deep Dive")
    lines.append("| Variant | Overall ERRS | Emissions | Energy | Water | Waste | Compliance |")
    lines.append("|---------|--------------|-----------|--------|-------|-------|------------|")
    if not scored.empty and "environmental_ERRS" in scored.columns:
        for vid, group in sorted(variant_groups, key=lambda x: x[0]):
            errs = _safe_mean(group["environmental_ERRS"])
            emi = _safe_mean(group["emissions_ERRS"]) if "emissions_ERRS" in group.columns else 0.0
            ene = _safe_mean(group["energy_ERRS"]) if "energy_ERRS" in group.columns else 0.0
            wat = _safe_mean(group["water_ERRS"]) if "water_ERRS" in group.columns else 0.0
            was = _safe_mean(group["waste_ERRS"]) if "waste_ERRS" in group.columns else 0.0
            com = _safe_mean(group["compliance_ERRS"]) if "compliance_ERRS" in group.columns else 0.0
            lines.append(f"| {vid} | {errs:.3f} | {emi:.3f} | {ene:.3f} | {wat:.3f} | {was:.3f} | {com:.3f} |")
    else:
        lines.append("> No ERRS data available.")
    lines.append("")

    # ---- 10. System Topology ----
    lines.append("## 10. System Topology")
    lines.append("- **Algorithm Filter Pre-stage**: Disabled" if scored.get("disable_pre_algorithm_filter", pd.Series([0])).mean() > 0 else "- **Algorithm Filter Pre-stage**: Enabled")
    lines.append("- **LLM Provider**: " + (scored["llm_provider"].iloc[0] if not scored.empty else "N/A"))
    lines.append("- **LLM Model**: " + (scored["llm_model_name"].iloc[0] if not scored.empty else "N/A"))
    lines.append("")

    # ---- 11. Test Data Characteristics ----
    lines.append("## 11. Test Data Characteristics")
    if not scored.empty:
        syn = len(scored[scored["is_synthetic"] == 1])
        real = len(scored[scored["is_real"] == 1])
        lines.append(f"- **Synthetic Runs**: {syn}")
        lines.append(f"- **Real Runs**: {real}")
        lines.append("| Doc Type | Count |")
        lines.append("|----------|-------|")
        dt_counts = scored["doc_type"].value_counts()
        for dt, count in dt_counts.items():
            lines.append(f"| {dt} | {count} |")
    lines.append("")

    # ---- 12. Raw Data Dump ----
    lines.append("## 12. Raw Data Dump")
    lines.append(f"Please see the CSV artifact: `{csv_path}`")
    lines.append("")

    # ---- Generate Figures ----
    from benchmarks.src.figures import generate_figures
    figures_dir = os.path.join(results_dir, "figures")
    saved_figs = generate_figures(csv_path, figures_dir)
    
    if saved_figs:
        lines.append("## Visualizations")
        lines.append("")
        for fig_path in saved_figs:
            rel_path = os.path.relpath(fig_path, results_dir)
            lines.append(f"![{os.path.basename(fig_path)}]({rel_path})")
            lines.append("")

    _write_report(report_path, lines)
    logger.info(f"Report written to: {report_path}")
    return report_path


def _write_report(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\n".join(lines))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--csv", default="benchmarks/results/no_filter_benchmark/axiomesg_no_filter_benchmark_runs.csv")
    parser.add_argument("--config", default="benchmarks/config/benchmark.yaml")
    args = parser.parse_args()

    generate_report(args.csv, args.config)


if __name__ == "__main__":
    main()
