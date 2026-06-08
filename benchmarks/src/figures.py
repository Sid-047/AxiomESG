"""
AxiomESG Benchmark Figure Generator
====================================

Generates all required plots from the benchmark CSV.
Uses matplotlib only — no seaborn or external plotting services.

Output to: benchmarks/results/figures/
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _setup_style():
    """Configure consistent plot style."""
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def generate_figures(csv_path: str, output_dir: str) -> List[str]:
    """Generate all benchmark figures from CSV. Returns list of saved paths."""
    import pandas as pd

    _setup_style()
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    scored = df[
        (df["variant_skipped_reason"].isna() | (df["variant_skipped_reason"] == ""))
        & (df["error_message"].isna() | (df["error_message"] == ""))
    ].copy()

    if scored.empty:
        return []

    saved = []
    variant_order = sorted(scored["variant_id"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(variant_order)))
    color_map = dict(zip(variant_order, colors))

    # 1. F1 by variant
    fig, ax = plt.subplots(figsize=(14, 6))
    means = scored.groupby("variant_id")["overall_relaxed_f1"].mean()
    means = means.reindex(variant_order)
    bars = ax.bar(range(len(means)), means.values, color=[color_map[v] for v in means.index])
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=45, ha="right")
    ax.set_ylabel("Relaxed F1")
    ax.set_title("Overall Relaxed F1 by Variant")
    ax.set_ylim(0, 1)
    for i, v in enumerate(means.values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "f1_by_variant.png")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    # 2. Environmental F1 by variant
    if "environmental_f1" in scored.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        means = scored.groupby("variant_id")["environmental_f1"].mean().reindex(variant_order)
        ax.bar(range(len(means)), means.values, color=[color_map[v] for v in means.index])
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=45, ha="right")
        ax.set_ylabel("Environmental F1")
        ax.set_title("Environmental F1 by Variant")
        ax.set_ylim(0, 1)
        for i, v in enumerate(means.values):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        path = os.path.join(output_dir, "environmental_f1_by_variant.png")
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    # 3. Miss rate by variant
    fig, ax = plt.subplots(figsize=(14, 6))
    miss_rate = 1.0 - scored.groupby("variant_id")["overall_relaxed_recall"].mean()
    miss_rate = miss_rate.reindex(variant_order)
    ax.bar(range(len(miss_rate)), miss_rate.values, color=[color_map[v] for v in miss_rate.index])
    ax.set_xticks(range(len(miss_rate)))
    ax.set_xticklabels(miss_rate.index, rotation=45, ha="right")
    ax.set_ylabel("Miss Rate (1 - Recall)")
    ax.set_title("Miss Rate by Variant")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(output_dir, "miss_rate_by_variant.png")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    # 4. Evidence alignment by variant
    fig, ax = plt.subplots(figsize=(14, 6))
    means = scored.groupby("variant_id")["evidence_hit_rate"].mean().reindex(variant_order)
    ax.bar(range(len(means)), means.values, color=[color_map[v] for v in means.index])
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=45, ha="right")
    ax.set_ylabel("Evidence Hit Rate")
    ax.set_title("Evidence Alignment by Variant")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(output_dir, "evidence_alignment_by_variant.png")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    # 5. Environmental ERRS by variant
    if "environmental_ERRS" in scored.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        means = scored.groupby("variant_id")["environmental_ERRS"].mean().reindex(variant_order)
        ax.bar(range(len(means)), means.values, color=[color_map[v] for v in means.index])
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=45, ha="right")
        ax.set_ylabel("Environmental ERRS")
        ax.set_title("Environmental Reporting Readiness Score by Variant")
        ax.set_ylim(0, 1)
        for i, v in enumerate(means.values):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        path = os.path.join(output_dir, "environmental_ERRS_by_variant.png")
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    # 6. Latency by variant
    fig, ax = plt.subplots(figsize=(14, 6))
    means = scored.groupby("variant_id")["total_latency_ms"].mean().reindex(variant_order)
    ax.bar(range(len(means)), means.values, color=[color_map[v] for v in means.index])
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=45, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Total Latency by Variant")
    for i, v in enumerate(means.values):
        ax.text(i, v + max(means.values) * 0.02, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "latency_by_variant.png")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    # 7. Doc type F1 heatmap
    doc_types = sorted(scored["doc_type"].unique())
    if len(doc_types) > 1:
        fig, ax = plt.subplots(figsize=(14, 8))
        heatmap_data = np.zeros((len(variant_order), len(doc_types)))
        for i, vid in enumerate(variant_order):
            for j, dt in enumerate(doc_types):
                subset = scored[(scored["variant_id"] == vid) & (scored["doc_type"] == dt)]
                heatmap_data[i, j] = subset["overall_relaxed_f1"].mean() if not subset.empty else 0

        im = ax.imshow(heatmap_data, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(doc_types)))
        ax.set_xticklabels(doc_types, rotation=45, ha="right")
        ax.set_yticks(range(len(variant_order)))
        ax.set_yticklabels(variant_order)
        ax.set_title("Relaxed F1 by Variant × Document Type")
        plt.colorbar(im, label="Relaxed F1")
        for i in range(len(variant_order)):
            for j in range(len(doc_types)):
                ax.text(j, i, f"{heatmap_data[i, j]:.2f}", ha="center", va="center", fontsize=7)
        plt.tight_layout()
        path = os.path.join(output_dir, "doc_type_f1_heatmap.png")
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    # 8. OCR ablation for scanned docs
    scanned = scored[scored["is_scanned"] == True]  # noqa: E712
    if not scanned.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ocr_modes = sorted(scanned["ocr_mode"].unique())
        for oi, ocr_mode in enumerate(ocr_modes):
            subset = scanned[scanned["ocr_mode"] == ocr_mode]
            means = subset.groupby("variant_id")["overall_relaxed_f1"].mean()
            x = range(len(means))
            ax.bar([xi + oi * 0.25 for xi in x], means.values, width=0.25, label=f"OCR: {ocr_mode}")
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=45, ha="right")
        ax.set_ylabel("Relaxed F1")
        ax.set_title("OCR Ablation on Scanned Documents")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, "ocr_ablation_scanned_docs.png")
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    # 9. AWFA delta vs baseline
    if "V0_BASELINE" in scored["variant_id"].values:
        baseline_f1 = scored[scored["variant_id"] == "V0_BASELINE"]["overall_relaxed_f1"].mean()
        fig, ax = plt.subplots(figsize=(14, 6))
        deltas = []
        labels = []
        for vid in variant_order:
            if vid == "V0_BASELINE":
                continue
            v_f1 = scored[scored["variant_id"] == vid]["overall_relaxed_f1"].mean()
            delta = v_f1 - baseline_f1
            deltas.append(delta)
            labels.append(vid)

        bar_colors = ["green" if d >= 0 else "red" for d in deltas]
        ax.bar(range(len(deltas)), deltas, color=bar_colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Δ Relaxed F1 vs Baseline")
        ax.set_title("F1 Delta vs V0 Baseline")
        ax.axhline(y=0, color="black", linewidth=0.5)
        for i, v in enumerate(deltas):
            ax.text(i, v + 0.005 if v >= 0 else v - 0.015, f"{v:+.3f}", ha="center", fontsize=7)
        plt.tight_layout()
        path = os.path.join(output_dir, "awfa_delta_vs_baseline.png")
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    return saved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark figures")
    parser.add_argument("--csv", default="benchmarks/results/axiomesg_benchmark_runs.csv")
    parser.add_argument("--out", default="benchmarks/results/figures")
    args = parser.parse_args()
    paths = generate_figures(args.csv, args.out)
    print(f"Generated {len(paths)} figures in {args.out}/")
    for p in paths:
        print(f"  {os.path.basename(p)}")
