"""
Report generator for ESG pipeline output.

Converts an ESGOutput dict into a structured Markdown report.
"""
from __future__ import annotations

from typing import Any, Dict

from app.pipeline.strategies import STRATEGY_META


def generate_report(result: Dict[str, Any]) -> str:
    """
    Generate a Markdown report from a pipeline result dict.

    Args:
        result: The ESGOutput model_dump() dictionary.

    Returns:
        A formatted Markdown string.
    """
    meta = result.get("metadata", {})
    agg = result.get("aggregation", {})
    env = result.get("environmental", {})
    soc = result.get("social", {})
    gov = result.get("governance", {})

    algorithm_key = meta.get("algorithm_used", "heuristic")
    algorithm_meta = STRATEGY_META.get(algorithm_key, {})
    algorithm_label = algorithm_meta.get("label", algorithm_key)
    algorithm_desc = algorithm_meta.get("description", "")

    lines = []

    # Header
    lines.append("# AxiomESG — ESG Extraction Report")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Metadata
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| **Extraction Date** | {meta.get('extraction_date', 'N/A')} |")
    lines.append(f"| **Source Files** | {', '.join(meta.get('source_files', []))} |")
    lines.append(f"| **Model Provider** | {meta.get('model_provider', 'N/A')} |")
    lines.append(f"| **Model Name** | {meta.get('model_name', 'N/A')} |")
    lines.append(f"| **Algorithm** | {algorithm_label} |")
    lines.append(f"| **AWFA Weights Preserved** | {'Yes' if meta.get('awfa_weights_preserved') else 'No'} |")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Total Documents Processed**: {agg.get('total_documents', 0)}")
    lines.append(f"- **Total ESG Sentences Identified**: {agg.get('total_esg_sentences', 0)}")
    lines.append(f"- **Total Weighted Blocks**: {agg.get('total_weighted_blocks', 0)}")
    lines.append(f"- **OCR Used**: {'Yes' if agg.get('ocr_used') else 'No'}")
    lines.append("")

    # Confidence Overview
    lines.append("### Confidence Scores")
    lines.append("")
    lines.append("| Pillar | Confidence |")
    lines.append("|---|---|")
    lines.append(f"| Environmental | {env.get('confidence_score', 0.0):.1%} |")
    lines.append(f"| Social | {soc.get('confidence_score', 0.0):.1%} |")
    lines.append(f"| Governance | {gov.get('confidence_score', 0.0):.1%} |")
    lines.append("")

    # E/S/G Sections
    for section_key, section_data, section_title in [
        ("environmental", env, "Environmental"),
        ("social", soc, "Social"),
        ("governance", gov, "Governance"),
    ]:
        lines.append(f"## {section_title}")
        lines.append("")

        # Narrative
        narrative = section_data.get("narrative", "No data available.")
        lines.append(f"### Narrative")
        lines.append("")
        lines.append(narrative)
        lines.append("")

        # Metrics
        metrics = section_data.get("metrics", [])
        if metrics:
            lines.append(f"### Metrics")
            lines.append("")
            lines.append("| Metric | Value | Unit | Year | Source |")
            lines.append("|---|---|---|---|---|")
            for m in metrics:
                name = m.get("name", "")
                value = m.get("value", "")
                unit = m.get("unit", "") or "—"
                year = m.get("year", "") or "—"
                source = m.get("source_text", "")
                # Truncate source for table readability
                if len(source) > 80:
                    source = source[:77] + "..."
                lines.append(f"| {name} | {value} | {unit} | {year} | {source} |")
            lines.append("")

        # Top Evidence
        evidence = section_data.get("top_evidence", [])
        if evidence:
            lines.append(f"### Top Evidence Spans")
            lines.append("")
            for i, ev in enumerate(evidence[:5], 1):
                text = ev.get("text", "")
                weight = ev.get("weight", 0.0)
                source = ev.get("source_file", "unknown")
                lines.append(f"{i}. **[{weight:.3f}]** {text}")
                lines.append(f"   _Source: {source}_")
                lines.append("")

        lines.append("---")
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append(f"**Algorithm**: {algorithm_label}")
    lines.append("")
    if algorithm_desc:
        lines.append(f"> {algorithm_desc}")
        lines.append("")
    lines.append(
        "The extraction pipeline follows a deterministic, stage-driven approach: "
        "documents are ingested, text is extracted (with optional OCR for scanned content), "
        "sentences are filtered for ESG relevance using keyword matching, "
        "evidence spans are weighted and deduplicated using the selected algorithm, "
        "and a single LLM call standardizes the output into the canonical ESG JSON schema. "
        "The result is validated against a strict Pydantic v2 schema before being returned."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by AxiomESG*")

    return "\n".join(lines)
