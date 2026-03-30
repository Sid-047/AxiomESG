"""
Synthetic ESG Document & Ground Truth Generator
================================================

Generates realistic synthetic ESG disclosures in PDF (text), PDF (scanned-image),
DOCX, and CSV formats with embedded ground truth for benchmarking.

Each document contains Environmental, Social, and Governance metrics with noise
(irrelevant numbers, non-ESG sections, table-like blocks, abbreviations).

Usage:
    python -m benchmarks.src.generate_synthetic_dataset --out benchmarks/dataset --n 50 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import string
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ground truth data model
# ---------------------------------------------------------------------------

@dataclass
class GTMetric:
    name: str
    value: str
    unit: str
    year: str
    source_text: str
    category: str  # E, S, or G


@dataclass
class GroundTruthDoc:
    doc_id: str
    file_path: str
    doc_type: str        # pdf_text, pdf_scanned, docx, csv
    is_scanned: bool
    company_name: str
    metrics: List[GTMetric] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metric templates — carefully authored original content (no real company data)
# ---------------------------------------------------------------------------

COMPANY_NAMES = [
    "Verdant Industries Ltd.", "AquaSphere Holdings", "TerraNova Resources Corp.",
    "Pinnacle Energy Group", "BlueSky Manufacturing", "Meridian Logistics Inc.",
    "Crestwood Materials", "Solaris Power Co.", "Ironclad Mining Partners",
    "Evergreen Consumer Goods", "Northwind Chemicals", "Silverline Pharma",
    "OceanEdge Fisheries", "SummitTech Solutions", "Horizon Agri Corp.",
    "CarbonLite Ventures", "GreenPath Packaging", "Apex Metals International",
    "CleanWave Utilities", "PrimeForge Steel", "BioCore Innovation",
    "Atlas Cement Group", "Windward Shipping LLC", "FusionGrid Energy",
    "Ember Textiles Co.", "Vanguard Paper Mills", "QuantumBridge Electronics",
    "Cascadia Timber Holdings", "Nexum Waste Solutions", "StellarChem Industries",
]

ENV_METRIC_TEMPLATES = [
    {
        "name": "Scope 1 GHG Emissions",
        "unit": "tCO2e",
        "value_range": (5000, 250000),
        "sentence": "During {year}, {company} reported total Scope 1 greenhouse gas emissions of {value} {unit}, measured in accordance with the GHG Protocol Corporate Standard.",
    },
    {
        "name": "Scope 2 GHG Emissions",
        "unit": "tCO2e",
        "value_range": (2000, 180000),
        "sentence": "Scope 2 (location-based) emissions for the reporting period {year} amounted to {value} {unit}, reflecting electricity consumption across all operational facilities.",
    },
    {
        "name": "Scope 3 GHG Emissions",
        "unit": "tCO2e",
        "value_range": (10000, 500000),
        "sentence": "Scope 3 value chain emissions were estimated at {value} {unit} in {year}, covering upstream supply chain logistics and downstream distribution.",
    },
    {
        "name": "Total Energy Consumption",
        "unit": "MWh",
        "value_range": (8000, 400000),
        "sentence": "Total energy consumption across operations reached {value} {unit} in {year}, including both renewable and non-renewable sources.",
    },
    {
        "name": "Renewable Energy Share",
        "unit": "%",
        "value_range": (5, 85),
        "sentence": "Renewable energy accounted for {value}{unit} of total energy usage during {year}, an increase from the prior reporting period.",
    },
    {
        "name": "Total Water Withdrawal",
        "unit": "megalitres",
        "value_range": (50, 12000),
        "sentence": "In {year}, total water withdrawal from all sources was {value} {unit}, with freshwater making up the majority of intake.",
    },
    {
        "name": "Total Waste Generated",
        "unit": "metric tons",
        "value_range": (200, 50000),
        "sentence": "The company generated {value} {unit} of total waste in {year}, of which a significant portion was diverted from landfill through recycling and recovery programs.",
    },
    {
        "name": "Waste Recycling Rate",
        "unit": "%",
        "value_range": (15, 92),
        "sentence": "The recycling and recovery rate for operational waste stood at {value}{unit} for the year ending {year}.",
    },
    {
        "name": "Carbon Intensity",
        "unit": "tCO2e per million USD revenue",
        "value_range": (10, 350),
        "sentence": "Carbon intensity was calculated at {value} {unit} for fiscal year {year}, representing a decrease compared to the baseline year.",
    },
    {
        "name": "NOx Emissions",
        "unit": "metric tons",
        "value_range": (5, 800),
        "sentence": "Nitrogen oxide (NOx) air emissions totaled {value} {unit} during {year}, monitored at all major combustion installations.",
    },
]

SOC_METRIC_TEMPLATES = [
    {
        "name": "Total Recordable Incident Rate",
        "unit": "per 200,000 hours worked",
        "value_range_float": (0.1, 4.5),
        "sentence": "The Total Recordable Incident Rate (TRIR) was {value} {unit} in {year}, reflecting our ongoing commitment to workplace safety.",
    },
    {
        "name": "Lost Time Injury Frequency Rate",
        "unit": "per million hours worked",
        "value_range_float": (0.05, 3.0),
        "sentence": "Lost Time Injury Frequency Rate (LTIFR) stood at {value} {unit} for {year}, with zero fatalities recorded across all sites.",
    },
    {
        "name": "Gender Diversity Ratio (Female)",
        "unit": "%",
        "value_range": (18, 52),
        "sentence": "Female representation across the global workforce was {value}{unit} as of year-end {year}, up from the prior period.",
    },
    {
        "name": "Board Gender Diversity (Female)",
        "unit": "%",
        "value_range": (20, 50),
        "sentence": "Women comprised {value}{unit} of the Board of Directors as at the end of {year}.",
    },
    {
        "name": "Average Training Hours per Employee",
        "unit": "hours",
        "value_range": (8, 65),
        "sentence": "Employees received an average of {value} {unit} of professional development and training during {year}.",
    },
    {
        "name": "Employee Turnover Rate",
        "unit": "%",
        "value_range": (5, 28),
        "sentence": "Voluntary employee turnover was {value}{unit} for {year}, within the industry benchmark range.",
    },
    {
        "name": "Community Investment",
        "unit": "USD",
        "value_range": (100000, 5000000),
        "sentence": "Total community investment and charitable contributions reached {value} {unit} during {year}.",
    },
    {
        "name": "Employee Engagement Score",
        "unit": "%",
        "value_range": (55, 92),
        "sentence": "The annual employee engagement survey yielded a score of {value}{unit} for {year}.",
    },
]

GOV_METRIC_TEMPLATES = [
    {
        "name": "Board Independence",
        "unit": "%",
        "value_range": (40, 90),
        "sentence": "Independent directors comprised {value}{unit} of the Board in {year}, exceeding the regulatory minimum.",
    },
    {
        "name": "Anti-Corruption Training Completion",
        "unit": "%",
        "value_range": (85, 100),
        "sentence": "Anti-corruption and ethics training completion rate was {value}{unit} across all employees and contractors for {year}.",
    },
    {
        "name": "Board Meeting Attendance Rate",
        "unit": "%",
        "value_range": (80, 100),
        "sentence": "Average board meeting attendance rate was {value}{unit} during {year}.",
    },
    {
        "name": "Whistleblower Reports Received",
        "unit": "cases",
        "value_range": (0, 45),
        "sentence": "A total of {value} {unit} were received through the anonymous whistleblower hotline in {year}, all of which were investigated.",
    },
    {
        "name": "Data Privacy Incidents",
        "unit": "incidents",
        "value_range": (0, 8),
        "sentence": "The organization recorded {value} data privacy {unit} during {year}, none of which constituted a material breach.",
    },
    {
        "name": "Audit Committee Meetings",
        "unit": "meetings",
        "value_range": (4, 12),
        "sentence": "The Audit & Risk Committee convened {value} {unit} in {year} to review internal controls and financial reporting.",
    },
    {
        "name": "ESG-Linked Executive Compensation",
        "unit": "%",
        "value_range": (5, 30),
        "sentence": "ESG performance metrics represented {value}{unit} of executive variable compensation for fiscal year {year}.",
    },
]


# ---------------------------------------------------------------------------
# Noise content generators
# ---------------------------------------------------------------------------

NOISE_PARAGRAPHS = [
    "This document has been prepared for informational purposes only and does not constitute an offer or solicitation. Past performance is not indicative of future results. All figures are unaudited unless stated otherwise.",
    "The company was founded in 1987 and has grown to operate across 14 countries with over 8,500 employees. Our headquarters are located at 42 Industrial Boulevard, with regional offices in Singapore, Munich, and São Paulo.",
    "Revenue for Q3 reached $127.4 million, compared to $118.9 million in the same quarter of the previous year. Operating margin improved by 230 basis points to 14.7%. Earnings per share were $1.23 (diluted).",
    "Capital expenditure for the plant expansion project totaled $34.2 million, with an additional $12.8 million allocated to IT infrastructure upgrades. Depreciation and amortization came to $19.6 million.",
    "The company maintains a fleet of 342 vehicles and operates from 28 distribution centers. Average delivery time improved to 2.3 business days from 3.1 business days in the prior year.",
    "Total assets stood at $2.4 billion as of December 31. Current ratio was 1.8:1 and debt-to-equity ratio was 0.42. Credit rating was confirmed at BBB+ by the rating agency.",
    "Market share in the domestic segment increased to approximately 22%, while international operations contributed 38% of consolidated revenue. Currency translation effects were immaterial.",
    "Inventory turnover ratio improved to 6.2x from 5.8x. Accounts receivable days outstanding decreased to 47 days. Free cash flow generation was $89.3 million.",
    "The Board approved a dividend of $0.45 per share, representing a payout ratio of 36.6%. Share buyback program authorized up to $50 million over the next 12 months.",
    "Note: Certain forward-looking statements in this report are based on management's current expectations and assumptions. Actual results may differ materially due to various risk factors.",
]

NOISE_TABLE_BLOCKS = [
    "Product Line | Q1 | Q2 | Q3 | Q4\nAlpha Series | 12,400 | 13,100 | 14,800 | 15,200\nBeta Range | 8,700 | 9,200 | 8,900 | 10,100\nGamma Plus | 5,300 | 5,800 | 6,100 | 6,400",
    "Region | Headcount | Revenue ($M) | Growth (%)\nNorth America | 3,200 | 456.7 | 8.3\nEurope | 2,100 | 312.4 | 5.1\nAsia Pacific | 1,800 | 198.2 | 12.7\nOther | 1,400 | 87.3 | 3.9",
    "Cost Category | 2022 ($M) | 2023 ($M) | Change (%)\nRaw Materials | 234.5 | 251.2 | 7.1\nLabor | 178.3 | 185.9 | 4.3\nLogistics | 67.8 | 72.1 | 6.3\nOverhead | 45.2 | 43.8 | -3.1",
]


def _gen_noise_section(rng: random.Random, min_blocks: int = 2, max_blocks: int = 5) -> str:
    """Generate a block of irrelevant content to serve as noise."""
    blocks = []
    n = rng.randint(min_blocks, max_blocks)
    for _ in range(n):
        if rng.random() < 0.25:
            blocks.append(rng.choice(NOISE_TABLE_BLOCKS))
        else:
            blocks.append(rng.choice(NOISE_PARAGRAPHS))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Metric instantiation
# ---------------------------------------------------------------------------

def _gen_value(template: Dict[str, Any], rng: random.Random) -> str:
    if "value_range_float" in template:
        lo, hi = template["value_range_float"]
        val = round(rng.uniform(lo, hi), 2)
        return str(val)
    elif "value_range" in template:
        lo, hi = template["value_range"]
        val = rng.randint(lo, hi)
        # Sometimes format with commas for realism
        if val >= 1000 and rng.random() < 0.5:
            return f"{val:,}"
        return str(val)
    return "0"


def _instantiate_metrics(
    templates: List[Dict[str, Any]],
    category: str,
    company: str,
    year: str,
    rng: random.Random,
    count: int | None = None,
) -> List[Tuple[GTMetric, str]]:
    """Pick a subset of metric templates and instantiate them.

    Returns list of (GTMetric, source_sentence) pairs.
    """
    if count is None:
        count = rng.randint(2, min(len(templates), 6))
    chosen = rng.sample(templates, min(count, len(templates)))
    results = []
    for tmpl in chosen:
        value = _gen_value(tmpl, rng)
        sentence = tmpl["sentence"].format(
            year=year, company=company, value=value, unit=tmpl["unit"]
        )
        gt = GTMetric(
            name=tmpl["name"],
            value=value,
            unit=tmpl["unit"],
            year=year,
            source_text=sentence,
            category=category,
        )
        results.append((gt, sentence))
    return results


# ---------------------------------------------------------------------------
# Full document text assembly
# ---------------------------------------------------------------------------

def _build_document_text(
    company: str,
    year: str,
    env_items: List[Tuple[GTMetric, str]],
    soc_items: List[Tuple[GTMetric, str]],
    gov_items: List[Tuple[GTMetric, str]],
    rng: random.Random,
) -> str:
    """Assemble a realistic ESG disclosure document text."""
    sections: List[str] = []

    # Title page
    sections.append(
        f"{company}\n"
        f"Annual ESG & Sustainability Disclosure Report - Fiscal Year {year}\n"
        f"{'=' * 60}\n"
    )

    # Noise intro
    sections.append("CORPORATE OVERVIEW\n" + "-" * 30)
    sections.append(_gen_noise_section(rng, 1, 3))

    # Environmental section
    sections.append("\nENVIRONMENTAL PERFORMANCE\n" + "-" * 30)
    sections.append(
        f"{company} is committed to reducing its environmental footprint. "
        f"Below are the key environmental metrics for fiscal year {year}."
    )
    for _, sentence in env_items:
        if rng.random() < 0.3:
            sections.append(_gen_noise_section(rng, 1, 1))
        sections.append(sentence)

    # Noise between sections
    sections.append(_gen_noise_section(rng, 1, 2))

    # Social section
    sections.append("\nSOCIAL RESPONSIBILITY & WORKFORCE\n" + "-" * 30)
    sections.append(
        f"Our people are the foundation of {company}'s success. "
        f"We track the following social indicators for {year}."
    )
    for _, sentence in soc_items:
        if rng.random() < 0.25:
            sections.append(_gen_noise_section(rng, 1, 1))
        sections.append(sentence)

    # Governance section
    sections.append("\nGOVERNANCE & ETHICS\n" + "-" * 30)
    sections.append(
        f"Strong governance practices underpin {company}'s long-term strategy. "
        f"The following governance metrics were reported for {year}."
    )
    for _, sentence in gov_items:
        if rng.random() < 0.2:
            sections.append(_gen_noise_section(rng, 1, 1))
        sections.append(sentence)

    # Footer noise
    sections.append("\n\nDISCLAIMER & NOTES\n" + "-" * 30)
    sections.append(_gen_noise_section(rng, 1, 2))
    sections.append(
        f"\nPrepared by {company} Sustainability Department. "
        f"Report reference: {''.join(rng.choices(string.ascii_uppercase + string.digits, k=8))}"
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Document file writers
# ---------------------------------------------------------------------------

def _write_pdf_text(text: str, path: str) -> None:
    """Write a text-based PDF using raw PDF syntax."""
    _write_raw_pdf(text, path)


def _write_raw_pdf(text: str, path: str) -> None:
    """
    Minimal PDF writer using no external dependencies.
    Produces a valid PDF 1.4 with embedded text.
    """
    lines = text.encode("latin-1", errors="replace").decode("latin-1")
    # Build a simple PDF
    stream_lines = []
    y = 750
    for line in lines.split("\n"):
        if y < 50:
            stream_lines.append("ET")
            stream_lines.append("endstream")
            y = 750
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"BT /F1 10 Tf 50 {y} Td ({escaped}) Tj ET")
        y -= 12

    stream_content = "\n".join(stream_lines)
    objects = []

    # Obj 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")
    # Obj 2: Pages
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj")
    # Obj 3: Page
    objects.append(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj"
    )
    # Obj 4: Content stream
    objects.append(
        f"4 0 obj\n<< /Length {len(stream_content)} >>\nstream\n{stream_content}\nendstream\nendobj"
    )
    # Obj 5: Font
    objects.append(
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj"
    )

    body = "\n".join(objects)
    xref_offset = len("%PDF-1.4\n") + len(body) + 1
    xref = f"xref\n0 6\n0000000000 65535 f \n"
    offset = len("%PDF-1.4\n")
    for i, obj in enumerate(objects):
        xref += f"{offset:010d} 00000 n \n"
        offset += len(obj) + 1

    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"

    with open(path, "w", encoding="latin-1") as f:
        f.write("%PDF-1.4\n")
        f.write(body + "\n")
        f.write(xref)
        f.write(trailer)


def _write_pdf_scanned(text: str, path: str) -> None:
    """
    Create a 'scanned' PDF by rendering text as an image, then embedding
    that image into a PDF. This simulates an image-only scanned document.
    Falls back to text PDF if PIL/reportlab are not available.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        _write_pdf_text(text, path)
        return

    # Render text to image
    width, margin = 1200, 40
    font_size = 16
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    lines = []
    for paragraph in text.split("\n"):
        # Word wrap
        words = paragraph.split()
        current_line = ""
        for word in words:
            test = f"{current_line} {word}".strip()
            if len(test) * (font_size * 0.6) > (width - 2 * margin):
                lines.append(current_line)
                current_line = word
            else:
                current_line = test
        if current_line:
            lines.append(current_line)
        lines.append("")  # Blank line between paragraphs

    line_height = font_size + 4
    height = max(margin * 2 + len(lines) * line_height, 800)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill="black", font=font)
        y += line_height

    # Save as single-page PDF via Pillow
    img.save(path, "PDF", resolution=150)


def _write_docx(text: str, path: str) -> None:
    """Write a DOCX file."""
    from docx import Document
    doc = Document()
    for para in text.split("\n"):
        stripped = para.strip()
        if stripped:
            doc.add_paragraph(stripped)
    doc.save(path)


def _write_csv(
    metrics: List[GTMetric],
    company: str,
    year: str,
    rng: random.Random,
    path: str,
) -> str:
    """
    Write a CSV file that embeds ESG metrics in a tabular format,
    along with noise rows. Returns the full text content of the CSV.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Category", "Metric Name", "Value", "Unit", "Reporting Year", "Notes"])

    # Noise rows before
    for _ in range(rng.randint(2, 5)):
        writer.writerow([
            rng.choice(["Financial", "Operations", "Admin"]),
            rng.choice(["Revenue", "Headcount", "CAPEX", "OPEX", "Inventory"]),
            str(rng.randint(100, 99999)),
            rng.choice(["USD", "units", "FTE", "$M"]),
            year,
            "",
        ])

    for m in metrics:
        notes = m.source_text[:80] if rng.random() < 0.5 else ""
        cat_label = {"E": "Environmental", "S": "Social", "G": "Governance"}.get(m.category, m.category)
        writer.writerow([cat_label, m.name, m.value, m.unit, m.year, notes])

        # Occasional noise row interspersed
        if rng.random() < 0.3:
            writer.writerow([
                "Operations",
                rng.choice(["Throughput", "Capacity Util.", "Downtime"]),
                str(round(rng.uniform(10, 99), 1)),
                "%",
                year,
                "",
            ])

    # Noise rows after
    for _ in range(rng.randint(1, 3)):
        writer.writerow([
            "Financial",
            rng.choice(["Net Income", "EBITDA", "Cash Flow"]),
            str(rng.randint(1000000, 50000000)),
            "USD",
            year,
            "",
        ])

    csv_text = output.getvalue()
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(csv_text)
    return csv_text


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset(
    output_dir: str,
    n_docs: int = 50,
    seed: int = 42,
    doc_type_weights: Dict[str, float] | None = None,
    augmentation_tag: str = "",
    stress_focus: str = "",
) -> List[GroundTruthDoc]:
    """
    Generate n_docs synthetic ESG documents with ground truth.

    Args:
        output_dir: Root directory for dataset output
        n_docs: Number of documents to generate
        seed: Random seed for reproducibility
        doc_type_weights: Fraction per doc type (pdf_text, pdf_scanned, docx, csv)
        augmentation_tag: If non-empty, appended to doc IDs (e.g., 'aug1')
        stress_focus: If set, bias generation towards a specific stress type:
                      'ocr', 'filter', 'weight', 'llm'

    Returns:
        List of GroundTruthDoc objects
    """
    rng = random.Random(seed)

    if doc_type_weights is None:
        doc_type_weights = {"pdf_text": 0.35, "pdf_scanned": 0.20, "docx": 0.30, "csv": 0.15}

    # Adjust weights for stress testing
    if stress_focus == "ocr":
        doc_type_weights = {"pdf_text": 0.10, "pdf_scanned": 0.70, "docx": 0.10, "csv": 0.10}
    elif stress_focus == "filter":
        pass  # More synonyms/abbreviations handled in text generation
    elif stress_focus == "weight":
        pass  # More similar/boilerplate lines
    elif stress_focus == "llm":
        pass  # Weird formatting

    # Build type assignments
    types = list(doc_type_weights.keys())
    weights = [doc_type_weights[t] for t in types]
    type_assignments = rng.choices(types, weights=weights, k=n_docs)

    docs_dir = os.path.join(output_dir, "synthetic_docs")
    gt_dir = os.path.join(output_dir, "ground_truth")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "real_docs"), exist_ok=True)

    results: List[GroundTruthDoc] = []
    years = ["2022", "2023", "2024"]

    for i in range(n_docs):
        doc_type = type_assignments[i]
        prefix = augmentation_tag + "_" if augmentation_tag else ""
        doc_id = f"{prefix}synth_{i:04d}"
        company = rng.choice(COMPANY_NAMES)
        year = rng.choice(years)

        # Determine metric counts — more for stress scenarios
        env_count = rng.randint(3, 6) if stress_focus == "weight" else rng.randint(2, 5)
        soc_count = rng.randint(2, 5) if stress_focus == "weight" else rng.randint(2, 4)
        gov_count = rng.randint(2, 5) if stress_focus == "weight" else rng.randint(2, 4)

        env_items = _instantiate_metrics(ENV_METRIC_TEMPLATES, "E", company, year, rng, env_count)
        soc_items = _instantiate_metrics(SOC_METRIC_TEMPLATES, "S", company, year, rng, soc_count)
        gov_items = _instantiate_metrics(GOV_METRIC_TEMPLATES, "G", company, year, rng, gov_count)

        all_items = env_items + soc_items + gov_items
        all_metrics = [item[0] for item in all_items]

        # Build text
        if doc_type == "csv":
            ext = ".csv"
            file_name = f"{doc_id}{ext}"
            file_path = os.path.join(docs_dir, file_name)
            text_content = _write_csv(all_metrics, company, year, rng, file_path)
        else:
            text_content = _build_document_text(
                company, year, env_items, soc_items, gov_items, rng
            )

            # Apply stress-focus text mutations
            if stress_focus == "filter":
                # Add synonyms, abbreviations, bullet-list formatting
                text_content = _apply_filter_stress(text_content, rng)
            elif stress_focus == "weight":
                # Add many similar boilerplate lines
                text_content = _apply_weight_stress(text_content, rng)
            elif stress_focus == "llm":
                # Weird formatting, mixed units, unusual headings
                text_content = _apply_llm_stress(text_content, rng)

            if doc_type == "pdf_text":
                ext = ".pdf"
                file_name = f"{doc_id}{ext}"
                file_path = os.path.join(docs_dir, file_name)
                _write_pdf_text(text_content, file_path)
            elif doc_type == "pdf_scanned":
                ext = ".pdf"
                file_name = f"{doc_id}_scanned{ext}"
                file_path = os.path.join(docs_dir, file_name)
                _write_pdf_scanned(text_content, file_path)
            elif doc_type == "docx":
                ext = ".docx"
                file_name = f"{doc_id}{ext}"
                file_path = os.path.join(docs_dir, file_name)
                _write_docx(text_content, file_path)

        # Build ground truth
        gt_doc = GroundTruthDoc(
            doc_id=doc_id,
            file_path=os.path.relpath(file_path, start=os.path.dirname(output_dir)),
            doc_type=doc_type,
            is_scanned=(doc_type == "pdf_scanned"),
            company_name=company,
            metrics=all_metrics,
        )
        results.append(gt_doc)

        # Write GT JSON
        gt_json = {
            "doc_id": gt_doc.doc_id,
            "file_path": gt_doc.file_path,
            "doc_type": gt_doc.doc_type,
            "is_scanned": gt_doc.is_scanned,
            "company_name": gt_doc.company_name,
            "metrics": [asdict(m) for m in gt_doc.metrics],
        }
        gt_path = os.path.join(gt_dir, f"{doc_id}.json")
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_json, f, indent=2, ensure_ascii=False)

    # Write manifest
    manifest = {
        "total_documents": len(results),
        "seed": seed,
        "augmentation_tag": augmentation_tag,
        "stress_focus": stress_focus,
        "doc_type_distribution": {
            t: sum(1 for d in results if d.doc_type == t) for t in types
        },
        "documents": [
            {"doc_id": d.doc_id, "file_path": d.file_path, "doc_type": d.doc_type}
            for d in results
        ],
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Stress-focus text mutators
# ---------------------------------------------------------------------------

def _apply_filter_stress(text: str, rng: random.Random) -> str:
    """Make text harder for keyword filters: synonyms, abbreviations, bullet lists."""
    substitutions = [
        ("emissions", rng.choice(["GHG output", "CO2 releases", "emission volumes"])),
        ("water withdrawal", rng.choice(["H2O intake", "water extracted"])),
        ("diversity", rng.choice(["representation", "demographic balance"])),
        ("governance", rng.choice(["corp. oversight", "mgmt. structure"])),
        ("training", rng.choice(["L&D", "capability building"])),
    ]
    for old, new in substitutions:
        if rng.random() < 0.6:
            text = text.replace(old, new, 1)

    # Convert some paragraphs to bullet lists
    lines = text.split("\n")
    for idx in range(len(lines)):
        if rng.random() < 0.15 and len(lines[idx]) > 20:
            lines[idx] = "  • " + lines[idx]
    return "\n".join(lines)


def _apply_weight_stress(text: str, rng: random.Random) -> str:
    """Add boilerplate and repetitive lines to stress weighting/dedup."""
    boilerplate = [
        "The company continues to monitor and report on key performance indicators.",
        "All data has been verified by internal audit procedures.",
        "Performance targets are reviewed annually by the board.",
        "Stakeholder engagement remains a priority for our organization.",
        "We adhere to internationally recognized reporting frameworks.",
    ]
    lines = text.split("\n")
    augmented = []
    for line in lines:
        augmented.append(line)
        if rng.random() < 0.2:
            augmented.extend(rng.sample(boilerplate, min(3, len(boilerplate))))
    return "\n".join(augmented)


def _apply_llm_stress(text: str, rng: random.Random) -> str:
    """Unusual formatting to challenge LLM JSON generation."""
    # Mix case in headings
    lines = text.split("\n")
    for idx in range(len(lines)):
        if lines[idx].isupper() and len(lines[idx]) > 5:
            if rng.random() < 0.4:
                lines[idx] = lines[idx].title() + " ---"
        # Randomly insert pipes and dashes
        if rng.random() < 0.1:
            lines[idx] = "| " + lines[idx] + " |"
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic ESG benchmark dataset"
    )
    parser.add_argument("--out", default="benchmarks/dataset", help="Output directory")
    parser.add_argument("--n", type=int, default=50, help="Number of documents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stress", default="", choices=["", "ocr", "filter", "weight", "llm"],
                        help="Stress-focus for augmentation")
    parser.add_argument("--tag", default="", help="Augmentation tag")
    args = parser.parse_args()

    docs = generate_dataset(
        output_dir=args.out,
        n_docs=args.n,
        seed=args.seed,
        stress_focus=args.stress,
        augmentation_tag=args.tag,
    )
    print(f"Generated {len(docs)} documents in {args.out}/")
    for dtype in ["pdf_text", "pdf_scanned", "docx", "csv"]:
        count = sum(1 for d in docs if d.doc_type == dtype)
        print(f"  {dtype}: {count}")


if __name__ == "__main__":
    main()
