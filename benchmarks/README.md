# AxiomESG Benchmarking & Ablation Harness

A research-paper-ready benchmarking system for evaluating the AxiomESG ESG intelligence pipeline across multiple algorithm variants, document types, and pipeline configurations.

---

## Quick Start

### 1. Install Dependencies

```bash
cd /Volumes/ReserveDisk/codeBase/AxiomESG
pip install -r backend/requirements.txt
pip install pyyaml pandas numpy fpdf2
```

### 2. Generate Synthetic Dataset

```bash
python -m benchmarks.src.generate_synthetic_dataset --out benchmarks/dataset --n 50 --seed 42
```

This creates:
- `benchmarks/dataset/synthetic_docs/` — 50 synthetic ESG documents (PDF, DOCX, CSV)
- `benchmarks/dataset/ground_truth/` — Per-document ground truth JSON
- `benchmarks/dataset/manifest.json` — Dataset manifest

### 3. Run Benchmarks

```bash
python -m benchmarks.src.run_benchmarks --config benchmarks/config/benchmark.yaml
```

This produces:
- `benchmarks/results/axiomesg_benchmark_runs.csv` — One row per (doc, variant, replica)
- `benchmarks/artifacts/` — Per-run output JSONs

### 4. Generate Report

```bash
python -m benchmarks.src.report --csv benchmarks/results/axiomesg_benchmark_runs.csv
```

Produces: `benchmarks/results/REPORT.md`

### 5. Run with Resolution Plan (Auto-Augmentation)

```bash
python -m benchmarks.src.run_benchmarks --config benchmarks/config/benchmark.yaml --augment --report
```

This will:
1. Run the full benchmark matrix (500+ runs)
2. Diagnose weak metrics (F1, groundedness, schema validity)
3. Auto-generate targeted augmentation documents
4. Re-run selected variants on augmented data
5. Generate the final report with delta summaries

---

## Experiment Variants

| Variant | Description | Filter | Weight | Algorithm |
|---------|------------|--------|--------|-----------|
| V0 | Baseline (no filter, no weight) | ✗ | ✗ | passthrough |
| V1 | Filter only | ✓ | ✗ | passthrough |
| V2 | Weight only (no filter) | ✗ | ✓ | heuristic |
| V3 | Filter + Heuristic AWFA | ✓ | ✓ | heuristic |
| V4 | Filter + Real AWFA (TF-IDF + Jaccard) | ✓ | ✓ | real_awfa |
| V5 | Filter + Heuristic AWFA + OCR ON | ✓ | ✓ | heuristic |
| V6 | Filter + Heuristic AWFA + OCR OFF | ✓ | ✓ | heuristic |
| V7+ | BERT variants (if weights available) | ✓ | ✓ | bert_* |

## Dataset

### Synthetic Documents
- **50 documents** across 4 formats: PDF (text), PDF (scanned-image), DOCX, CSV
- Each contains realistic ESG disclosures with Environmental, Social, and Governance metrics
- Noise content (financial data, operational tables, disclaimers) interspersed
- Ground truth JSON per document with exact metric values and source sentences

### Real Documents (Optional)
Drop real ESG documents into `benchmarks/dataset/real_docs/`. These run as unlabeled experiments (no GT scoring), measuring:
- JSON/schema validity rates
- Groundedness rates
- Latency metrics

## Evaluation Metrics

### Extraction Quality
- **Strict F1**: Exact match on (name, value, unit, year)
- **Relaxed F1**: Case-insensitive, whitespace-normalized, numeric normalization

### Groundedness
- **Grounded metric rate**: Fraction of predicted metrics traceable to source text
- **Narrative grounded rate**: N-gram overlap between narratives and extracted text

### Evidence Quality (AWFA Contribution)
- **Evidence hit rate**: GT source lines found in evidence spans
- **Recall@K** (K=10, 30, 60): Evidence coverage at different cutoffs
- **Dedup rate**: Percentage removed by deduplication
- **Compression ratio**: Evidence chars / extracted chars

### Statistical Rigor
- **Bootstrap 95% CI** for all key metrics
- **Paired comparisons** between V0, V3, V4

## Output Files

| File | Description |
|------|------------|
| `benchmarks/results/axiomesg_benchmark_runs.csv` | Canonical results (one row per run) |
| `benchmarks/results/REPORT.md` | Summary report with tables and findings |
| `benchmarks/artifacts/<run_id>/output.json` | Per-run ESG output JSON |
| `benchmarks/dataset/manifest.json` | Dataset description |

## Running Tests

```bash
cd /Volumes/ReserveDisk/codeBase/AxiomESG
python -m pytest benchmarks/tests/ -v
```

## Configuration

Edit `benchmarks/config/benchmark.yaml` to control:
- Number of synthetic documents
- Random seed
- Variant definitions
- Evaluation thresholds
- Resolution plan parameters

## LLM Modes

- **With API keys** (env vars set): Uses real LLM for intelligence stage
- **Without API keys** (default): Uses deterministic mock LLM that extracts metrics via regex from evidence spans — still produces valid benchmarks for all non-LLM metrics

## BERT Handling

BERT-based variants (V7-V10) require model weights in:
```
backend/app/bert_esg_classifier/content/bert_esg_classifier_v2/
```
If weights are absent, those variants are automatically marked `UNAVAILABLE` and skipped. The harness auto-scales document count or replicas to maintain 500+ scored runs.
