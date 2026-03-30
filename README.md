# AxiomESG: Deterministic ESG Intelligence

**AxiomESG** is an advanced, deterministic intelligence layer designed to autonomously ingest unstructured corporate documents (PDFs, DOCX, XLSX, PPTX, Images) and distill them into highly structured, canonical ESG (Environmental, Social, Governance) data. 

Unlike raw LLM wrappers that suffer from hallucinations, undefined schemas, and token limit crashes, AxiomESG employs a multi-stage **Pipeline Architecture** with an embedded **Algorithm Strategy Registry** (featuring Heuristic AWFA and BERT-based semantic fusion) to pre-process, filter, and weight evidence *before* a final, constrained LLM extraction pass generates strict Pydantic-validated JSON and human-readable Markdown reports.

---

## 🎯 Business Use Cases

- **Automated ESG Reporting**: Accelerate the preparation of annual/quarterly ESG reports by transforming thousands of pages of unstructured disclosures into actionable structured metrics.
- **Portfolio Due Diligence**: Standardize the extraction of ESG evidence across target companies to build a level playing field for comparisons.
- **Supplier ESG Monitoring**: Parse supplier audits, scorecards, and questionnaires to extract verified claims and flag labor/environmental risks.
- **Regulatory Readiness & Audit Trails**: Every metric extracted is tethered directly to the exact source file and text span, satisfying CSRD and TCFD transparency requirements.

---

## 🧠 The Intelligence Pipeline

AxiomESG runs a deterministic, deeply observable 7-stage pipeline:

1. **UPLOAD (`INTAKE`)**
   Takes in raw documents (Multipart upload). Enforces file size limits (25MB per file, 50MB total).
2. **EXTRACT (`MULTI-FORMAT PARSING`)**
   Converts PDFs, DOCX, CSV/XLSX, and PPTX into normalized text. Optionally falls back to **Azure Document Intelligence** for OCR on images and scanned PDFs with exponential backoff.
3. **FILTER (`SIGNAL SEPARATION`)**
   Splits text into sentences and applies hyper-optimized keyword dictionaries to classify sentences into E, S, or G buckets.
4. **WEIGHT (`ALGORITHMIC STRATEGY`)**
   Evaluates the filtered sentences using a selectable strategy (Heuristic AWFA, BERT Fusion, etc.), assigns a relevance weight (0.0 to 1.0), sorts them, and deterministically deduplicates similar semantic blocks.
5. **INTELLIGENCE (`CONSTRICTED LLM EXTRACTION`)**
   Passes only the highest-weighted evidence spans into a selected LLM (Azure OpenAI, Gemini, or OpenRouter). The LLM is strictly constrained via its system prompt to output pure schema-compliant JSON, without fabricating metrics or normalizing units. 
6. **VALIDATE (`SCHEMA ENFORCEMENT`)**
   Validates the LLM output against a strict `Pydantic v2` schema tree. If validation fails, an automatic single-pass "repair" prompt is triggered internally to fix the JSON casing/keys.
7. **OUTPUT (`FINAL MANIFEST`)**
   Returns the canonical ESG JSON, detailed processing metadata, aggregation stats, and raw text preview strings back to the client.

---

## 🧬 Algorithm Strategy Layer

A core mechanic of AxiomESG is its extensible **Strategy Registry** (`app/pipeline/strategies.py`). The frontend UI provides a toggle to dispatch jobs using one of several distinct weighting algorithms:

- **`Heuristic AWFA` (Default)**
  An enhanced execution of the **Axiom Weighting & Filtering Algorithm**. It scores blocks purely based on deterministic factors: Keyword Density, Normalized Sentence Length, Numerical Density (presence of %, $, dates, digits), and Category specific weights. Ultra-fast and uses no GPU memory.
  
- **`BERT + Mean Fusion`**
  A hybrid approach. The system lazy-loads a fine-tuned HuggingFace `FinBERT` ESG sequence-classifier to generate probability distributions for E/S/G classifications. It then statically fuses the BERT probability tensors with the base Heuristic AWFA score using a balanced 50/50 mean fusion strategy. Great for heavily nuanced, dense prose.

- **`BERT + Static Fusion`**
  Applies a harder static multiplier. Heavily penalizes text that AWFA thinks is relevant but the BERT neural network classifies with low ESG probability. 

- **`BERT + AWFA v1 & v2`**
  Legacy algorithmic permutations retained for analytical continuity and backwards-compatible testing.

---

## 🏗 System Architecture

The repository is modularly decoupled into two main environments:

### 1. The Backend (`FastAPI`)
- **Core Engine**: Python 3.11-slim, robust async endpoints (`uvicorn`).
- **Pydantic v2**: Handles `ESGOutput`, `ESGSection`, `Metric`, and `EvidenceSpan` validation.
- **In-Memory Job Store**: Background tasks allow for asynchronous polling, providing a fast ingestion UI even for massive 100-page PDFs. (Redis integration available).
- **Report Generator**: (`app/pipeline/report.py`) Dynamically translates the 3-dimensional ESG JSON graph into a clean, executive-ready Markdown Report.

### 2. The Frontend (`Next.js 14 App Router`)
- **React / TypeScript / TailwindCSS**
- **Brand UI/UX**: AxiomESG uses a strict, hyper-minimalist monochrome design system. White backgrounds, 1px black hairlines, pure semantic HTML layout grids, and stark typographic hierarchy. No rounded corners. No gradients.
- **Key Components**:
  - `AlgorithmSelector`: Real-time strategy switching UI.
  - `Stepper`: Real-time WebSocket-like (via async polling) progress indicators for the 7-stage engine.
  - `ReportPane`: A tabbed toggle bridging the raw JSON Tree view and the generated Executive Markdown Report view.

---

## 🐳 Docker & The `start.sh` Launcher

The entire ecosystem is containerized for instant, consistent deployment across any environment. We provide a single-shot entrypoint script.

### Launching the Stack

To build the images and boot the system in the background, simply run:
```bash
./start.sh
```

### The `start.sh` CLI Interface

If you need precise control, the script operates as a robust CLI:
```bash
./start.sh up          # Build & start all services (detached)
./start.sh up:dev      # Build & start in foreground (with hot-reloading logs)
./start.sh down        # Stop & remove all containers
./start.sh restart     # Restart all services
./start.sh logs        # Tail logs from all services
./start.sh logs:back   # Tail logs exclusively from the FastAPI backend
./start.sh logs:front  # Tail logs exclusively from the Next.js frontend
./start.sh status      # Show container status
./start.sh shell:back  # Open an interactive /bin/bash shell in the backend
./start.sh test        # Run the backend pytest validation suite
./start.sh health      # Ping backend health endpoints
./start.sh algorithms  # List available ESG algorithms directly from the API
./start.sh clean       # Nuke containers, images & volumes
```

---

## ⚙️ Environment Construction

When evaluating `start.sh` for the first time, it will autonomously generate a placeholder `backend/.env` file if one does not exist. 

### Required Variables
Choose one LLM reasoning engine to attach to the pipeline:

**Azure OpenAI (Recommended for Speed):**
```env
LLM_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

**OpenRouter:**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

**Optional Modules:**
If parsing image-heavy or scanned PDFs, you should enable Azure Document Intelligence OCR:
```env
AZURE_DOCINTEL_ENDPOINT=https://...
AZURE_DOCINTEL_KEY=your_key
```

---

## 🔌 API Reference

### `GET /`
Standard Healthcheck ping.

### `GET /api/algorithms`
Returns a strict mapping of the Strategy Registry keys and their UI display names.
```json
{
  "heuristic": "Heuristic AWFA",
  "bert_mean": "BERT + Mean Fusion",
  ...
}
```

### `POST /api/extract` & `POST /api/extract_sync`
Accepts `multipart/form-data` file uploads. Accepts an optional `algorithm` query parameter. Initiates the pipeline.

### `GET /api/jobs/{job_id}`
Returns the real-time pipeline status of a specific job ticket.
```json
{
  "job_id": "893c8d-...",
  "status": "running",
  "stage": "WEIGHT",
  "progress": 55,
  "result": null
}
```

### `GET /api/jobs/{job_id}/report`
Returns a structured Markdown representation of the completed ESG JSON.

---

## 🛡 Reliability & Hardening

- **Memory Constraints**: File limits strictly enforced at the HTTP middleware layer.
- **Fail-Safes**: Missing internal model weights for BERT automatically downgrade gracefully or halt predictably using custom exceptions.
- **Transient Recovery**: Network timeouts during LLM API interactions and Azure OCR interactions trigger exponential backoff retry loops.
- **Idempotency**: The application strictly separates storage state from pipeline processing logic.

---

## 🧪 Benchmarking & Ablation Harness

AxiomESG includes a research-paper-ready benchmarking system for evaluating pipeline variants.

### Quick Start

```bash
# 1. Generate synthetic ESG dataset (50 docs with ground truth)
python -m benchmarks.src.generate_synthetic_dataset --out benchmarks/dataset --n 50 --seed 42

# 2. Run full experiment matrix (500+ runs)
python -m benchmarks.src.run_benchmarks --config benchmarks/config/benchmark.yaml

# 3. Generate summary report
python -m benchmarks.src.report --csv benchmarks/results/axiomesg_benchmark_runs.csv

# 4. Run with auto-augmentation and report
python -m benchmarks.src.run_benchmarks --config benchmarks/config/benchmark.yaml --augment --report

# 5. Run tests
python -m pytest benchmarks/tests/ -v
```

### Outputs
- `benchmarks/results/axiomesg_benchmark_runs.csv` — Canonical results CSV
- `benchmarks/results/REPORT.md` — Summary report with tables, CIs, and key findings

See [`benchmarks/README.md`](benchmarks/README.md) for full documentation.

---
*Legacy Note: Previous engine files have been sequestered securely into `OldFiles_esg-ai-pipeline/` for historical archival and pattern analysis. The current `backend/` engine supersedes all legacy operations.*
