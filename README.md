# AxiomESG

AxiomESG is a deterministic intelligence layer that ingests unstructured documents (PDF/DOCX/XLSX/CSV/PPTX/images), extracts ESG-relevant evidence, converts it to a canonical ESG JSON, validates it, and returns the final payload for downstream systems.

This repository contains:
- `backend/` — FastAPI service with a hardened, modular pipeline
- `frontend/` — Next.js App Router UI for ingestion, preview, and JSON output
- `OldFiles_esg-ai-pipeline/` — legacy pipeline retained for reference

## Business Use Cases

- **ESG reporting automation**: accelerate annual/quarterly ESG report prep by turning unstructured documents into structured ESG data.
- **Portfolio diligence**: standardize ESG evidence extraction across target companies during due diligence.
- **Regulatory readiness**: build audit trails by keeping evidence spans aligned to output metrics.
- **Supplier ESG monitoring**: extract ESG claims and metrics from supplier disclosures and scorecards.
- **Sustainability data platforms**: feed structured ESG JSON into data lakes or analytics tools.

## System Overview

AxiomESG runs a deterministic, stage-driven pipeline:

1) **UPLOAD** — document intake and size validation  
2) **EXTRACT** — text extraction (OCR optional)  
3) **FILTER** — ESG sentence filtering (E/S/G keywords)  
4) **WEIGHT** — AWFA weighting + deterministic dedup  
5) **INTELLIGENCE** — single LLM call to standardize ESG JSON  
6) **VALIDATE** — Pydantic v2 schema validation  
7) **OUTPUT** — return strict JSON and preview text

### Data Flow

Documents → Text Extraction → ESG Sentence Filter → AWFA → Evidence Spans → LLM → Pydantic → ESG JSON

### Output Contract (Canonical JSON)

The output strictly conforms to `ESGOutput`:
- `metadata` — source files, model info, ISO8601 timestamp, AWFA flag
- `aggregation` — counts, OCR usage, totals
- `environmental/social/governance` sections — narrative, metrics, confidence score, top evidence spans

This enables downstream systems to rely on stable, predictable structure.

## Technical Aspects

### Backend (FastAPI)

- **Python 3.11+**
- **FastAPI** with async endpoints
- **Pydantic v2** schema validation
- **Uvicorn** with reload for dev
- **In-memory job store** (optional Redis if `REDIS_URL` is set)
- **No document persistence** by default (in-memory processing)

### Pipeline Modules

- `extractor.py` — multi-format extraction
  - PDF (pypdf), DOCX (python-docx), PPTX (python-pptx)
  - CSV (utf-8 text), XLSX (openpyxl to CSV)
  - Images via OCR if configured
- `ocr_azure.py` — Azure Document Intelligence (prebuilt-read), retried with backoff
- `esg_filter.py` — configurable keyword lists for E/S/G
- `awfa.py` — deterministic weighting + dedup
- `llm/` — provider adapters (OpenRouter, Azure OpenAI, Gemini)
- `schema.py` — canonical ESG output model
- `orchestrator.py` — pipeline coordination + logging

### LLM Provider Adapters

Controlled by `LLM_PROVIDER`:
- `openrouter` — OpenAI-compatible
- `azure_openai` — OpenAI-compatible with Azure deployment
- `gemini` — Google GenAI REST

Each adapter:
- enforces strict JSON output
- retries once on transient errors
- returns usage where available

### Prompt Hardening

- extracted text treated as data only
- explicit instruction to ignore embedded document prompts
- strict JSON output (no markdown)
- one repair pass if JSON invalid

### Observability

- job ID is included in logs
- stage timing is logged (extract/filter/weight/LLM/validate)
- token usage logged where provided
- secrets never logged

## UX & UI Notes

The UI is monochrome, sharp, and minimal:
- **White-first** layout with black hairlines and accents
- 12-column grid, large whitespace
- No pills or rounded elements
- Calm 300ms ease-out transitions
- Accessibility: focus outlines and ARIA labels

## Run (Local Dev)

### Docker
```bash
docker compose up
```

### Manual
```bash
make backend
make frontend
```

Backend runs on `http://localhost:8000`, frontend on `http://localhost:3000`.

## Environment Setup

- `backend/.env.example` → copy to `backend/.env`
- `frontend/.env.example` → copy to `frontend/.env.local`

Only placeholders are provided. Do not commit real secrets.

### Required (choose one LLM provider)

**Azure OpenAI (recommended):**
```
LLM_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=...
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

**OpenRouter:**
```
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openrouter/auto
```

**Gemini:**
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
```

### Optional OCR

```
AZURE_DOCINTEL_ENDPOINT=...
AZURE_DOCINTEL_KEY=...
```

## API Reference

### GET `/`
Returns:
```
{ "status": "ok", "service": "AxiomESG" }
```

### POST `/api/extract`
Multipart upload, returns:
```
{ "job_id": "...", "status": "queued" }
```

### GET `/api/jobs/{job_id}`
Returns:
```
{
  "job_id": "...",
  "status": "queued|running|done|error",
  "stage": "UPLOAD|EXTRACT|FILTER|WEIGHT|INTELLIGENCE|VALIDATE|OUTPUT",
  "progress": 0-100,
  "source_files": [...],
  "raw_text_preview": "first N chars ...",
  "result": { ESG JSON } | null,
  "error": { "message": "...", "detail": "..."} | null
}
```

## Reliability & Safety

- File size limits enforced (per-file + total)
- OCR retries with exponential backoff
- LLM retry once on transient errors
- Strict JSON output + one repair pass
- No document persistence by default
- CORS limited to configured origins

## Module Map

```
backend/
  app/
    api/          FastAPI routes + job store
    core/         settings + logging
    pipeline/     extract → filter → AWFA → LLM → validate
  tests/          minimal pytest coverage

frontend/
  app/            Next.js App Router shell + main screen
  components/     UI components (dropzone, stepper, preview, JSON)
```

## Notes

- Legacy pipeline lives in `OldFiles_esg-ai-pipeline/` and remains untouched.
- Optional OCR uses Azure Document Intelligence when configured.
