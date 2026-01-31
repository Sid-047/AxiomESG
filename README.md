# AxiomESG

AxiomESG is a deterministic intelligence layer that ingests unstructured documents, extracts ESG-relevant evidence, and returns validated ESG JSON for downstream systems.

## Run (Local Dev)

### Docker
```bash
docker-compose up
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

Only placeholders are provided. Never place real secrets in the repo.

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
