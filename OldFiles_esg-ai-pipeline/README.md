# ESG AI Pipeline

Automated ESG data extraction from unstructured documents with adaptive weighted fusion and LLM-based JSON conversion.

Note: This legacy pipeline is retained for reference. The vNext implementation lives in `backend/` and `frontend/`.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file from `.env.example` and fill in your credentials:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

3. Run the backend:
```bash
uvicorn main:app --reload
```

4. Open `frontend/index.html` in a browser or serve it with a simple HTTP server.

## Features

- Multi-format document support (PDF, DOCX, XLSX, CSV, PPTX, images)
- Automatic OCR fallback for scanned documents
- ESG content filtering
- Adaptive Weighted Fusion Algorithm (AWFA)
- Single LLM call for JSON conversion
- Fixed, portable JSON schema
