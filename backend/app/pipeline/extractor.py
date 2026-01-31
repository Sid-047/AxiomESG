from __future__ import annotations

import io
from typing import Dict, List, Tuple

import pandas as pd
from docx import Document
from PIL import Image
from pypdf import PdfReader
from pptx import Presentation

from app.core.config import Settings
from app.pipeline.ocr_azure import azure_read_document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".pptx", ".png", ".jpg", ".jpeg"}


def _extension(filename: str) -> str:
    dot = filename.lower().rfind(".")
    return filename[dot:] if dot >= 0 else ""


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(t.strip() for t in texts if t.strip())


def _extract_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def _extract_pptx(data: bytes) -> str:
    prs = Presentation(io.BytesIO(data))
    texts: List[str] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text = shape.text.strip()
                if text:
                    texts.append(text)
    return "\n".join(texts)


def _extract_csv(data: bytes) -> str:
    df = pd.read_csv(io.BytesIO(data))
    return df.to_csv(index=False)


def _extract_xlsx(data: bytes) -> str:
    df = pd.read_excel(io.BytesIO(data))
    return df.to_csv(index=False)


def _extract_image(data: bytes) -> None:
    Image.open(io.BytesIO(data))


def extract_documents(
    files: List[Tuple[str, bytes, str | None]], settings: Settings
) -> Tuple[Dict[str, str], bool]:
    texts: Dict[str, str] = {}
    ocr_used = False
    for filename, data, content_type in files:
        ext = _extension(filename)
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {filename}")

        if ext == ".pdf":
            extracted = _extract_pdf(data)
            if len(extracted.strip()) < 200 and settings.azure_docintel_endpoint and settings.azure_docintel_key:
                extracted = azure_read_document(data, content_type or "application/pdf", settings)
                ocr_used = True
        elif ext == ".docx":
            extracted = _extract_docx(data)
        elif ext == ".pptx":
            extracted = _extract_pptx(data)
        elif ext == ".csv":
            extracted = _extract_csv(data)
        elif ext == ".xlsx":
            extracted = _extract_xlsx(data)
        else:
            _extract_image(data)
            if settings.azure_docintel_endpoint and settings.azure_docintel_key:
                extracted = azure_read_document(data, content_type or "image/png", settings)
                ocr_used = True
            else:
                raise ValueError("OCR not configured for image extraction.")
        texts[filename] = extracted
    return texts, ocr_used
