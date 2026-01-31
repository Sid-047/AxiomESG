from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.logging import get_logger
from app.api.job_store import JobRecord, get_job_store
from app.pipeline.orchestrator import run_pipeline


router = APIRouter()
logger = get_logger("api")


async def _store_set(store, job: JobRecord) -> None:
    if hasattr(store, "set") and asyncio.iscoroutinefunction(store.set):
        await store.set(job)
    else:
        store.set(job)


async def _store_get(store, job_id: str):
    if hasattr(store, "get") and asyncio.iscoroutinefunction(store.get):
        return await store.get(job_id)
    return store.get(job_id)


@router.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "AxiomESG"}


@router.post("/api/extract")
async def extract(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    settings = get_settings()
    store = get_job_store()

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    total_bytes = 0
    buffers: List[Tuple[str, bytes, str | None]] = []
    for f in files:
        data = await f.read()
        size = len(data)
        if size > settings.max_file_bytes():
            raise HTTPException(status_code=413, detail=f"{f.filename} exceeds max file size.")
        total_bytes += size
        buffers.append((f.filename, data, f.content_type))

    if total_bytes > settings.max_total_bytes():
        raise HTTPException(status_code=413, detail="Total upload exceeds max size.")

    job_id = str(uuid.uuid4())
    record = JobRecord(
        job_id=job_id,
        status="queued",
        stage="UPLOAD",
        progress=5,
        source_files=[b[0] for b in buffers],
    )
    await _store_set(store, record)

    async def run_job() -> None:
        try:
            record.status = "running"
            record.stage = "EXTRACT"
            record.progress = 20
            await _store_set(store, record)

            def stage_update(stage: str, progress: int) -> None:
                record.stage = stage
                record.progress = progress
                asyncio.create_task(_store_set(store, record))

            output, raw_text, usage = run_pipeline(buffers, settings, job_id, stage_update)

            record.stage = "OUTPUT"
            record.progress = 100
            record.status = "done"
            record.raw_text_preview = raw_text[: settings.preview_chars]
            record.result = output.model_dump()
            record.error = None
            await _store_set(store, record)
        except Exception as exc:
            record.status = "error"
            record.stage = "OUTPUT"
            record.progress = 100
            record.error = {"message": "Pipeline failed.", "detail": str(exc)}
            await _store_set(store, record)
            logger.error("job_failed", extra={"job_id": job_id, "error": str(exc)})

    asyncio.create_task(run_job())
    return {"job_id": job_id, "status": "queued"}


@router.post("/api/extract_sync")
async def extract_sync(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    settings = get_settings()
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    total_bytes = 0
    buffers: List[Tuple[str, bytes, str | None]] = []
    for f in files:
        data = await f.read()
        size = len(data)
        if size > settings.max_file_bytes():
            raise HTTPException(status_code=413, detail=f"{f.filename} exceeds max file size.")
        total_bytes += size
        buffers.append((f.filename, data, f.content_type))

    if total_bytes > settings.max_total_bytes():
        raise HTTPException(status_code=413, detail="Total upload exceeds max size.")

    job_id = str(uuid.uuid4())
    output, raw_text, usage = run_pipeline(buffers, settings, job_id)
    return {
        "job_id": job_id,
        "status": "done",
        "stage": "OUTPUT",
        "progress": 100,
        "source_files": [b[0] for b in buffers],
        "raw_text_preview": raw_text[: settings.preview_chars],
        "result": output.model_dump(),
        "error": None,
    }


@router.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> Dict[str, Any]:
    settings = get_settings()
    store = get_job_store()
    record = await _store_get(store, job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found.")
    return record.to_dict()
