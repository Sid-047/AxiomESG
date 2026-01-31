from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("job_store")


@dataclass
class JobRecord:
    job_id: str
    status: str = "queued"
    stage: str = "UPLOAD"
    progress: int = 0
    source_files: list[str] = field(default_factory=list)
    raw_text_preview: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    updated_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


class InMemoryJobStore:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, JobRecord] = {}

    def set(self, job: JobRecord) -> None:
        job.updated_at = time.time()
        self._store[job.job_id] = job

    def get(self, job_id: str) -> Optional[JobRecord]:
        job = self._store.get(job_id)
        if not job:
            return None
        if time.time() - job.updated_at > self.ttl_seconds:
            self._store.pop(job_id, None)
            return None
        return job


class RedisJobStore:
    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        import redis.asyncio as redis

        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl_seconds = ttl_seconds

    async def set(self, job: JobRecord) -> None:
        payload = json.dumps(job.to_dict())
        await self.redis.setex(job.job_id, self.ttl_seconds, payload)

    async def get(self, job_id: str) -> Optional[JobRecord]:
        payload = await self.redis.get(job_id)
        if not payload:
            return None
        data = json.loads(payload)
        return JobRecord(**data)


@lru_cache
def get_job_store():
    settings = get_settings()
    if settings.redis_url:
        try:
            return RedisJobStore(settings.redis_url, settings.job_poll_ttl_seconds)
        except Exception as exc:
            logger.warning("redis_unavailable", extra={"error": str(exc)})
    return InMemoryJobStore(settings.job_poll_ttl_seconds)
