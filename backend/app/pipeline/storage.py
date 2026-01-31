from __future__ import annotations


class StorageAdapter:
    def save(self, filename: str, data: bytes) -> str:
        raise NotImplementedError("Storage adapter not configured.")
