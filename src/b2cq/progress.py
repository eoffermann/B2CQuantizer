"""Async pub/sub bus for job progress events. Per-job channels, in-memory."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator


class ProgressBus:
    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, job_id: str, event: dict) -> None:
        async with self._lock:
            queues = list(self._queues.get(job_id, []))
        for q in queues:
            await q.put(event)

    async def subscribe(self, job_id: str) -> AsyncIterator[dict]:
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._queues.setdefault(job_id, []).append(q)
        try:
            while True:
                evt = await q.get()
                yield evt
        finally:
            async with self._lock:
                self._queues[job_id].remove(q)
