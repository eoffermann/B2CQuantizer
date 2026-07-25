"""Job / quant-result data model + in-memory JobStore."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field

from b2cq.calibration import CalibrationSource


class QuantStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    UPLOADING = "uploading"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class QuantResult(BaseModel):
    quant_id: str
    status: QuantStatus
    lane: Literal["A", "B"]
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None
    output_size_bytes: Optional[int] = None
    error: Optional[str] = None
    upload_url: Optional[str] = None
    repo_id: Optional[str] = None
    log_tail: list[str] = Field(default_factory=list)  # bounded to 100 by writers


class Job(BaseModel):
    id: str
    source_model: str
    owner: str
    quants: list[QuantResult]
    calibration: CalibrationSource
    private: bool
    update_source_readme: bool
    started_at: datetime
    finished_at: Optional[datetime] = None
    status: Literal["pending", "running", "complete", "failed"] = "pending"


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, *, source_model: str, owner: str, quants: list[QuantResult],
               calibration: CalibrationSource, private: bool,
               update_source_readme: bool) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            source_model=source_model,
            owner=owner,
            quants=quants,
            calibration=calibration,
            private=private,
            update_source_readme=update_source_readme,
            started_at=datetime.now(timezone.utc),
        )
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job:
        return self._jobs[job_id]

    def list(self) -> list[Job]:
        return list(self._jobs.values())
