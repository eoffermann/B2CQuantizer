"""FastAPI routes for setup, job submission, live job view, history."""
from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from pathlib import Path
from tempfile import mkdtemp

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from b2cq.quant_catalog import CATALOG, QuantFamily, by_family, get as get_quant
from b2cq.job_model import JobStore, QuantResult, QuantStatus
from b2cq.calibration import CalibrationSource, load_calibration
from b2cq.hf_client import HFClient
from b2cq.progress import ProgressBus
from b2cq.job_runner import run_job

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Module-level singletons for the app's lifetime.
JOB_STORE = JobStore()
PROGRESS = ProgressBus()


@lru_cache(maxsize=1)
def _blackwell_available() -> bool:
    # Hardware doesn't change mid-process, so cache the result: torch import
    # is expensive, and in environments without a working torch install
    # (e.g. this dev venv's broken DLL) attempting it repeatedly is wasteful
    # at best and can surface as a noisy OS-level fault at worst.
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 12
    except ImportError:
        return False
    except Exception:
        return False


@router.get("/", response_class=HTMLResponse)
async def setup(request: Request):
    return templates.TemplateResponse("setup.html", {
        "request": request,
        "gguf_k": by_family(QuantFamily.GGUF_K),
        "gguf_i": by_family(QuantFamily.GGUF_I),
        "gguf_misc": by_family(QuantFamily.GGUF_MISC),
        "gguf_mmproj": by_family(QuantFamily.GGUF_MMPROJ),
        "safetensors": by_family(QuantFamily.SAFETENSORS),
        "blackwell_available": _blackwell_available(),
    })


@router.post("/calibration/upload")
async def calibration_upload(calibration_file: UploadFile = File(...)):
    """HTMX upload handler; returns a token id the setup form uses."""
    workdir = Path(mkdtemp(prefix="b2cq_cal_"))
    cal_path = workdir / "cal.jsonl"
    cal_path.write_bytes(await calibration_file.read())
    return {"token": str(cal_path)}


@router.post("/jobs")
async def create_job(
    source_model: str = Form(...),
    hf_token: str = Form(...),
    calibration_type: str = Form("bundled"),
    calibration_file: UploadFile | None = File(None),
    calibration_dataset: str = Form(""),
    quants: list[str] = Form(...),
    owner: str = Form(""),
    private: bool = Form(False),
    update_source_readme: bool = Form(False),
):
    # Build calibration source
    workdir = Path(mkdtemp(prefix="b2cq_"))
    if calibration_type == "upload" and calibration_file is not None:
        cal_path = workdir / "cal.jsonl"
        cal_path.write_bytes(await calibration_file.read())
        cal_source = CalibrationSource(type="upload", local_path=cal_path)
    elif calibration_type == "hf_dataset":
        cal_source = CalibrationSource(type="hf_dataset", hf_dataset_id=calibration_dataset,
                                        hf_token=hf_token)
    else:
        cal_source = CalibrationSource(type="bundled")

    calibration = load_calibration(cal_source)

    # Resolve owner from token if not supplied
    hf_client = HFClient(token=hf_token)
    if not owner:
        owner = hf_client.whoami().get("name", "unknown")

    # Build quant results
    quant_results = []
    for qid in quants:
        spec = get_quant(qid)
        lane = "A" if spec.family == QuantFamily.SAFETENSORS else "B"
        quant_results.append(QuantResult(quant_id=qid, status=QuantStatus.PENDING, lane=lane))

    job = JOB_STORE.create(
        source_model=source_model, owner=owner, quants=quant_results,
        calibration=cal_source, private=private,
        update_source_readme=update_source_readme,
    )

    asyncio.create_task(run_job(job, hf_client, calibration, PROGRESS, workdir))
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@router.get("/jobs", response_class=HTMLResponse)
async def history(request: Request):
    return templates.TemplateResponse("history.html", {
        "request": request, "jobs": JOB_STORE.list(),
    })


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_view(request: Request, job_id: str):
    try:
        job = JOB_STORE.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return templates.TemplateResponse("job.html", {"request": request, "job": job})


@router.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    async def event_gen():
        async for evt in PROGRESS.subscribe(job_id):
            yield {"data": json.dumps(evt)}
    return EventSourceResponse(event_gen())
