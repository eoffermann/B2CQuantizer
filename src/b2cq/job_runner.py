"""Two-lane job orchestrator: safetensors (GPU) + GGUF (CPU) concurrently.

Guarantees:
- Source downloads exactly once, before either lane starts.
- BF16 GGUF conversion happens once per job, shared across all GGUF quants.
- imatrix computed once per job if any I-quant is selected.
- Model loaded on GPU once per job, held across all safetensors quants.
- Per-quant failure marks that quant FAILED and continues with siblings.
- Per-lane setup failure marks all downstream quants in that lane SKIPPED.
- Token wiped from hf_client on completion regardless of success/failure.
"""
from __future__ import annotations

import asyncio
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from b2cq.calibration import to_plaintext
from b2cq.hf_client import HFClient
from b2cq.job_model import Job, QuantStatus
from b2cq.progress import ProgressBus
from b2cq.quant_catalog import get as get_quant, QuantFamily
from b2cq.workers.safetensors import load_model_for_safetensors, quantize_safetensors
from b2cq.workers.gguf_convert import convert_to_bf16_gguf
from b2cq.workers.gguf_quantize import gguf_quantize, compute_imatrix
from b2cq.workers.mmproj import export_mmproj, is_multimodal
from b2cq.readme_updater import update_source_readme


def _mk_log_cb(job_id: str, quant_id: str, progress: ProgressBus, tail: list[str]):
    """Bounded log-tail + publish each line to the progress bus."""
    loop = asyncio.get_running_loop()

    def cb(line: str) -> None:
        tail.append(line)
        if len(tail) > 100:
            del tail[:-100]
        asyncio.run_coroutine_threadsafe(
            progress.publish(job_id, {"type": "log", "quant": quant_id, "line": line}),
            loop,
        )

    return cb


async def _run_lane_a(job: Job, source_dir: Path, calibration, hf_client, progress, workdir):
    """safetensors lane: load model once, run each selected safetensors quant serially."""
    safetensors_quants = [q for q in job.quants if get_quant(q.quant_id).family == QuantFamily.SAFETENSORS]
    if not safetensors_quants:
        return

    # Load model once
    try:
        setup_tail: list[str] = []
        setup_cb = _mk_log_cb(job.id, "__lane_a_setup__", progress, setup_tail)
        model, tokenizer = await asyncio.to_thread(load_model_for_safetensors, source_dir, setup_cb)
    except Exception as e:
        for q in safetensors_quants:
            q.status = QuantStatus.SKIPPED
            q.error = f"Lane A setup failed: {e}"
        await progress.publish(job.id, {"type": "lane_failed", "lane": "A", "error": str(e)})
        return

    for q in safetensors_quants:
        await _run_one_safetensors_quant(job, q, model, tokenizer, source_dir, calibration, hf_client, progress, workdir)


async def _run_one_safetensors_quant(job, q, model, tokenizer, source_dir, calibration, hf_client, progress, workdir):
    q.status = QuantStatus.RUNNING
    q.started_at = datetime.now(timezone.utc)
    t0 = time.time()
    await progress.publish(job.id, {"type": "status", "quant": q.quant_id, "status": q.status})
    tail = q.log_tail
    log_cb = _mk_log_cb(job.id, q.quant_id, progress, tail)

    try:
        output_dir = workdir / f"safetensors_{q.quant_id}"
        await asyncio.to_thread(
            quantize_safetensors,
            model, tokenizer, get_quant(q.quant_id).format, calibration, output_dir, log_cb,
            source_dir=source_dir,
        )
        q.status = QuantStatus.UPLOADING
        await progress.publish(job.id, {"type": "status", "quant": q.quant_id, "status": q.status})

        default_repo_id = f"{job.owner}/{Path(job.source_model).name}-{q.quant_id}"
        repo_id = q.repo_id or default_repo_id
        url = await asyncio.to_thread(
            hf_client.upload_folder, repo_id, output_dir,
            create_if_missing=True, private=job.private,
            commit_message=f"B2CQuantizer: {q.quant_id}",
        )
        q.repo_id = repo_id
        q.upload_url = url
        q.output_size_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        q.status = QuantStatus.DONE
        # cleanup local
        shutil.rmtree(output_dir, ignore_errors=True)
    except Exception as e:
        q.status = QuantStatus.FAILED
        q.error = f"{type(e).__name__}: {e}"
    finally:
        q.finished_at = datetime.now(timezone.utc)
        q.elapsed_seconds = time.time() - t0
        await progress.publish(job.id, {"type": "status", "quant": q.quant_id, "status": q.status,
                                        "elapsed": q.elapsed_seconds, "error": q.error})


async def _run_lane_b(job: Job, source_dir: Path, calibration, hf_client, progress, workdir):
    """GGUF lane: BF16 GGUF once, imatrix once if needed, each variant serially."""
    gguf_quants = [q for q in job.quants if get_quant(q.quant_id).family in (
        QuantFamily.GGUF_K, QuantFamily.GGUF_I, QuantFamily.GGUF_MISC, QuantFamily.GGUF_MMPROJ)]
    if not gguf_quants:
        return

    # BF16 GGUF intermediate
    bf16_gguf = workdir / f"{Path(job.source_model).name}-bf16.gguf"
    try:
        setup_tail: list[str] = []
        setup_cb = _mk_log_cb(job.id, "__lane_b_setup__", progress, setup_tail)
        await asyncio.to_thread(convert_to_bf16_gguf, source_dir, bf16_gguf, setup_cb)
    except Exception as e:
        for q in gguf_quants:
            q.status = QuantStatus.SKIPPED
            q.error = f"Lane B setup failed: {e}"
        await progress.publish(job.id, {"type": "lane_failed", "lane": "B", "error": str(e)})
        return

    # imatrix if any I-quant
    imatrix_path: Path | None = None
    if any(get_quant(q.quant_id).family == QuantFamily.GGUF_I for q in gguf_quants):
        try:
            imatrix_tail: list[str] = []
            imatrix_cb = _mk_log_cb(job.id, "__imatrix__", progress, imatrix_tail)
            cal_text = to_plaintext(calibration, workdir / "cal.txt")
            imatrix_path = workdir / "imatrix.dat"
            await asyncio.to_thread(compute_imatrix, bf16_gguf, cal_text, imatrix_path, imatrix_cb)
        except Exception as e:
            for q in gguf_quants:
                if get_quant(q.quant_id).family == QuantFamily.GGUF_I:
                    q.status = QuantStatus.SKIPPED
                    q.error = f"imatrix failed: {e}"

    # For each GGUF quant, produce and upload
    default_repo_id = f"{job.owner}/{Path(job.source_model).name}-GGUF"
    try:
        for q in gguf_quants:
            if q.status == QuantStatus.SKIPPED:  # from imatrix failure
                continue
            await _run_one_gguf_quant(job, q, bf16_gguf, imatrix_path, source_dir, default_repo_id,
                                        hf_client, progress, workdir)
    finally:
        # cleanup BF16 intermediate -- must run even if the loop above escapes
        # with an unexpected exception, so the large intermediate file never
        # lingers on disk.
        bf16_gguf.unlink(missing_ok=True)


async def _run_one_gguf_quant(job, q, bf16_gguf, imatrix_path, source_dir, default_repo_id,
                                hf_client, progress, workdir):
    q.status = QuantStatus.RUNNING
    q.started_at = datetime.now(timezone.utc)
    t0 = time.time()
    await progress.publish(job.id, {"type": "status", "quant": q.quant_id, "status": q.status})
    tail = q.log_tail
    log_cb = _mk_log_cb(job.id, q.quant_id, progress, tail)
    spec = get_quant(q.quant_id)

    try:
        if spec.family == QuantFamily.GGUF_MMPROJ:
            if not is_multimodal(source_dir):
                q.status = QuantStatus.SKIPPED
                q.error = "source is not multimodal; mmproj export skipped"
                return
            out_gguf = workdir / f"{Path(job.source_model).name}-mmproj-f16.gguf"
            await asyncio.to_thread(export_mmproj, source_dir, out_gguf, log_cb)
        else:
            out_gguf = workdir / f"{Path(job.source_model).name}-{q.quant_id}.gguf"
            await asyncio.to_thread(gguf_quantize, bf16_gguf, out_gguf, spec.format, log_cb,
                                    imatrix_path if spec.needs_imatrix else None)

        q.status = QuantStatus.UPLOADING
        await progress.publish(job.id, {"type": "status", "quant": q.quant_id, "status": q.status})
        repo_id = q.repo_id or default_repo_id
        url = await asyncio.to_thread(hf_client.upload_file, repo_id, out_gguf, out_gguf.name,
                                        create_if_missing=True, private=job.private,
                                        commit_message=f"B2CQuantizer: {q.quant_id}")
        q.upload_url = url
        q.repo_id = repo_id
        q.output_size_bytes = out_gguf.stat().st_size
        q.status = QuantStatus.DONE
        out_gguf.unlink(missing_ok=True)
    except Exception as e:
        q.status = QuantStatus.FAILED
        q.error = f"{type(e).__name__}: {e}"
    finally:
        q.finished_at = datetime.now(timezone.utc)
        q.elapsed_seconds = time.time() - t0
        await progress.publish(job.id, {"type": "status", "quant": q.quant_id, "status": q.status,
                                        "elapsed": q.elapsed_seconds, "error": q.error})


async def run_job(job: Job, hf_client: HFClient, calibration: list[dict],
                  progress: ProgressBus, workdir: Path) -> None:
    """Orchestrate a full job: download source, run both lanes concurrently,
    update the source README, then always wipe the HF token.

    Both lanes are awaited with `return_exceptions=True`, so an unexpected
    exception escaping one lane never leaves the sibling lane running
    detached while `hf_client.close()` fires underneath it. Only after both
    lanes have genuinely finished do we check for an escaped exception; if
    one occurred, the job is marked failed and the README-update phase is
    skipped.

    Unexpected exceptions escaping the lane/README phase are caught,
    published as `job_failed`, and swallowed (never re-raised) -- this
    coroutine is expected to run as a background task, where an unhandled
    exception would otherwise vanish silently. `hf_client.close()` is
    guaranteed to run exactly once, on every exit path, via the outer
    try/finally.
    """
    job.status = "running"
    await progress.publish(job.id, {"type": "job_started", "job_id": job.id})

    source_dir = workdir / "source"
    try:
        try:
            source_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(hf_client.download_snapshot, job.source_model, source_dir)
        except Exception as e:
            job.status = "failed"
            job.finished_at = datetime.now(timezone.utc)
            await progress.publish(job.id, {"type": "job_failed", "error": str(e)})
            return

        try:
            # Run both lanes concurrently. return_exceptions=True is required:
            # without it, an unexpected exception escaping one lane would
            # propagate immediately while the sibling lane keeps running
            # detached, and the outer finally's hf_client.close() would then
            # wipe the token out from under the still-running lane mid-upload.
            # With return_exceptions=True, gather waits for BOTH lanes to
            # genuinely finish (success or exception) before we inspect the
            # results below.
            lane_results = await asyncio.gather(
                _run_lane_a(job, source_dir, calibration, hf_client, progress, workdir),
                _run_lane_b(job, source_dir, calibration, hf_client, progress, workdir),
                return_exceptions=True,
            )

            lane_names = ("A", "B")
            lane_errors = [
                (lane_names[i], result)
                for i, result in enumerate(lane_results)
                if isinstance(result, BaseException)
            ]
            if lane_errors:
                error_msg = "; ".join(
                    f"lane {name}: {type(err).__name__}: {err}" for name, err in lane_errors
                )
                job.status = "failed"
                job.finished_at = datetime.now(timezone.utc)
                await progress.publish(job.id, {"type": "job_failed", "error": error_msg})
                return

            # README update
            if job.update_source_readme:
                try:
                    await asyncio.to_thread(update_source_readme, job, hf_client)
                except Exception as e:
                    await progress.publish(job.id, {"type": "readme_failed", "error": str(e)})

            job.status = "complete"
            job.finished_at = datetime.now(timezone.utc)
            await progress.publish(job.id, {"type": "job_complete", "job_id": job.id})
        except Exception as e:
            job.status = "failed"
            job.finished_at = datetime.now(timezone.utc)
            await progress.publish(job.id, {"type": "job_failed", "error": str(e)})
    finally:
        hf_client.close()
