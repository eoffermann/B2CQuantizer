"""Two-lane job orchestrator: per-quant isolation, per-lane skip, token wipe.

All worker functions (`load_model_for_safetensors`, `quantize_safetensors`,
`convert_to_bf16_gguf`, `compute_imatrix`, `gguf_quantize`, `export_mmproj`,
`is_multimodal`) and `update_source_readme` are monkeypatched at the
job_runner module's imported-name level with lightweight fakes, so these
tests exercise only the orchestration semantics -- no torch/llm-compressor/
llama.cpp required.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import b2cq.job_runner as job_runner_module
from b2cq.calibration import CalibrationSource
from b2cq.hf_client import HFClient
from b2cq.job_model import Job, QuantResult, QuantStatus
from b2cq.job_runner import run_job
from b2cq.quant_catalog import get as get_quant, QuantFamily
from b2cq.workers.safetensors import _MULTIMODAL_ARCH, _TEXT_ARCH


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CALIBRATION = [{"messages": [{"role": "user", "content": "hello there"}]}]


class RecordingProgress:
    """Minimal stand-in for ProgressBus that just records published events."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish(self, job_id: str, event: dict) -> None:
        self.events.append((job_id, event))

    def of_type(self, event_type: str) -> list[dict]:
        return [evt for _, evt in self.events if evt.get("type") == event_type]


def _make_job(quant_ids, *, owner="acme", source_model="acme/model-7b",
              update_source_readme=False, private=False) -> Job:
    quants = []
    for qid in quant_ids:
        family = get_quant(qid).family
        lane = "A" if family == QuantFamily.SAFETENSORS else "B"
        quants.append(QuantResult(quant_id=qid, status=QuantStatus.PENDING, lane=lane))
    return Job(
        id="job1",
        source_model=source_model,
        owner=owner,
        quants=quants,
        calibration=CalibrationSource(type="bundled"),
        private=private,
        update_source_readme=update_source_readme,
        started_at=datetime.now(timezone.utc),
    )


def _make_hf_client(arch: str = "MistralForCausalLM") -> MagicMock:
    hf_client = MagicMock(spec=HFClient)
    hf_client.download_snapshot.return_value = Path("/fake/source")
    hf_client.upload_folder.return_value = "https://huggingface.co/repo/commit/abc"
    hf_client.upload_file.return_value = "https://huggingface.co/repo/blob/main/file.gguf"

    # I4: run_job now fetches config.json via download_file before downloading
    # the source snapshot, to refuse non-Mistral architectures. Return a real
    # on-disk config.json carrying `arch` so the arch guard can parse it.
    def _download_file(repo_id, filename):
        fd, path = tempfile.mkstemp(suffix="_config.json")
        p = Path(path)
        p.write_text(json.dumps({"architectures": [arch]}), encoding="utf-8")
        return p

    hf_client.download_file.side_effect = _download_file
    return hf_client


# ---- default happy-path fakes for every worker entry point ----------------

def _fake_load_model_for_safetensors(source_dir, log_cb):
    return ("MODEL", "TOKENIZER")


def _fake_quantize_safetensors(model, tokenizer, format, calibration, output_dir, source_dir, log_cb):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.safetensors").write_bytes(b"fake-weights")
    (output_dir / "config.json").write_text("{}")


def _fake_convert_to_bf16_gguf(source_dir, output_gguf, log_cb):
    output_gguf = Path(output_gguf)
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    output_gguf.write_bytes(b"bf16-fake")


def _fake_compute_imatrix(bf16_gguf, cal_text, output_imatrix, log_cb, n_chunks=100):
    Path(output_imatrix).write_bytes(b"imatrix-fake")


def _fake_gguf_quantize(bf16_gguf, output_gguf, format, log_cb, imatrix=None):
    output_gguf = Path(output_gguf)
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    output_gguf.write_bytes(b"gguf-fake")


def _fake_export_mmproj(source_dir, output_gguf, log_cb):
    Path(output_gguf).write_bytes(b"mmproj-fake")


def _fake_is_multimodal(source_dir):
    return False


def _fake_update_source_readme(job, hf_client):
    return "https://huggingface.co/user/model/commit/readme"


def _patch_workers(monkeypatch, **overrides):
    defaults = {
        "load_model_for_safetensors": _fake_load_model_for_safetensors,
        "quantize_safetensors": _fake_quantize_safetensors,
        "convert_to_bf16_gguf": _fake_convert_to_bf16_gguf,
        "compute_imatrix": _fake_compute_imatrix,
        "gguf_quantize": _fake_gguf_quantize,
        "export_mmproj": _fake_export_mmproj,
        "is_multimodal": _fake_is_multimodal,
        "update_source_readme": _fake_update_source_readme,
    }
    defaults.update(overrides)
    for name, fn in defaults.items():
        monkeypatch.setattr(f"b2cq.job_runner.{name}", fn)


def _status_map(job: Job) -> dict[str, QuantResult]:
    return {q.quant_id: q for q in job.quants}


# ---------------------------------------------------------------------------
# (a) per-quant failure isolation
# ---------------------------------------------------------------------------

async def test_per_quant_failure_isolation_lane_b(tmp_path, monkeypatch):
    job = _make_job(["Q4_K_M", "Q5_K_M"])  # both GGUF_K -> lane B

    def flaky_gguf_quantize(bf16_gguf, output_gguf, format, log_cb, imatrix=None):
        if format == "Q5_K_M":
            raise RuntimeError("quantize boom")
        output_gguf = Path(output_gguf)
        output_gguf.parent.mkdir(parents=True, exist_ok=True)
        output_gguf.write_bytes(b"data")

    _patch_workers(monkeypatch, gguf_quantize=flaky_gguf_quantize)
    hf_client = _make_hf_client()
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    statuses = _status_map(job)
    assert statuses["Q4_K_M"].status == QuantStatus.DONE
    assert statuses["Q5_K_M"].status == QuantStatus.FAILED
    assert "quantize boom" in statuses["Q5_K_M"].error
    assert job.status == "complete"
    hf_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# (b) lane-A setup failure skips lane-A quants, lane B proceeds
# ---------------------------------------------------------------------------

async def test_lane_a_setup_failure_skips_lane_a_but_lane_b_proceeds(tmp_path, monkeypatch):
    job = _make_job(["W4A16_GPTQ", "Q4_K_M"])

    def failing_load_model(source_dir, log_cb):
        raise RuntimeError("cuda out of memory")

    _patch_workers(monkeypatch, load_model_for_safetensors=failing_load_model)
    hf_client = _make_hf_client()
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    statuses = _status_map(job)
    assert statuses["W4A16_GPTQ"].status == QuantStatus.SKIPPED
    assert "Lane A setup failed" in statuses["W4A16_GPTQ"].error
    assert statuses["Q4_K_M"].status == QuantStatus.DONE
    assert job.status == "complete"
    assert progress.of_type("lane_failed")[0]["lane"] == "A"
    hf_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# (c) token wipe on success AND on source-download failure
# ---------------------------------------------------------------------------

async def test_token_wiped_on_success(tmp_path, monkeypatch):
    job = _make_job(["Q4_K_M"])
    _patch_workers(monkeypatch)
    hf_client = _make_hf_client()
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    assert job.status == "complete"
    hf_client.close.assert_called_once()


async def test_token_wiped_when_source_download_raises(tmp_path, monkeypatch):
    job = _make_job(["Q4_K_M"])
    _patch_workers(monkeypatch)
    hf_client = _make_hf_client()
    hf_client.download_snapshot.side_effect = RuntimeError("network is down")
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    assert job.status == "failed"
    # the lane never ran -- quant is untouched (still pending)
    assert _status_map(job)["Q4_K_M"].status == QuantStatus.PENDING
    hf_client.close.assert_called_once()
    assert progress.of_type("job_failed")


# ---------------------------------------------------------------------------
# (d) imatrix failure skips only I-quants, K-quants proceed
# ---------------------------------------------------------------------------

async def test_imatrix_failure_skips_only_i_quants(tmp_path, monkeypatch):
    job = _make_job(["Q4_K_M", "IQ2_XS"])

    def failing_imatrix(bf16_gguf, cal_text, output_imatrix, log_cb, n_chunks=100):
        raise RuntimeError("imatrix crashed")

    _patch_workers(monkeypatch, compute_imatrix=failing_imatrix)
    hf_client = _make_hf_client()
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    statuses = _status_map(job)
    assert statuses["IQ2_XS"].status == QuantStatus.SKIPPED
    assert "imatrix failed" in statuses["IQ2_XS"].error
    assert statuses["Q4_K_M"].status == QuantStatus.DONE
    assert job.status == "complete"
    hf_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# (e) readme update failure publishes readme_failed but job still completes
# ---------------------------------------------------------------------------

async def test_readme_update_failure_publishes_readme_failed_but_job_completes(tmp_path, monkeypatch):
    job = _make_job(["Q4_K_M"], update_source_readme=True)

    def failing_readme(job, hf_client):
        raise RuntimeError("readme write failed")

    _patch_workers(monkeypatch, update_source_readme=failing_readme)
    hf_client = _make_hf_client()
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    assert job.status == "complete"
    readme_failed = progress.of_type("readme_failed")
    assert len(readme_failed) == 1
    assert "readme write failed" in readme_failed[0]["error"]
    hf_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Controller decision 2: quantize_safetensors receives source_dir explicitly
# ---------------------------------------------------------------------------

async def test_quantize_safetensors_receives_explicit_source_dir(tmp_path, monkeypatch):
    job = _make_job(["W4A16_GPTQ"])
    captured = {}

    def capturing_quantize_safetensors(model, tokenizer, format, calibration, output_dir, source_dir, log_cb):
        captured["source_dir"] = source_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model.safetensors").write_bytes(b"x")

    _patch_workers(monkeypatch, quantize_safetensors=capturing_quantize_safetensors)
    hf_client = _make_hf_client()
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    assert captured["source_dir"] == tmp_path / "source"
    assert _status_map(job)["W4A16_GPTQ"].status == QuantStatus.DONE


# ---------------------------------------------------------------------------
# (f) unexpected exception escaping one lane doesn't orphan the sibling lane
# ---------------------------------------------------------------------------

async def test_unexpected_lane_exception_does_not_orphan_sibling_lane(tmp_path, monkeypatch):
    """If a lane coroutine raises an exception that escapes its own internal
    try/except (i.e. a bug, not a modeled per-quant/per-lane-setup failure),
    `run_job` must still wait for the sibling lane to finish naturally before
    tearing anything down -- otherwise `hf_client.close()` would wipe the
    token out from under a lane that's still mid-upload.
    """
    job = _make_job(["W4A16_GPTQ", "Q4_K_M"])  # A: well-behaved, B: broken

    async def broken_lane_b(job, source_dir, calibration, hf_client, progress, workdir):
        raise RuntimeError("lane b exploded unexpectedly")

    _patch_workers(monkeypatch)  # lane A's workers stay well-behaved
    monkeypatch.setattr(job_runner_module, "_run_lane_b", broken_lane_b)
    hf_client = _make_hf_client()
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    statuses = _status_map(job)
    # the job as a whole is marked failed because of the escaped lane error
    assert job.status == "failed"
    job_failed = progress.of_type("job_failed")
    assert len(job_failed) == 1
    assert "lane B" in job_failed[0]["error"]
    assert "lane b exploded unexpectedly" in job_failed[0]["error"]

    # the sibling (lane A) lane was NOT orphaned -- it ran to completion and
    # reached a terminal status before close() fired
    assert statuses["W4A16_GPTQ"].status == QuantStatus.DONE

    # close() still only fires exactly once, after both lanes genuinely
    # finished
    hf_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Controller decision 3: per-quant repo_id overrides honored in both lanes
# ---------------------------------------------------------------------------

async def test_repo_id_overrides_honored_per_lane(tmp_path, monkeypatch):
    job = _make_job(["Q4_K_M", "W4A16_GPTQ"], owner="acme", source_model="acme/model-7b")
    _status_map(job)["Q4_K_M"].repo_id = "custom/gguf-repo"

    _patch_workers(monkeypatch)
    hf_client = _make_hf_client()
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    statuses = _status_map(job)
    # explicit override preserved and actually used for the upload call
    assert statuses["Q4_K_M"].repo_id == "custom/gguf-repo"
    upload_file_repo_ids = [call.args[0] for call in hf_client.upload_file.call_args_list]
    assert "custom/gguf-repo" in upload_file_repo_ids

    # no override -> default "{owner}/{basename}-{quant_id}" pattern, and the
    # resolved repo_id is written back onto the QuantResult
    assert statuses["W4A16_GPTQ"].repo_id == "acme/model-7b-W4A16_GPTQ"
    upload_folder_repo_ids = [call.args[0] for call in hf_client.upload_folder.call_args_list]
    assert "acme/model-7b-W4A16_GPTQ" in upload_folder_repo_ids


# ---------------------------------------------------------------------------
# C2: the job workdir (source snapshot etc.) is always deleted at job end
# ---------------------------------------------------------------------------

async def test_workdir_removed_after_success(tmp_path, monkeypatch):
    job = _make_job(["Q4_K_M"])
    _patch_workers(monkeypatch)
    hf_client = _make_hf_client()
    progress = RecordingProgress()
    workdir = tmp_path / "wd"
    workdir.mkdir()

    await run_job(job, hf_client, CALIBRATION, progress, workdir)

    assert job.status == "complete"
    assert not workdir.exists()


async def test_workdir_removed_after_failure(tmp_path, monkeypatch):
    job = _make_job(["Q4_K_M"])
    _patch_workers(monkeypatch)
    hf_client = _make_hf_client()
    hf_client.download_snapshot.side_effect = RuntimeError("network is down")
    progress = RecordingProgress()
    workdir = tmp_path / "wd"
    workdir.mkdir()

    await run_job(job, hf_client, CALIBRATION, progress, workdir)

    assert job.status == "failed"
    assert not workdir.exists()


# ---------------------------------------------------------------------------
# I4: launch-time architecture refusal (only Mistral / Mistral3 allowed)
# ---------------------------------------------------------------------------

async def test_non_mistral_arch_refused_before_source_download(tmp_path, monkeypatch):
    job = _make_job(["Q4_K_M"])
    _patch_workers(monkeypatch)
    hf_client = _make_hf_client(arch="LlamaForCausalLM")
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    assert job.status == "failed"
    # The expensive source download must never have started.
    hf_client.download_snapshot.assert_not_called()
    failed = progress.of_type("job_failed")
    assert failed and "Unsupported architecture" in failed[0]["error"]


@pytest.mark.parametrize("arch", [_TEXT_ARCH, _MULTIMODAL_ARCH])
async def test_mistral_archs_proceed_past_guard(tmp_path, monkeypatch, arch):
    job = _make_job(["Q4_K_M"])
    _patch_workers(monkeypatch)
    hf_client = _make_hf_client(arch=arch)
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    assert job.status == "complete"
    hf_client.download_snapshot.assert_called_once()


# ---------------------------------------------------------------------------
# I5: full per-quant logs are written to LOG_ROOT/<job_id>/<quant>.log
# ---------------------------------------------------------------------------

async def test_per_quant_log_files_written(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner_module, "LOG_ROOT", tmp_path / "logs")
    job = _make_job(["Q4_K_M"])

    def logging_gguf_quantize(bf16_gguf, output_gguf, format, log_cb, imatrix=None):
        log_cb("hello from the quant worker")
        output_gguf = Path(output_gguf)
        output_gguf.parent.mkdir(parents=True, exist_ok=True)
        output_gguf.write_bytes(b"data")

    _patch_workers(monkeypatch, gguf_quantize=logging_gguf_quantize)
    hf_client = _make_hf_client()
    progress = RecordingProgress()
    workdir = tmp_path / "wd"
    workdir.mkdir()

    await run_job(job, hf_client, CALIBRATION, progress, workdir)

    log_file = tmp_path / "logs" / job.id / "Q4_K_M.log"
    assert log_file.exists()
    assert "hello from the quant worker" in log_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# I6: hf uploads are retried with backoff (3 attempts) before giving up
# ---------------------------------------------------------------------------

async def test_upload_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner_module, "UPLOAD_BACKOFF_BASE", 0)  # no real sleeps
    job = _make_job(["Q4_K_M"])
    _patch_workers(monkeypatch)
    hf_client = _make_hf_client()

    calls = {"n": 0}

    def flaky_upload_file(repo_id, file_path, path_in_repo, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient network blip")
        return "https://huggingface.co/repo/blob/main/file.gguf"

    hf_client.upload_file.side_effect = flaky_upload_file
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    assert _status_map(job)["Q4_K_M"].status == QuantStatus.DONE
    assert calls["n"] == 3  # failed twice, succeeded on the third attempt


async def test_upload_fails_after_max_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(job_runner_module, "UPLOAD_BACKOFF_BASE", 0)
    job = _make_job(["Q4_K_M"])
    _patch_workers(monkeypatch)
    hf_client = _make_hf_client()

    calls = {"n": 0}

    def always_fail(repo_id, file_path, path_in_repo, **kwargs):
        calls["n"] += 1
        raise RuntimeError("network stays down")

    hf_client.upload_file.side_effect = always_fail
    progress = RecordingProgress()

    await run_job(job, hf_client, CALIBRATION, progress, tmp_path)

    assert _status_map(job)["Q4_K_M"].status == QuantStatus.FAILED
    assert calls["n"] == 3  # exactly UPLOAD_MAX_ATTEMPTS attempts
