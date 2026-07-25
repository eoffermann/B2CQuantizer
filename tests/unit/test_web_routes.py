"""FastAPI routes: setup page, health, job creation/history/view.

HFClient and run_job are monkeypatched at the b2cq.web.routes module level
for the job-creation tests so no network access or torch/llm-compressor
work happens. The SSE stream endpoint is not exercised here beyond
existing (streaming tests hang easily under TestClient).
"""
from __future__ import annotations

from collections import namedtuple
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import b2cq.web.routes as routes
from b2cq.calibration import CalibrationSource
from b2cq.job_model import QuantResult, QuantStatus
from b2cq.main import app

client = TestClient(app)

_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])


@pytest.fixture(autouse=True)
def _web_test_env(monkeypatch):
    """Isolate the module-level JobStore between tests and neutralize the disk
    preflight so it doesn't depend on the host machine's real free space.

    - JobStore isolation keeps the single-job guard (I2) from tripping on a
      leftover pending/running job created by an earlier test.
    - disk_usage is stubbed to report plenty of free space so the I3 preflight
      passes deterministically; individual tests can re-stub it to go low.
    """
    routes.JOB_STORE._jobs.clear()
    monkeypatch.setattr(
        routes.shutil, "disk_usage",
        lambda path: _DiskUsage(total=10**13, used=0, free=10**13),
    )
    yield


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_setup_page_lists_all_quant_groups():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    # All five quant families should be represented by at least one member.
    assert "Q4_K_M" in body       # gguf_k
    assert "IQ2_XS" in body       # gguf_i
    assert "Q8_0" in body         # gguf_misc
    assert "mmproj-f16" in body   # gguf_mmproj
    assert "NVFP4" in body        # safetensors
    # NVFP4 should be gated (no torch/no Blackwell in this venv).
    assert "needs Blackwell GPU" in body


def test_history_empty_initially():
    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert "Jobs in this session" in resp.text


def test_job_view_unknown_id_is_404():
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def _install_fakes(monkeypatch):
    """Monkeypatch HFClient + run_job at the routes module level."""
    fake_client_instance = MagicMock()
    fake_client_instance.whoami.return_value = {"name": "testuser"}

    def fake_hf_client(token: str):
        assert token  # sanity: token was forwarded
        return fake_client_instance

    async def fake_run_job(job, hf_client, calibration, progress, workdir):
        return None

    monkeypatch.setattr(routes, "HFClient", fake_hf_client)
    monkeypatch.setattr(routes, "run_job", fake_run_job)
    return fake_client_instance


def test_create_job_redirects_and_renders_quant_table(monkeypatch):
    _install_fakes(monkeypatch)

    data = {
        "source_model": "acme/test-model-7b",
        "hf_token": "hf_testtoken123",
        "calibration_type": "bundled",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "owner": "acme",
    }
    # Force multipart encoding (endpoint has an UploadFile param), without
    # actually exercising the upload branch (calibration_type is "bundled").
    files = {"calibration_file": ("cal.jsonl", b"", "application/octet-stream")}

    resp = client.post("/jobs", data=data, files=files, follow_redirects=False)
    assert resp.status_code == 303
    location = resp.headers["location"]
    assert location.startswith("/jobs/")
    job_id = location.split("/jobs/")[1]

    view = client.get(location)
    assert view.status_code == 200
    body = view.text
    assert job_id in body
    assert "acme/test-model-7b" in body
    assert "Q4_K_M" in body
    assert "Q5_K_M" in body

    # And it now shows up in history.
    history = client.get("/jobs")
    assert history.status_code == 200
    assert job_id in history.text


def test_create_job_owner_resolved_from_whoami_when_absent(monkeypatch):
    fake_client_instance = _install_fakes(monkeypatch)

    data = {
        "source_model": "acme/test-model-7b",
        "hf_token": "hf_testtoken123",
        "calibration_type": "bundled",
        "quants": ["Q4_K_M"],
        # owner intentionally omitted
    }
    files = {"calibration_file": ("cal.jsonl", b"", "application/octet-stream")}

    resp = client.post("/jobs", data=data, files=files, follow_redirects=False)
    assert resp.status_code == 303
    fake_client_instance.whoami.assert_called_once()

    view = client.get(resp.headers["location"])
    assert "testuser" in view.text


def test_create_job_hf_dataset_token_not_persisted_on_job(monkeypatch):
    """CRITICAL: the raw HF token must never live on the stored Job/JobStore
    (it's never evicted), even though it's needed transiently to load the
    calibration dataset and to build the HFClient."""
    _install_fakes(monkeypatch)
    monkeypatch.setattr(routes, "load_calibration", lambda source: [{"messages": []}])

    data = {
        "source_model": "acme/test-model-7b",
        "hf_token": "hf_supersecrettoken",
        "calibration_type": "hf_dataset",
        "calibration_dataset": "acme/some-dataset",
        "quants": ["Q4_K_M"],
        "owner": "acme",
    }
    resp = client.post("/jobs", data=data, follow_redirects=False)
    assert resp.status_code == 303
    job_id = resp.headers["location"].split("/jobs/")[1]

    job = routes.JOB_STORE.get(job_id)
    assert job.calibration.type == "hf_dataset"
    assert job.calibration.hf_token is None
    # And the secret must not leak into the rendered page either.
    assert "hf_supersecrettoken" not in client.get(resp.headers["location"]).text


def test_job_view_seeds_alpine_state_from_stored_job(monkeypatch):
    """IMPORTANT: reloading the job page (e.g. mid-flight or after
    completion) must reflect the persisted QuantResult state, not always
    show 'pending'/blank cells until the next SSE event arrives."""
    _install_fakes(monkeypatch)

    data = {
        "source_model": "acme/test-model-7b",
        "hf_token": "hf_testtoken123",
        "calibration_type": "bundled",
        "quants": ["Q4_K_M"],
        "owner": "acme",
    }
    files = {"calibration_file": ("cal.jsonl", b"", "application/octet-stream")}
    resp = client.post("/jobs", data=data, files=files, follow_redirects=False)
    job_id = resp.headers["location"].split("/jobs/")[1]

    job = routes.JOB_STORE.get(job_id)
    q = job.quants[0]
    q.status = QuantStatus.DONE
    q.elapsed_seconds = 42.0
    q.output_size_bytes = 12345
    q.upload_url = "https://huggingface.co/acme/test-model-7b-Q4_K_M"

    view = client.get(resp.headers["location"])
    assert view.status_code == 200
    body = view.text
    assert "'done'" in body or '"done"' in body
    assert "42.0" in body
    assert "https://huggingface.co/acme/test-model-7b-Q4_K_M" in body


def test_calibration_upload_endpoint_removed():
    """MINOR/controller decision: the unauthenticated multipart-upload
    endpoint is dead code and a disk-fill/arbitrary-file-write surface; it
    has been removed in favor of the setup form's direct POST /jobs."""
    resp = client.post(
        "/calibration/upload",
        files={"calibration_file": ("cal.jsonl", b"{}", "application/octet-stream")},
    )
    assert resp.status_code == 404


def test_job_stream_route_returns_event_stream():
    """MINOR: cheap smoke test for the SSE route.

    A real client.stream(...) GET hangs under Starlette's TestClient here
    (the endpoint never flushes a response until PROGRESS publishes an
    event for the job, which never happens for an unused job id), so per
    the fallback plan we assert the route is registered with the expected
    path/methods instead of consuming the stream.
    """
    matches = [r for r in app.routes if getattr(r, "path", None) == "/jobs/{job_id}/stream"]
    assert len(matches) == 1
    assert "GET" in matches[0].methods


def _job_form(**overrides):
    data = {
        "source_model": "acme/test-model-7b",
        "hf_token": "hf_testtoken123",
        "calibration_type": "bundled",
        "quants": ["Q4_K_M"],
        "owner": "acme",
    }
    data.update(overrides)
    files = {"calibration_file": ("cal.jsonl", b"", "application/octet-stream")}
    return data, files


# ---------------------------------------------------------------------------
# I2: single-job guard -- reject a second submission while one is in flight
# ---------------------------------------------------------------------------

def test_create_job_rejected_when_another_job_running(monkeypatch):
    _install_fakes(monkeypatch)
    # Seed a job that is still running.
    running = routes.JOB_STORE.create(
        source_model="acme/other", owner="acme",
        quants=[QuantResult(quant_id="Q4_K_M", status=QuantStatus.RUNNING, lane="B")],
        calibration=CalibrationSource(type="bundled"), private=False,
        update_source_readme=False,
    )
    running.status = "running"

    data, files = _job_form()
    resp = client.post("/jobs", data=data, files=files, follow_redirects=False)
    assert resp.status_code == 409
    assert running.id in resp.json()["detail"]


# ---------------------------------------------------------------------------
# I3: disk preflight -- refuse when free space is below the threshold
# ---------------------------------------------------------------------------

def test_create_job_rejected_when_disk_below_threshold(monkeypatch):
    _install_fakes(monkeypatch)
    # Only ~10 GB free, well under the 150 GB requirement.
    monkeypatch.setattr(
        routes.shutil, "disk_usage",
        lambda path: _DiskUsage(total=200 * 10**9, used=190 * 10**9, free=10 * 10**9),
    )
    data, files = _job_form()
    resp = client.post("/jobs", data=data, files=files, follow_redirects=False)
    assert resp.status_code == 400
    assert "Insufficient disk" in resp.json()["detail"]


def test_create_job_proceeds_when_disk_sufficient(monkeypatch):
    _install_fakes(monkeypatch)
    # Plenty of free space (the autouse fixture already stubs this, but assert
    # the happy path explicitly as the other half of the both-ways I3 check).
    monkeypatch.setattr(
        routes.shutil, "disk_usage",
        lambda path: _DiskUsage(total=500 * 10**9, used=0, free=500 * 10**9),
    )
    data, files = _job_form()
    resp = client.post("/jobs", data=data, files=files, follow_redirects=False)
    assert resp.status_code == 303
