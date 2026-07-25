"""FastAPI routes: setup page, health, job creation/history/view.

HFClient and run_job are monkeypatched at the b2cq.web.routes module level
for the job-creation tests so no network access or torch/llm-compressor
work happens. The SSE stream endpoint is not exercised here beyond
existing (streaming tests hang easily under TestClient).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import b2cq.web.routes as routes
from b2cq.main import app

client = TestClient(app)


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
