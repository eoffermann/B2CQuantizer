# B2CQuantizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RunPod-hosted Docker web app that quantizes a HuggingFace Mistral / Mistral3 source model into a user-selected set of GGUF + safetensors formats, uploads each result to its own HuggingFace repo (separate repo per safetensors format; single shared repo for all GGUFs of a model), and appends a canonical `## Quantizations` section to the source model's README with links to every generated repo.

**Architecture:** FastAPI backend (Python 3.10) + HTMX-served HTML frontend, single-user single-job pod. On job launch, two lanes run concurrently: **Lane A** (GPU-bound) runs `llm-compressor` for each selected safetensors quant serially against a single BF16 base loaded on GPU; **Lane B** (CPU-bound) converts BF16 → BF16 GGUF via `llama.cpp`'s `convert_hf_to_gguf.py`, optionally builds an imatrix, then runs `llama-quantize` for each selected GGUF variant. Both lanes upload each result independently and stream logs via Server-Sent Events to a live job dashboard. Token held in-memory only, wiped on job completion.

**Tech Stack:** Python 3.10, FastAPI, Uvicorn, HTMX, Alpine.js, Tailwind (CDN), Server-Sent Events, `huggingface_hub`, `llm-compressor`, `transformers`, `accelerate`, `torch` (CUDA 12.8), `datasets`, `llama.cpp` (built from source in-image), Docker base `nvidia/cuda:12.8.0-devel-ubuntu22.04`.

## Global Constraints

- **Python version:** 3.10 exactly. `pyproject.toml` sets `requires-python = "==3.10.*"`. (`llm-compressor` and `torch` wheel matrices target 3.10 most cleanly at the CUDA-12.8 tier.)
- **CUDA:** 12.8 (needed for Blackwell SM120 support so NVFP4 kernels are available on RTX 5090 / RTX PRO 6000 Blackwell pods).
- **GPU class assumed at runtime:** at least H100 80GB or A100 96GB for 24B-parameter source models. NVFP4 requires Blackwell (RTX 5090, RTX PRO 6000 Blackwell). App auto-detects with `torch.cuda.get_device_properties(0).major >= 12` and grays out NVFP4 in the UI if not present.
- **Disk assumed at runtime:** ≥200 GB pod storage for 24B models. App checks free space at job start and refuses to launch with a clear error if <150 GB free.
- **Model scope:** Mistral / Mistral3 family only. Refuse to launch on any other `config.json` `architectures` value with a clear error message; do not silently proceed.
- **HuggingFace token:** in-memory only. Never written to disk, never exported as an env var visible to subprocesses (pass explicitly to `huggingface_hub` API calls). Wiped on job end (success OR failure).
- **Web-UI auth:** none. The pod's port exposure is the trust boundary. App does not implement any auth mechanism.
- **Package pins** (add to `pyproject.toml` — verify latest compatible on first build, but these are the anchors):
  - `fastapi==0.115.*`
  - `uvicorn[standard]==0.32.*`
  - `sse-starlette==2.1.*`
  - `huggingface_hub==0.26.*`
  - `transformers==4.46.*`
  - `accelerate==1.1.*`
  - `torch==2.5.*` (CUDA 12.8 wheel)
  - `llm-compressor==0.3.*` (or latest — verify Mistral3 support; anchor after first build)
  - `datasets==3.1.*`
  - `python-multipart==0.0.20` (for form uploads)
  - `jinja2==3.1.*`
  - `pydantic==2.9.*`
  - `mistral-common==1.11.5` (EXACT pin; see FrndoBrain-class invariants below)
  - `pytest==8.3.*`
  - `pytest-asyncio==0.24.*`
- **llama.cpp version:** build a specific commit into the image; anchor commit SHA after first successful build. Recommend a recent tag (e.g., `b4200` or newer) that supports Mistral3 conversion. Note the two upstream bugs in `MmprojModel.filter_tensors` and pixtral activation flag documented in the `patch_convert_hf_to_gguf.py` section of Task 8 — the plan includes a runtime patch script instead of a fork.
- **mistral-common pin (FrndoBrain-class artifacts only):** `mistral-common == 1.11.5`, added to `pyproject.toml`. This matches the FrndoBrain serving stack; the private attribute `_message_position_to_encode_tools_settings` on `InstructTokenizerV11` is version-fragile (it was `_user_message_position_to_encode_tools` in 1.9.x). The safetensors worker MUST hard-fail on version mismatch AND on attribute-missing (see SPEC §14). The GGUF path does NOT use mistral-common and is unaffected.
- **FrndoBrain-class artifact detection is automatic** via presence of `tekken.json` in the source model repo (see SPEC §14). If detected, the safetensors calibration pass renders through mistral-common with the tool-placement attribute flip; otherwise it uses HF `apply_chat_template` as a fallback for generic Mistral models.

## Repository layout

```
B2CQuantizer/
├── SPEC.md                         # design specification (already present)
├── PLAN.md                         # this file
├── README.md                       # user-facing readme (created in Task 1)
├── LICENSE                         # MIT (created in Task 1)
├── pyproject.toml                  # Python deps + tooling config
├── .gitignore                      # standard Python + Docker + build artifacts
├── .dockerignore                   # what NOT to send to Docker build context
├── Dockerfile                      # nvidia/cuda 12.8 + Python 3.10 + llama.cpp + app
├── docker/
│   └── entrypoint.sh              # container startup (dump diag, run uvicorn)
├── src/
│   └── b2cq/
│       ├── __init__.py
│       ├── main.py                # FastAPI app entry
│       ├── config.py              # settings (port, log dir, calibration path, etc.)
│       ├── quant_catalog.py       # declarative list of all 30 quant variants
│       ├── hf_client.py           # thin wrapper around huggingface_hub, holds token
│       ├── calibration.py         # bundled + upload + HF-dataset loaders
│       ├── progress.py            # in-memory pub/sub for SSE
│       ├── job_model.py           # Job / QuantResult dataclasses, in-memory store
│       ├── job_runner.py          # two-lane orchestrator
│       ├── workers/
│       │   ├── __init__.py
│       │   ├── safetensors.py     # llm-compressor recipes + runner
│       │   ├── gguf_convert.py    # convert_hf_to_gguf.py wrapper (BF16 GGUF)
│       │   ├── gguf_quantize.py   # llama-quantize + imatrix wrapper
│       │   └── mmproj.py          # mmproj export + patch script application
│       ├── readme_updater.py      # source-repo README ## Quantizations section
│       └── web/
│           ├── __init__.py
│           ├── routes.py          # /, /jobs, /jobs/<id>, /jobs/<id>/stream (SSE)
│           └── templates/
│               ├── base.html      # HTMX + Alpine + Tailwind CDN
│               ├── setup.html     # setup form
│               ├── job.html       # live job dashboard
│               └── history.html   # in-session job list
├── src/b2cq_data/
│   └── calibration/
│       └── bundled.jsonl          # 1000-sample generic corpus (pileval + wikitext)
├── scripts/
│   ├── build_llama_cpp.sh         # invoked from Dockerfile; clones + builds llama.cpp
│   └── patch_convert_hf_to_gguf.py # runtime patch for two Mistral3 mmproj bugs
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_quant_catalog.py
│   │   ├── test_hf_client.py
│   │   ├── test_calibration.py
│   │   ├── test_progress.py
│   │   ├── test_job_model.py
│   │   └── test_readme_updater.py
│   ├── integration/
│   │   ├── test_gguf_workers.py   # needs llama.cpp built; runs on tiny stub model
│   │   └── test_safetensors_worker.py  # needs GPU; can be skipped without --gpu
│   └── smoke/
│       └── README.md              # manual E2E smoke procedure (real HF model)
└── docs/
    ├── deployment.md              # how to launch a RunPod pod for this app
    └── troubleshooting.md         # common issues + fixes
```

---

### Task 1: Repository scaffolding

**Files:**
- Create: `.gitignore`, `.dockerignore`, `pyproject.toml`, `LICENSE`, `README.md`, `src/b2cq/__init__.py`, `src/b2cq_data/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: nothing
- Produces: a Python package importable as `b2cq`, project metadata, license, high-level README

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.venv/
venv/
.env
.pytest_cache/
.coverage
htmlcov/

# Docker artifacts
*.tar

# App runtime
/workspace/
/logs/
*.log

# OS / editor
.DS_Store
Thumbs.db
.vscode/
.idea/
```

- [ ] **Step 2: Create `.dockerignore`**

```dockerignore
.git/
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
tests/
docs/
*.md
!SPEC.md
!PLAN.md
!README.md
.gitignore
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "b2cq"
version = "0.1.0"
description = "Web app that batch-quantizes Mistral/Mistral3 models on RunPod GPU pods."
readme = "README.md"
requires-python = "==3.10.*"
license = {text = "MIT"}
authors = [{name = "BigBlueCeiling"}]
dependencies = [
    "fastapi==0.115.*",
    "uvicorn[standard]==0.32.*",
    "sse-starlette==2.1.*",
    "huggingface_hub==0.26.*",
    "transformers==4.46.*",
    "accelerate==1.1.*",
    "datasets==3.1.*",
    "python-multipart==0.0.20",
    "jinja2==3.1.*",
    "pydantic==2.9.*",
    # torch installed separately in Dockerfile to pick up CUDA 12.8 wheel
    # llm-compressor installed separately (may need extra index)
]

[project.optional-dependencies]
dev = [
    "pytest==8.3.*",
    "pytest-asyncio==0.24.*",
    "httpx==0.27.*",  # for FastAPI test client
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 4: Create MIT `LICENSE`**

Standard MIT license text; author "BigBlueCeiling", year 2026.

- [ ] **Step 5: Create top-level `README.md`**

```markdown
# B2CQuantizer

Docker web app that quantizes a HuggingFace Mistral / Mistral3 model into GGUF + safetensors formats and uploads them to HuggingFace, updating the source model's README with links to every generated quant repo. Runs on a RunPod GPU pod.

## Quick start

1. Launch a RunPod pod with an 80GB+ GPU (H100 recommended; RTX 5090 / Blackwell required if you want NVFP4). ≥200 GB pod disk for 24B models.
2. `docker run -d --gpus all -p 8000:8000 ghcr.io/bigblueceiling/b2cquantizer:latest`
3. Access the UI via RunPod's port-forwarding for port 8000.
4. Enter source model + HF token, select quants, hit Launch.

Design: see `SPEC.md`. Implementation plan: see `PLAN.md`.

## License

MIT
```

- [ ] **Step 6: Create empty `__init__.py` files and conftest**

```python
# src/b2cq/__init__.py
"""B2CQuantizer — Docker-based batch model quantizer for Mistral / Mistral3."""
__version__ = "0.1.0"
```

```python
# src/b2cq_data/__init__.py
```

```python
# tests/__init__.py
```

```python
# tests/conftest.py
"""Shared pytest fixtures — extended in each test module as needed."""
```

- [ ] **Step 7: Install and verify importability**

Run:
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -e ".[dev]"
python -c "import b2cq; print(b2cq.__version__)"
```
Expected: prints `0.1.0`.

- [ ] **Step 8: Commit**

```bash
git add .gitignore .dockerignore pyproject.toml LICENSE README.md src/ tests/
git commit -m "chore: repo scaffolding, license, README, pyproject"
```

---

### Task 2: Dockerfile with CUDA + llama.cpp build

**Files:**
- Create: `Dockerfile`, `docker/entrypoint.sh`, `scripts/build_llama_cpp.sh`

**Interfaces:**
- Consumes: repo contents (per `.dockerignore`)
- Produces: a Docker image tagged `b2cq:dev` that (a) has CUDA 12.8, (b) has Python 3.10 + all deps installed, (c) has `llama.cpp` built with CUDA, (d) starts uvicorn on port 8000

- [ ] **Step 1: Write `scripts/build_llama_cpp.sh`**

```bash
#!/usr/bin/env bash
# Clone and build llama.cpp with CUDA support. Anchored to a specific
# commit SHA after first successful build (edit LLAMA_CPP_COMMIT below).
set -euo pipefail

LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-master}"  # replace with pinned SHA after first build
INSTALL_PREFIX="${INSTALL_PREFIX:-/opt/llama.cpp}"

apt-get update
apt-get install -y --no-install-recommends git cmake build-essential libcurl4-openssl-dev
rm -rf /var/lib/apt/lists/*

git clone https://github.com/ggerganov/llama.cpp.git "${INSTALL_PREFIX}"
cd "${INSTALL_PREFIX}"
git checkout "${LLAMA_CPP_COMMIT}"

cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120"
cmake --build build --config Release -j "$(nproc)" \
    --target llama-quantize llama-imatrix

# Also install the Python conversion scripts and their requirements.
python3 -m pip install --no-cache-dir -r requirements/requirements-convert_hf_to_gguf.txt

echo "llama.cpp built at commit $(git rev-parse HEAD)"
```

- [ ] **Step 2: Write `docker/entrypoint.sh`**

```bash
#!/usr/bin/env bash
set +e  # we handle failures manually so the pod stays up for debugging

echo "======================= DIAG START ======================="
echo "[entrypoint] date UTC:   $(date -u)"
echo "[entrypoint] hostname:   $(hostname)"
echo ""
echo "--- nvidia-smi ---"
nvidia-smi 2>&1 | head -30
echo ""
echo "--- torch CUDA visibility ---"
python3 - <<'PY' 2>&1
try:
    import torch
    print(f"torch version : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}  sm_{p.major}{p.minor}  {p.total_memory / (1024**3):.1f} GiB")
except Exception as e:
    print(f"torch probe FAILED: {e!r}")
PY
echo ""
echo "--- llama.cpp binaries ---"
ls -la /opt/llama.cpp/build/bin/ | grep -E "quantize|imatrix" | head -5
echo ""
echo "--- disk ---"
df -h /workspace 2>/dev/null || df -h /
echo "======================= DIAG END ========================="

mkdir -p /workspace /workspace/logs

exec uvicorn b2cq.main:app --host 0.0.0.0 --port 8000 --log-level info
```

- [ ] **Step 3: Write `Dockerfile`**

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf-cache

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git curl ca-certificates \
        # llama.cpp build deps handled inside build_llama_cpp.sh but we pre-install cmake for speed
        cmake build-essential \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 12.8 wheel (must come from pytorch index)
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.5.*"

# llm-compressor and other heavy deps
RUN python3 -m pip install "llm-compressor==0.3.*"

# App deps (from pyproject.toml, but copied first for cache reuse)
COPY pyproject.toml /app/pyproject.toml
COPY src/b2cq/__init__.py /app/src/b2cq/__init__.py
COPY src/b2cq_data/__init__.py /app/src/b2cq_data/__init__.py
WORKDIR /app
RUN python3 -m pip install -e ".[dev]"

# Build llama.cpp
COPY scripts/build_llama_cpp.sh /opt/build_llama_cpp.sh
RUN chmod +x /opt/build_llama_cpp.sh && /opt/build_llama_cpp.sh

# Copy app source (last, so app changes don't invalidate dep layers)
COPY src/ /app/src/
COPY scripts/patch_convert_hf_to_gguf.py /opt/patch_convert_hf_to_gguf.py
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/opt/entrypoint.sh"]
```

- [ ] **Step 4: Write a trivial `src/b2cq/main.py` so the image can start**

```python
"""FastAPI app entry — will be expanded in later tasks."""
from fastapi import FastAPI

app = FastAPI(title="B2CQuantizer", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 5: Write a stub `scripts/patch_convert_hf_to_gguf.py`**

Placeholder that will be filled in Task 11:
```python
"""Runtime patch for llama.cpp's convert_hf_to_gguf.py — placeholder.
Filled in Task 11 (mmproj export) with the two Mistral3-specific fixes."""
if __name__ == "__main__":
    print("[patch] placeholder — no fixes applied yet")
```

- [ ] **Step 6: Build the image**

Run: `docker build -t b2cq:dev .`
Expected: build succeeds. Log a note in the plan-execution transcript with the actual llama.cpp commit SHA (from the build output line `llama.cpp built at commit …`), then edit `scripts/build_llama_cpp.sh` line `LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-master}"` to pin that SHA.

- [ ] **Step 7: Run the image + smoke-test /health**

Run:
```bash
docker run --rm -d --name b2cq-test --gpus all -p 8000:8000 b2cq:dev
sleep 20  # wait for startup diag + uvicorn ready
curl -sSf http://localhost:8000/health
docker rm -f b2cq-test
```
Expected: `curl` prints `{"status":"ok"}`.

- [ ] **Step 8: Commit**

```bash
git add Dockerfile docker/ scripts/ src/b2cq/main.py
git commit -m "feat: Dockerfile with CUDA 12.8, llama.cpp build, uvicorn entrypoint"
```

---

### Task 3: Quant catalog

**Files:**
- Create: `src/b2cq/quant_catalog.py`
- Create: `tests/unit/test_quant_catalog.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `class QuantFamily(str, Enum)`: `GGUF_K`, `GGUF_I`, `GGUF_MISC`, `GGUF_MMPROJ`, `SAFETENSORS`
  - `class QuantSpec(BaseModel)`: `id`, `family`, `name`, `format` (e.g., `"Q4_K_M"`, `"W4A16_GPTQ"`), `needs_calibration: bool`, `needs_imatrix: bool`, `notes: str`, `hardware_requirements: list[str]` (e.g. `["blackwell"]` for NVFP4)
  - `CATALOG: list[QuantSpec]` — the full 30-item catalog
  - `def get(id: str) -> QuantSpec`
  - `def by_family(family: QuantFamily) -> list[QuantSpec]`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_quant_catalog.py`:
```python
"""Catalog must expose the full v1 quant list with correct metadata."""
import pytest
from b2cq.quant_catalog import CATALOG, QuantFamily, get, by_family


def test_catalog_has_expected_counts():
    assert len(by_family(QuantFamily.GGUF_K)) == 9   # Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K
    assert len(by_family(QuantFamily.GGUF_I)) == 12  # IQ1_S, IQ1_M, IQ2_XXS/XS/S/M, IQ3_XXS/XS/S/M, IQ4_XS/NL
    assert len(by_family(QuantFamily.GGUF_MISC)) == 3   # Q8_0, F16, BF16
    assert len(by_family(QuantFamily.GGUF_MMPROJ)) == 1  # mmproj-f16
    assert len(by_family(QuantFamily.SAFETENSORS)) == 5  # W4A16_GPTQ, W4A16_AWQ, NVFP4, FP8_E4M3, FP8_E5M2
    assert len(CATALOG) == 30


def test_calibration_flags_correct():
    assert not get("Q4_K_M").needs_calibration
    assert not get("Q8_0").needs_calibration
    assert get("IQ4_XS").needs_calibration and get("IQ4_XS").needs_imatrix
    assert get("W4A16_GPTQ").needs_calibration and not get("W4A16_GPTQ").needs_imatrix
    assert get("NVFP4").needs_calibration
    assert get("FP8_E4M3").needs_calibration


def test_hardware_gates():
    assert "blackwell" in get("NVFP4").hardware_requirements
    assert "blackwell" not in get("W4A16_GPTQ").hardware_requirements


def test_ids_unique():
    ids = [q.id for q in CATALOG]
    assert len(ids) == len(set(ids)), f"duplicate ids: {[i for i in ids if ids.count(i) > 1]}"


def test_get_raises_on_unknown():
    with pytest.raises(KeyError):
        get("BOGUS_QUANT_ID")
```

- [ ] **Step 2: Run — expect FAIL (module doesn't exist)**

Run: `pytest tests/unit/test_quant_catalog.py -v`
Expected: `ModuleNotFoundError: No module named 'b2cq.quant_catalog'`.

- [ ] **Step 3: Implement `src/b2cq/quant_catalog.py`**

```python
"""Declarative catalog of every quant variant the app can produce.

Adding a new quant here does NOT automatically wire it into a worker; the
worker code needs the format string to know what to invoke. Kept declarative
so the UI can render selection groups without importing the worker code.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel


class QuantFamily(str, Enum):
    GGUF_K = "gguf_k"
    GGUF_I = "gguf_i"
    GGUF_MISC = "gguf_misc"
    GGUF_MMPROJ = "gguf_mmproj"
    SAFETENSORS = "safetensors"


class QuantSpec(BaseModel):
    id: str
    family: QuantFamily
    name: str
    format: str
    needs_calibration: bool
    needs_imatrix: bool = False
    notes: str = ""
    hardware_requirements: list[str] = []


CATALOG: list[QuantSpec] = [
    # ---- GGUF K-quants ----
    QuantSpec(id="Q2_K",     family=QuantFamily.GGUF_K, name="Q2_K",     format="Q2_K",     needs_calibration=False),
    QuantSpec(id="Q3_K_S",   family=QuantFamily.GGUF_K, name="Q3_K_S",   format="Q3_K_S",   needs_calibration=False),
    QuantSpec(id="Q3_K_M",   family=QuantFamily.GGUF_K, name="Q3_K_M",   format="Q3_K_M",   needs_calibration=False),
    QuantSpec(id="Q3_K_L",   family=QuantFamily.GGUF_K, name="Q3_K_L",   format="Q3_K_L",   needs_calibration=False),
    QuantSpec(id="Q4_K_S",   family=QuantFamily.GGUF_K, name="Q4_K_S",   format="Q4_K_S",   needs_calibration=False),
    QuantSpec(id="Q4_K_M",   family=QuantFamily.GGUF_K, name="Q4_K_M",   format="Q4_K_M",   needs_calibration=False),
    QuantSpec(id="Q5_K_S",   family=QuantFamily.GGUF_K, name="Q5_K_S",   format="Q5_K_S",   needs_calibration=False),
    QuantSpec(id="Q5_K_M",   family=QuantFamily.GGUF_K, name="Q5_K_M",   format="Q5_K_M",   needs_calibration=False),
    QuantSpec(id="Q6_K",     family=QuantFamily.GGUF_K, name="Q6_K",     format="Q6_K",     needs_calibration=False),

    # ---- GGUF I-quants (all need imatrix) ----
    QuantSpec(id="IQ1_S",    family=QuantFamily.GGUF_I, name="IQ1_S",    format="IQ1_S",    needs_calibration=True, needs_imatrix=True, notes="Extreme compression; noticeable quality drop"),
    QuantSpec(id="IQ1_M",    family=QuantFamily.GGUF_I, name="IQ1_M",    format="IQ1_M",    needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ2_XXS",  family=QuantFamily.GGUF_I, name="IQ2_XXS",  format="IQ2_XXS",  needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ2_XS",   family=QuantFamily.GGUF_I, name="IQ2_XS",   format="IQ2_XS",   needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ2_S",    family=QuantFamily.GGUF_I, name="IQ2_S",    format="IQ2_S",    needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ2_M",    family=QuantFamily.GGUF_I, name="IQ2_M",    format="IQ2_M",    needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ3_XXS",  family=QuantFamily.GGUF_I, name="IQ3_XXS",  format="IQ3_XXS",  needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ3_XS",   family=QuantFamily.GGUF_I, name="IQ3_XS",   format="IQ3_XS",   needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ3_S",    family=QuantFamily.GGUF_I, name="IQ3_S",    format="IQ3_S",    needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ3_M",    family=QuantFamily.GGUF_I, name="IQ3_M",    format="IQ3_M",    needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ4_XS",   family=QuantFamily.GGUF_I, name="IQ4_XS",   format="IQ4_XS",   needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ4_NL",   family=QuantFamily.GGUF_I, name="IQ4_NL",   format="IQ4_NL",   needs_calibration=True, needs_imatrix=True),

    # ---- GGUF misc (no calibration) ----
    QuantSpec(id="Q8_0",     family=QuantFamily.GGUF_MISC, name="Q8_0",  format="Q8_0",     needs_calibration=False, notes="Reference-quality 8-bit"),
    QuantSpec(id="F16",      family=QuantFamily.GGUF_MISC, name="F16",   format="F16",      needs_calibration=False, notes="Repack; no quantization"),
    QuantSpec(id="BF16",     family=QuantFamily.GGUF_MISC, name="BF16",  format="BF16",     needs_calibration=False, notes="Repack; no quantization"),

    # ---- GGUF mmproj (multimodal projector) ----
    QuantSpec(id="mmproj-f16", family=QuantFamily.GGUF_MMPROJ, name="mmproj-f16", format="mmproj", needs_calibration=False, notes="Vision projector for multimodal models"),

    # ---- safetensors ----
    QuantSpec(id="W4A16_GPTQ", family=QuantFamily.SAFETENSORS, name="W4A16 GPTQ",  format="W4A16_GPTQ", needs_calibration=True, notes="Symmetric g128; vLLM gptq_marlin kernel"),
    QuantSpec(id="W4A16_AWQ",  family=QuantFamily.SAFETENSORS, name="W4A16 AWQ",   format="W4A16_AWQ",  needs_calibration=True, notes="Asymmetric g128; vLLM awq_marlin kernel"),
    QuantSpec(id="NVFP4",      family=QuantFamily.SAFETENSORS, name="NVFP4",       format="NVFP4",      needs_calibration=True, hardware_requirements=["blackwell"], notes="4-bit weights + activations; Blackwell tensor cores only"),
    QuantSpec(id="FP8_E4M3",   family=QuantFamily.SAFETENSORS, name="FP8 E4M3",    format="FP8_E4M3",   needs_calibration=True, notes="8-bit float; Hopper+ compute, Ampere storage"),
    QuantSpec(id="FP8_E5M2",   family=QuantFamily.SAFETENSORS, name="FP8 E5M2",    format="FP8_E5M2",   needs_calibration=True, notes="8-bit float, larger exponent range"),
]

_BY_ID = {q.id: q for q in CATALOG}


def get(quant_id: str) -> QuantSpec:
    if quant_id not in _BY_ID:
        raise KeyError(f"unknown quant id: {quant_id!r}")
    return _BY_ID[quant_id]


def by_family(family: QuantFamily) -> list[QuantSpec]:
    return [q for q in CATALOG if q.family == family]
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/unit/test_quant_catalog.py -v`
Expected: 5/5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/b2cq/quant_catalog.py tests/unit/test_quant_catalog.py
git commit -m "feat: quant catalog (30 variants across 5 families)"
```

---

### Task 4: HF client wrapper with in-memory token

**Files:**
- Create: `src/b2cq/hf_client.py`
- Create: `tests/unit/test_hf_client.py`

**Interfaces:**
- Consumes: `huggingface_hub`
- Produces:
  - `class HFClient`: constructor takes token (str). Methods: `whoami() -> dict`, `download_snapshot(repo_id, local_dir) -> Path`, `upload_folder(repo_id, folder_path, create_if_missing, private) -> str`, `upload_file(repo_id, file_path, path_in_repo) -> str`, `update_file(repo_id, path_in_repo, content, commit_message) -> str`, `close() -> None` (wipes token from memory).
  - Token stored as an instance attribute cleared on `close()`; never logged; never returned via `__repr__`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_hf_client.py`:
```python
"""HFClient thin wrapper — token handling + method signatures."""
from unittest.mock import MagicMock, patch
import pytest
from b2cq.hf_client import HFClient


def test_repr_never_leaks_token():
    c = HFClient(token="hf_supersecret_xyz")
    r = repr(c)
    assert "hf_supersecret" not in r
    assert "xyz" not in r


def test_close_wipes_token():
    c = HFClient(token="hf_supersecret")
    assert c._token is not None
    c.close()
    assert c._token is None


def test_calls_after_close_raise():
    c = HFClient(token="hf_supersecret")
    c.close()
    with pytest.raises(RuntimeError, match="closed"):
        c.whoami()


@patch("b2cq.hf_client.HfApi")
def test_whoami_forwards_token(HfApi):
    inst = MagicMock()
    inst.whoami.return_value = {"name": "eddie"}
    HfApi.return_value = inst
    c = HFClient(token="hf_x")
    result = c.whoami()
    HfApi.assert_called_once_with(token="hf_x")
    assert result == {"name": "eddie"}


@patch("b2cq.hf_client.HfApi")
def test_upload_folder_forwards_args(HfApi):
    inst = MagicMock()
    inst.upload_folder.return_value = "https://huggingface.co/user/repo/commit/abc"
    HfApi.return_value = inst
    c = HFClient(token="hf_x")
    url = c.upload_folder("user/repo", "/local/dir", create_if_missing=True, private=False)
    assert url == "https://huggingface.co/user/repo/commit/abc"
    inst.create_repo.assert_called_once_with(repo_id="user/repo", private=False, exist_ok=True)
    inst.upload_folder.assert_called_once()
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/unit/test_hf_client.py -v`
Expected: ImportError on `b2cq.hf_client`.

- [ ] **Step 3: Implement `src/b2cq/hf_client.py`**

```python
"""Thin wrapper around huggingface_hub with careful token handling.

Token is held only in an instance attribute — cleared on close(). Never
logged, never included in __repr__, never written to disk. Instance
methods forward operations to a fresh HfApi with the token attached; no
long-lived HfApi is stored (each call constructs one) so revocation via
close() is immediate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


class HFClient:
    def __init__(self, token: str):
        if not token or not token.startswith(("hf_", "hf-")):
            raise ValueError("HF token must start with 'hf_' or 'hf-'")
        self._token: str | None = token

    def __repr__(self) -> str:
        state = "closed" if self._token is None else "open"
        return f"<HFClient state={state}>"

    def close(self) -> None:
        self._token = None

    def _api(self) -> HfApi:
        if self._token is None:
            raise RuntimeError("HFClient is closed")
        return HfApi(token=self._token)

    def whoami(self) -> dict[str, Any]:
        return self._api().whoami()

    def download_snapshot(self, repo_id: str, local_dir: str | Path) -> Path:
        from huggingface_hub import snapshot_download
        p = snapshot_download(repo_id=repo_id, local_dir=str(local_dir), token=self._token)
        return Path(p)

    def upload_folder(self, repo_id: str, folder_path: str | Path, *,
                      create_if_missing: bool = True, private: bool = False,
                      commit_message: str | None = None) -> str:
        api = self._api()
        if create_if_missing:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        return api.upload_folder(
            repo_id=repo_id,
            folder_path=str(folder_path),
            commit_message=commit_message or "B2CQuantizer: upload quantized weights",
        )

    def upload_file(self, repo_id: str, file_path: str | Path, path_in_repo: str, *,
                    create_if_missing: bool = True, private: bool = False,
                    commit_message: str | None = None) -> str:
        api = self._api()
        if create_if_missing:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        return api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            commit_message=commit_message or f"B2CQuantizer: upload {path_in_repo}",
        )

    def update_file(self, repo_id: str, path_in_repo: str, content: str,
                    commit_message: str) -> str:
        """Upload a text file (README, etc.) directly from a string."""
        import io
        api = self._api()
        return api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=io.BytesIO(content.encode("utf-8")),
            path_in_repo=path_in_repo,
            commit_message=commit_message,
        )
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/unit/test_hf_client.py -v`
Expected: 5/5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/b2cq/hf_client.py tests/unit/test_hf_client.py
git commit -m "feat: HFClient with in-memory token, close() wipes it"
```

---

### Task 5: Calibration loader

**Files:**
- Create: `src/b2cq/calibration.py`
- Create: `src/b2cq_data/calibration/bundled.jsonl` (or a script to generate it at image-build time)
- Create: `tests/unit/test_calibration.py`

**Interfaces:**
- Consumes: `datasets` (for HF dataset id path), local file (for upload path), bundled corpus (for default)
- Produces: `load_calibration(source: CalibrationSource) -> list[dict]` returning ~500-1024 `{"messages": [...]}` samples.
  - `class CalibrationSource(BaseModel)`: `type: Literal["bundled", "upload", "hf_dataset"]`, plus one of `local_path: Path`, `hf_dataset_id: str`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_calibration.py`:
```python
"""Calibration loader — all three sources produce a list of message dicts."""
import json, pytest
from pathlib import Path
from b2cq.calibration import load_calibration, CalibrationSource


@pytest.fixture
def upload_jsonl(tmp_path) -> Path:
    p = tmp_path / "cal.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(50):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"msg {i}"}]}) + "\n")
    return p


def test_bundled_loads_default():
    samples = load_calibration(CalibrationSource(type="bundled"))
    assert isinstance(samples, list)
    assert len(samples) >= 500
    assert all("messages" in s for s in samples)


def test_upload_loads_from_file(upload_jsonl):
    samples = load_calibration(CalibrationSource(type="upload", local_path=upload_jsonl))
    assert len(samples) == 50
    assert samples[0]["messages"][0]["content"] == "msg 0"


def test_upload_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibration(CalibrationSource(type="upload", local_path=tmp_path / "nonexistent.jsonl"))
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/unit/test_calibration.py -v`
Expected: ImportError.

- [ ] **Step 3: Create bundled corpus generator**

Create `scripts/build_bundled_calibration.py` (invoked at Docker build time, or manually):
```python
"""Build the bundled calibration corpus at Docker-build time.

1000 samples split ~600 wikitext + ~400 openassistant-oasst1 (chat-style).
Saved as OpenAI-messages JSONL to src/b2cq_data/calibration/bundled.jsonl.
"""
import json
from pathlib import Path
from datasets import load_dataset

OUT = Path(__file__).parent.parent / "src/b2cq_data/calibration/bundled.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

samples = []

# 600 wikitext samples as single-turn "user says X" for language coverage.
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
for row in wiki:
    text = row["text"].strip()
    if len(text) > 200:
        samples.append({"messages": [{"role": "user", "content": text[:1500]}]})
    if len(samples) >= 600:
        break

# 400 conversational samples from a small open corpus.
oasst = load_dataset("OpenAssistant/oasst1", split="train")
for row in oasst:
    if row["role"] == "prompter" and row["parent_id"] is None:
        samples.append({"messages": [{"role": "user", "content": row["text"][:1500]}]})
    if len(samples) >= 1000:
        break

with OUT.open("w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Wrote {len(samples)} bundled calibration samples to {OUT}")
```

Add to Dockerfile before the `COPY src/` line:
```dockerfile
COPY scripts/build_bundled_calibration.py /tmp/build_cal.py
RUN python3 /tmp/build_cal.py && rm /tmp/build_cal.py
```

For local testing without Docker, run once by hand: `python scripts/build_bundled_calibration.py`.

- [ ] **Step 4: Implement `src/b2cq/calibration.py`**

```python
"""Calibration data loader — bundled / upload / HF-dataset sources.

All three sources return a `list[dict]` where each dict has a "messages"
key. Downstream consumers (llm-compressor recipes, llama-imatrix input
prep) render that shape to whatever they need — plain text concatenation
for imatrix, tokenized prompts for llm-compressor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


class CalibrationSource(BaseModel):
    type: Literal["bundled", "upload", "hf_dataset"]
    local_path: Optional[Path] = None
    hf_dataset_id: Optional[str] = None
    hf_token: Optional[str] = None


_BUNDLED_PATH = Path(__file__).parent.parent / "b2cq_data/calibration/bundled.jsonl"


def load_calibration(source: CalibrationSource) -> list[dict]:
    if source.type == "bundled":
        return _load_jsonl(_BUNDLED_PATH)
    if source.type == "upload":
        if source.local_path is None:
            raise ValueError("upload source requires local_path")
        return _load_jsonl(source.local_path)
    if source.type == "hf_dataset":
        return _load_hf(source.hf_dataset_id, source.hf_token)
    raise ValueError(f"unknown calibration source type: {source.type}")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"calibration file not found: {path}")
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _load_hf(dataset_id: str, token: str | None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split="train", token=token)
    out = []
    for row in ds:
        if "messages" in row:
            out.append({"messages": row["messages"]})
        elif "text" in row:
            out.append({"messages": [{"role": "user", "content": row["text"]}]})
    return out
```

- [ ] **Step 5: Run — expect PASS**

Run: `pytest tests/unit/test_calibration.py -v`
Expected: 3/3 PASS. If the bundled test fails because `bundled.jsonl` doesn't exist locally, run `python scripts/build_bundled_calibration.py` first.

- [ ] **Step 6: Commit**

```bash
git add src/b2cq/calibration.py scripts/build_bundled_calibration.py tests/unit/test_calibration.py Dockerfile
git commit -m "feat: calibration loader (bundled/upload/HF dataset)"
```

---

### Task 6: Progress bus + job model

**Files:**
- Create: `src/b2cq/progress.py`, `src/b2cq/job_model.py`
- Create: `tests/unit/test_progress.py`, `tests/unit/test_job_model.py`

**Interfaces:**
- `ProgressBus`: async pub/sub keyed by job id. `publish(job_id, event: dict)`, `subscribe(job_id) -> AsyncIterator[dict]`.
- `class QuantStatus(str, Enum)`: `PENDING`, `RUNNING`, `UPLOADING`, `DONE`, `FAILED`, `SKIPPED`
- `class QuantResult(BaseModel)`: `quant_id`, `status`, `lane` (`"A"` or `"B"`), `started_at`, `finished_at`, `elapsed_seconds`, `output_size_bytes`, `error`, `upload_url`, `log_tail: list[str]` (last 100 lines)
- `class Job(BaseModel)`: `id: str`, `source_model: str`, `owner: str`, `quants: list[QuantResult]`, `calibration: CalibrationSource`, `private: bool`, `update_source_readme: bool`, `started_at`, `finished_at`, `status: Literal["pending","running","complete","failed"]`
- `class JobStore`: `create(...) -> Job`, `get(job_id) -> Job`, `list() -> list[Job]`, in-memory only.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_progress.py`:
```python
"""ProgressBus: async pub/sub, per-job channels."""
import asyncio
import pytest
from b2cq.progress import ProgressBus


@pytest.mark.asyncio
async def test_subscribe_receives_published_events():
    bus = ProgressBus()
    received = []
    async def consumer():
        async for evt in bus.subscribe("job1"):
            received.append(evt)
            if len(received) == 2:
                break
    consumer_task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    await bus.publish("job1", {"type": "log", "msg": "hello"})
    await bus.publish("job1", {"type": "status", "quant": "Q4_K_M", "status": "running"})
    await consumer_task
    assert received == [
        {"type": "log", "msg": "hello"},
        {"type": "status", "quant": "Q4_K_M", "status": "running"},
    ]


@pytest.mark.asyncio
async def test_channels_isolated():
    bus = ProgressBus()
    got = []
    async def consumer():
        async for evt in bus.subscribe("jobA"):
            got.append(evt); break
    t = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    await bus.publish("jobB", {"m": "wrong"})
    await bus.publish("jobA", {"m": "right"})
    await t
    assert got == [{"m": "right"}]
```

Create `tests/unit/test_job_model.py`:
```python
"""Job / QuantResult / JobStore basics."""
from b2cq.job_model import Job, QuantResult, QuantStatus, JobStore
from b2cq.calibration import CalibrationSource


def test_jobstore_create_and_get():
    store = JobStore()
    job = store.create(
        source_model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        owner="bigblueceiling",
        quants=[QuantResult(quant_id="Q4_K_M", status=QuantStatus.PENDING, lane="B")],
        calibration=CalibrationSource(type="bundled"),
        private=False,
        update_source_readme=True,
    )
    assert store.get(job.id) is job
    assert store.list() == [job]
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/unit/test_progress.py tests/unit/test_job_model.py -v`
Expected: ImportError on both modules.

- [ ] **Step 3: Implement `src/b2cq/progress.py`**

```python
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
```

- [ ] **Step 4: Implement `src/b2cq/job_model.py`**

```python
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
```

- [ ] **Step 5: Run — expect PASS**

Run: `pytest tests/unit/test_progress.py tests/unit/test_job_model.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/b2cq/progress.py src/b2cq/job_model.py tests/unit/test_progress.py tests/unit/test_job_model.py
git commit -m "feat: progress bus (async pub/sub) + job model + in-memory store"
```

---

### Task 7: safetensors quant worker (llm-compressor)

**Files:**
- Create: `src/b2cq/workers/__init__.py`, `src/b2cq/workers/safetensors.py`
- Create: `tests/integration/test_safetensors_worker.py`

**Interfaces:**
- Consumes: `llm-compressor`, `transformers`, `torch`, `mistral-common` (pinned per Global Constraints); calibration data (list of message dicts, optionally with tools per entry); source model (already downloaded on disk); a `QuantSpec` with format in `{W4A16_GPTQ, W4A16_AWQ, NVFP4, FP8_E4M3, FP8_E5M2}`
- Produces:
  - `def quantize_safetensors(model, tokenizer, format: str, calibration: list[dict], output_dir: Path, source_dir: Path, log_cb) -> None` — Loads model ONCE per lane; the caller keeps the loaded model across formats. `source_dir` is required so the worker can detect FrndoBrain-class artifacts (tekken.json presence) and pick the right calibration render path per SPEC §14.
  - `def load_model_for_safetensors(source_dir: Path, log_cb) -> tuple[model, tokenizer]` — separate helper so Lane A can call it once and pass the result into each per-format quantize call.
  - `def is_frndobrain_class(source_dir: Path) -> bool` — detection helper.
  - `def build_frndobrain_tokenizer(source_dir: Path) -> MistralTokenizer` — pinned + version-checked + attribute-flip-applied.

**Critical Mistral3 handling** (do NOT skip these — they are the reason your existing `frndobrain_quantize.py` works):

1. **Architecture detection.** Read `<source_dir>/config.json`'s `architectures[0]`. If `"Mistral3ForConditionalGeneration"`, load via `AutoModelForImageTextToText` and use multimodal ignore patterns + scoped AWQ mappings. If `"MistralForCausalLM"`, load via `AutoModelForCausalLM` and use minimal ignore patterns + default AWQ mappings.

2. **Ignore patterns** (multimodal path — pass to `llm-compressor` recipe as the `ignore` list):
   ```python
   ignore_multimodal = [
       "re:.*vision_tower.*",
       "re:.*multi_modal_projector.*",
       "lm_head",
   ]
   ignore_text_only = ["lm_head"]
   ```

3. **AWQ scoped mappings** (multimodal path only — `llm-compressor`'s default `AWQMapping` list globs decoder-layer patterns across all 40 layers into one `match_modules_set` group and errors out on Mistral3 wrapped models). Build one `AWQMapping` per decoder layer explicitly:
   ```python
   from llmcompressor.modifiers.transform.awq import AWQMapping
   NUM_DECODER_LAYERS = model.config.text_config.num_hidden_layers  # 40 for Mistral-Small-3.2-24B
   mappings = []
   for i in range(NUM_DECODER_LAYERS):
       prefix = f"language_model.model.layers.{i}"
       # Attention smoothing block
       mappings.append(AWQMapping(smooth_layer=f"{prefix}.input_layernorm",
                                   balance_layers=[f"{prefix}.self_attn.q_proj",
                                                   f"{prefix}.self_attn.k_proj",
                                                   f"{prefix}.self_attn.v_proj"]))
       # MLP smoothing block
       mappings.append(AWQMapping(smooth_layer=f"{prefix}.post_attention_layernorm",
                                   balance_layers=[f"{prefix}.mlp.gate_proj",
                                                   f"{prefix}.mlp.up_proj"]))
       # Attention output balance
       mappings.append(AWQMapping(smooth_layer=f"{prefix}.self_attn.v_proj",
                                   balance_layers=[f"{prefix}.self_attn.o_proj"]))
       # MLP output balance
       mappings.append(AWQMapping(smooth_layer=f"{prefix}.mlp.up_proj",
                                   balance_layers=[f"{prefix}.mlp.down_proj"]))
   ```
   For text-only Mistral, `llm-compressor`'s default AWQ mappings work — do not build custom.

4. **Recipe per format:**
   - **W4A16_GPTQ:** `GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore, group_size=128, sym=True)`
   - **W4A16_AWQ:** For multimodal, use `AWQModifier(ignore=ignore, mappings=<scoped_mappings>) + QuantizationModifier(config_groups={"group_0": QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(num_bits=4, type=QuantizationType.INT, symmetric=False, strategy=QuantizationStrategy.GROUP, group_size=128, actorder=None))})`. For text-only, `AWQModifier(ignore=ignore) + <same QuantizationModifier>` — leaving `mappings` unset uses defaults.
   - **NVFP4:** `QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)` — no GPTQ or AWQ; FP4 rounding doesn't benefit from Hessian/activation smoothing.
   - **FP8_E4M3 / FP8_E5M2:** `QuantizationModifier(targets="Linear", scheme="FP8", ignore=ignore, weights=QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT, symmetric=True, strategy=QuantizationStrategy.CHANNEL))`. E4M3 vs E5M2 differ in the float format; pass via `weights.observer_options={"dtype": "float8_e4m3fn"}` or `"float8_e5m2"` (verify exact param name against installed `llm-compressor` version).

5. **Calibration:** feed `llm-compressor`'s `oneshot(...)` runner with the calibration data pre-tokenized by the source model's tokenizer (using `mistral-common` if the source is a FrndoBrain-style artifact, else the source's shipped tokenizer). Sample count: 512.

6. **After save:** copy `tekken.json`, `params.json`, and `chat_template.jinja` from the source dir into the output dir if present — llm-compressor doesn't carry these forward but downstream vLLM serving needs them for FrndoBrain artifacts.

- [ ] **Step 1: Write the (integration) test**

Create `tests/integration/test_safetensors_worker.py`:
```python
"""GPU integration test — skipped without --gpu marker.

Uses a tiny stub model (TinyLlama or a synthetic Mistral3-shaped test model)
so the test runs in reasonable time. Full-model verification is manual
via the smoke procedure (tests/smoke/README.md)."""
import pytest
import os
from pathlib import Path

pytestmark = pytest.mark.skipif(
    "GPU_INTEGRATION" not in os.environ,
    reason="Set GPU_INTEGRATION=1 to run GPU integration tests",
)


def test_quantize_w4a16_gptq_smoke(tmp_path):
    from b2cq.workers.safetensors import load_model_for_safetensors, quantize_safetensors
    # Use a tiny known-Mistral model for smoke — swap for a permanently
    # available small mistral-family model when this test is enabled.
    source = Path("/workspace/hf-cache/mistralai/Mistral-7B-v0.1-tiny")  # fixture-provided
    if not source.exists():
        pytest.skip("tiny Mistral fixture not available")

    logs = []
    model, tok = load_model_for_safetensors(source, log_cb=logs.append)
    output = tmp_path / "gptq"
    quantize_safetensors(model, tok, "W4A16_GPTQ",
                         calibration=[{"messages": [{"role": "user", "content": "hi"}]}] * 8,
                         output_dir=output, log_cb=logs.append)
    assert (output / "config.json").exists()
    assert any(f.name.endswith(".safetensors") for f in output.iterdir())
```

- [ ] **Step 2: Implement `src/b2cq/workers/safetensors.py`**

Follow the "Critical Mistral3 handling" list above. Structure:

```python
"""safetensors quantization worker via llm-compressor."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch

_MULTIMODAL_ARCH = "Mistral3ForConditionalGeneration"
_TEXT_ARCH = "MistralForCausalLM"

_IGNORE_MULTIMODAL = ["re:.*vision_tower.*", "re:.*multi_modal_projector.*", "lm_head"]
_IGNORE_TEXT = ["lm_head"]


def _detect_arch(source_dir: Path) -> str:
    cfg = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    arch = cfg.get("architectures", [None])[0]
    if arch not in (_MULTIMODAL_ARCH, _TEXT_ARCH):
        raise ValueError(f"Unsupported architecture: {arch!r}. B2CQuantizer v1 supports only Mistral/Mistral3.")
    return arch


def load_model_for_safetensors(source_dir: Path, log_cb: Callable[[str], None]):
    arch = _detect_arch(source_dir)
    log_cb(f"Detected architecture: {arch}")
    from transformers import AutoTokenizer
    if arch == _MULTIMODAL_ARCH:
        from transformers import AutoModelForImageTextToText as ModelClass
    else:
        from transformers import AutoModelForCausalLM as ModelClass
    model = ModelClass.from_pretrained(source_dir, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    log_cb(f"Loaded {arch} on {next(model.parameters()).device}")
    return model, tokenizer


def _build_awq_mappings_scoped(model):
    """Explicit per-decoder-layer AWQ mappings for Mistral3ForConditionalGeneration.

    llm-compressor's default mappings glob across all decoder layers into
    one match_modules_set group and error out with 'AWQ needs to match a
    single smoothlayer per set'. This function builds one mapping set per
    decoder layer explicitly.
    """
    from llmcompressor.modifiers.transform.awq import AWQMapping
    n_layers = model.config.text_config.num_hidden_layers
    mappings = []
    for i in range(n_layers):
        prefix = f"language_model.model.layers.{i}"
        mappings.append(AWQMapping(smooth_layer=f"{prefix}.input_layernorm",
                                   balance_layers=[f"{prefix}.self_attn.q_proj",
                                                   f"{prefix}.self_attn.k_proj",
                                                   f"{prefix}.self_attn.v_proj"]))
        mappings.append(AWQMapping(smooth_layer=f"{prefix}.post_attention_layernorm",
                                   balance_layers=[f"{prefix}.mlp.gate_proj",
                                                   f"{prefix}.mlp.up_proj"]))
        mappings.append(AWQMapping(smooth_layer=f"{prefix}.self_attn.v_proj",
                                   balance_layers=[f"{prefix}.self_attn.o_proj"]))
        mappings.append(AWQMapping(smooth_layer=f"{prefix}.mlp.up_proj",
                                   balance_layers=[f"{prefix}.mlp.down_proj"]))
    return mappings


def _build_recipe(format: str, arch: str, model):
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    ignore = _IGNORE_MULTIMODAL if arch == _MULTIMODAL_ARCH else _IGNORE_TEXT

    if format == "W4A16_GPTQ":
        return [GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore, group_size=128, sym=True)]

    if format == "W4A16_AWQ":
        awq_kwargs = {"ignore": ignore}
        if arch == _MULTIMODAL_ARCH:
            awq_kwargs["mappings"] = _build_awq_mappings_scoped(model)
        return [
            AWQModifier(**awq_kwargs),
            QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=ignore, group_size=128),
        ]

    if format == "NVFP4":
        return [QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)]

    if format in ("FP8_E4M3", "FP8_E5M2"):
        # llm-compressor's FP8 scheme naming — verify against installed version.
        # As of 0.3.x, "FP8" default is E4M3; explicit E5M2 requires custom weights args.
        scheme = "FP8" if format == "FP8_E4M3" else "FP8_E5M2"
        return [QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)]

    raise ValueError(f"unsupported safetensors format: {format!r}")


def is_frndobrain_class(source_dir: Path) -> bool:
    """Detect FrndoBrain-class artifact by presence of tekken.json in source.

    FrndoBrain merges carry tekken.json byte-identical to the base Mistral repo
    per the tokenizer-invariant training contract. Its presence marks an artifact
    whose calibration MUST go through mistral-common with the tool-placement flip.
    See SPEC §14 for the full invariant set.
    """
    return (source_dir / "tekken.json").exists()


# Pin locked to what the FrndoBrain serving stack ships. The private attribute
# below is version-fragile — it was `_user_message_position_to_encode_tools` in
# mistral-common 1.9.x and renamed to `_message_position_to_encode_tools_settings`
# in 1.11.x. Silent divergence between calibration-time and serving-time
# tokenization gets baked into the quantized weights; hence the hard-fails.
_EXPECTED_MISTRAL_COMMON = "1.11.5"
_TOOL_PLACEMENT_ATTR = "_message_position_to_encode_tools_settings"


def build_frndobrain_tokenizer(source_dir: Path):
    """Build a mistral-common MistralTokenizer with the tools-at-first-user-turn
    flip, pinned to the version the serving stack ships. Fails loudly if
    version or attribute has drifted."""
    import mistral_common
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.tokens.tokenizers.base import UserMessagePosition
    from mistral_common.protocol.instruct.validator import ValidationMode

    if mistral_common.__version__ != _EXPECTED_MISTRAL_COMMON:
        raise RuntimeError(
            f"mistral-common version mismatch: got {mistral_common.__version__}, "
            f"expected {_EXPECTED_MISTRAL_COMMON}. FrndoBrain-class calibration "
            "is version-locked to match training + serving; audit the InstructTokenizer "
            "source and update the pin (see SPEC §14 for the drift history)."
        )
    # Load from the local source_dir (tekken.json lives here).
    tok = MistralTokenizer.from_file(str(source_dir / "tekken.json"),
                                      mode=ValidationMode.serving)  # user-approved change; serving-mode rendering = serving-time tokenization, and accepts user-terminated calibration samples
    it = tok.instruct_tokenizer
    if not hasattr(it, _TOOL_PLACEMENT_ATTR):
        raise RuntimeError(
            f"expected attribute {_TOOL_PLACEMENT_ATTR!r} not present on "
            f"{type(it).__name__}; attribute has moved again — audit InstructTokenizer "
            "source and update the pin."
        )
    setattr(it, _TOOL_PLACEMENT_ATTR, UserMessagePosition.first)
    return tok


def _render_calibration_frndobrain(mtok, sample: dict) -> str:
    """Render one calibration sample via mistral-common. Sample shape is
    {messages: [...], tools: [...] or omitted}. Returns plain text — llm-compressor
    will re-tokenize with the model's HF tokenizer, which for FrndoBrain artifacts
    produces the same IDs since tokenizer.json is regenerated from tekken.json
    at training time. What matters is the CONVERSATION STRUCTURE is rendered by
    mistral-common (so [AVAILABLE_TOOLS] lands at first user turn, tool_calls
    render with V11's [CALL_ID] format, etc.)."""
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    req = ChatCompletionRequest(messages=sample["messages"], tools=sample.get("tools"))
    return mtok.encode_chat_completion(req).text


def _render_calibration_hf(tokenizer, sample: dict) -> str:
    """Fallback render for non-FrndoBrain Mistral models. Uses HF chat template."""
    return tokenizer.apply_chat_template(sample["messages"], tokenize=False)


def quantize_safetensors(model, tokenizer, format: str, calibration: list[dict],
                         output_dir: Path, source_dir: Path,
                         log_cb: Callable[[str], None]) -> None:
    from llmcompressor.transformers import oneshot

    arch = model.config.architectures[0]
    recipe = _build_recipe(format, arch, model)

    # Detect artifact class and pick the calibration render path.
    if is_frndobrain_class(source_dir):
        log_cb(f"FrndoBrain-class artifact detected (tekken.json present in {source_dir})")
        log_cb("Calibration rendering via mistral-common with tools-at-first-user-turn flip")
        mtok = build_frndobrain_tokenizer(source_dir)
        # If ANY calibration sample fails validation (e.g., non-assistant-terminated),
        # fail loudly — the user's calibration corpus does not match the training
        # invariant and would produce calibration/inference divergence.
        try:
            texts = [_render_calibration_frndobrain(mtok, s) for s in calibration]
        except Exception as e:
            raise RuntimeError(
                f"FrndoBrain-class calibration failed to render at least one sample "
                f"via mistral-common: {type(e).__name__}: {e}. This means your "
                "calibration corpus does not match the training/serving invariant. "
                "For best results, calibrate on the same data (or a same-shape subset) "
                "as was used for training. See SPEC §14 for the invariants that must hold."
            ) from e
    else:
        log_cb(f"Generic Mistral artifact (no tekken.json in {source_dir})")
        log_cb("Calibration rendering via HF chat template")
        texts = [_render_calibration_hf(tokenizer, s) for s in calibration]

    log_cb(f"Prepared {len(texts)} calibration samples for {format}")
    log_cb(f"Starting oneshot quantization: {format}")
    oneshot(
        model=model,
        recipe=recipe,
        dataset=texts,
        output_dir=str(output_dir),
        num_calibration_samples=len(texts),
        max_seq_length=2048,
    )

    # Carry forward Mistral-native tokenizer artifacts if present. llm-compressor
    # doesn't propagate these; downstream vLLM serving needs them for FrndoBrain.
    from shutil import copy
    for name in ("tekken.json", "params.json", "chat_template.jinja"):
        src = source_dir / name
        if src.exists():
            copy(src, output_dir / name)
            log_cb(f"Carried forward {name}")

    log_cb(f"Wrote {format} quant to {output_dir}")
```

Note: the caller (Lane A in the orchestrator, Task 10) MUST pass `source_dir` — the local path to the downloaded source model — into `quantize_safetensors`. The signature change from the earlier draft is intentional: source_dir is where we detect FrndoBrain-class-ness and where we pull tekken.json from for the calibration tokenizer. Update the Task 10 call site accordingly.

- [ ] **Step 3: Verify locally**

If GPU available with a small Mistral-family model cached: `GPU_INTEGRATION=1 pytest tests/integration/test_safetensors_worker.py -v`

If not available: mark this task complete based on code review + note in `tests/smoke/README.md` that this worker needs full E2E validation against a real 7B or 24B model on RunPod.

- [ ] **Step 4: Commit**

```bash
git add src/b2cq/workers/ tests/integration/test_safetensors_worker.py
git commit -m "feat: safetensors quant worker (W4A16 GPTQ/AWQ, NVFP4, FP8) with Mistral3 scoped mappings"
```

---

### Task 8: GGUF conversion + K-quants + misc

**Files:**
- Create: `src/b2cq/workers/gguf_convert.py`, `src/b2cq/workers/gguf_quantize.py`

**Interfaces:**
- `def convert_to_bf16_gguf(source_dir: Path, output_gguf: Path, log_cb) -> None` — wraps `python3 /opt/llama.cpp/convert_hf_to_gguf.py <source> --outtype bf16 --outfile <output>`.
- `def gguf_quantize(bf16_gguf: Path, output_gguf: Path, format: str, log_cb) -> None` — wraps `/opt/llama.cpp/build/bin/llama-quantize <bf16> <out> <FORMAT>`.

- [ ] **Step 1: Implement `src/b2cq/workers/gguf_convert.py`**

```python
"""BF16 GGUF conversion (source safetensors -> BF16 GGUF)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

CONVERT_SCRIPT = "/opt/llama.cpp/convert_hf_to_gguf.py"


def convert_to_bf16_gguf(source_dir: Path, output_gguf: Path, log_cb: Callable[[str], None]) -> None:
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", CONVERT_SCRIPT, str(source_dir), "--outtype", "bf16",
           "--outfile", str(output_gguf)]
    log_cb(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log_cb(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"convert_hf_to_gguf.py failed with exit {proc.returncode}")
    if not output_gguf.exists() or output_gguf.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"BF16 GGUF suspiciously small or missing: {output_gguf}")
    log_cb(f"BF16 GGUF written: {output_gguf} ({output_gguf.stat().st_size / 2**30:.1f} GiB)")
```

- [ ] **Step 2: Implement `src/b2cq/workers/gguf_quantize.py`**

```python
"""GGUF quantize wrapper (llama-quantize)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

QUANTIZE_BIN = "/opt/llama.cpp/build/bin/llama-quantize"
IMATRIX_BIN = "/opt/llama.cpp/build/bin/llama-imatrix"


def gguf_quantize(bf16_gguf: Path, output_gguf: Path, format: str,
                  log_cb: Callable[[str], None], imatrix: Path | None = None) -> None:
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [QUANTIZE_BIN]
    if imatrix is not None:
        cmd += ["--imatrix", str(imatrix)]
    cmd += [str(bf16_gguf), str(output_gguf), format]
    log_cb(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log_cb(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"llama-quantize failed with exit {proc.returncode}")
    if not output_gguf.exists():
        raise RuntimeError(f"llama-quantize output missing: {output_gguf}")
    log_cb(f"Wrote {format} GGUF: {output_gguf} ({output_gguf.stat().st_size / 2**30:.2f} GiB)")


def compute_imatrix(bf16_gguf: Path, calibration_text: Path, output_imatrix: Path,
                    log_cb: Callable[[str], None], n_chunks: int = 100) -> None:
    output_imatrix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [IMATRIX_BIN, "-m", str(bf16_gguf), "-f", str(calibration_text),
           "-o", str(output_imatrix), "--chunks", str(n_chunks)]
    log_cb(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log_cb(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"llama-imatrix failed with exit {proc.returncode}")
    log_cb(f"Wrote imatrix: {output_imatrix}")
```

- [ ] **Step 3: Add helper for calibration -> plain-text file** (needed by `compute_imatrix`)

In `src/b2cq/calibration.py`, add:
```python
def to_plaintext(samples: list[dict], output_path: Path) -> Path:
    """Render calibration samples as concatenated plain text for llama-imatrix."""
    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            for m in s.get("messages", []):
                f.write((m.get("content") or "") + "\n\n")
    return output_path
```

- [ ] **Step 4: Commit**

```bash
git add src/b2cq/workers/gguf_convert.py src/b2cq/workers/gguf_quantize.py src/b2cq/calibration.py
git commit -m "feat: GGUF conversion + llama-quantize + imatrix wrappers"
```

---

### Task 9: mmproj export with runtime llama.cpp patch

**Files:**
- Modify: `scripts/patch_convert_hf_to_gguf.py` (fill in the two Mistral3-specific fixes)
- Create: `src/b2cq/workers/mmproj.py`

**Interfaces:**
- `def export_mmproj(source_dir: Path, output_gguf: Path, log_cb) -> None` — applies the patch script (idempotent) then runs `convert_hf_to_gguf.py <source> --mmproj --outtype f16 --outfile <output>`, then asserts `v.token_embd.img_break` is in the output tensor list.

**Bug context (from operator's unsloth repo CLAUDE.md gotcha #9):** llama.cpp's `convert_hf_to_gguf.py` has two upstream bugs affecting Mistral3 / pixtral mmproj export:
1. `MmprojModel.filter_tensors` drops every tensor whose name contains `"language_model."`. Mistral3's LM `embed_tokens` lives at `language_model.model.embed_tokens.weight`, so the `[IMG_BREAK]` extraction in `LlavaVisionModel.modify_tensors` never sees it and the `v.token_embd.img_break` tensor is silently missing (222 tensors output instead of 223 — LM Studio "failed to load model").
2. The pixtral activation flag (`clip.use_silu` / `clip.use_gelu`) is the *projector's* activation, not the encoder's. Recent code reads `hparams["hidden_act"]` (encoder, silu/SwiGLU) and emits `use_silu=True`. The projector for Mistral3 is gelu (`projector_hidden_act` at the top level of `config.json`), so the flag must be `use_gelu=True`.

- [ ] **Step 1: Fill in `scripts/patch_convert_hf_to_gguf.py`**

```python
"""Runtime patch for llama.cpp's convert_hf_to_gguf.py.

Fixes two Mistral3 / pixtral mmproj export bugs. Idempotent — safe to
re-run. Applied on-demand by the mmproj worker before invoking the
converter, so we don't need to fork llama.cpp.
"""
from __future__ import annotations

import re
from pathlib import Path

CONVERT_PY = Path("/opt/llama.cpp/convert_hf_to_gguf.py")


def apply_patches() -> None:
    if not CONVERT_PY.exists():
        raise FileNotFoundError(f"llama.cpp converter not found at {CONVERT_PY}")
    src = CONVERT_PY.read_text(encoding="utf-8")
    original = src

    # Patch 1: MmprojModel.filter_tensors — allow language_model.model.embed_tokens through
    marker1 = "# --- B2CQ patch: allow LM embed for [IMG_BREAK] extraction ---"
    if marker1 not in src:
        # Locate the "language_model." exclusion line in MmprojModel.filter_tensors
        # and add an exemption for embed_tokens BEFORE the return-False branch.
        pattern1 = re.compile(
            r"(class MmprojModel.*?def filter_tensors.*?)(if\s+\"language_model\.\"\s+in\s+name\s*:\s*\n\s*return\s+False)",
            re.DOTALL,
        )
        replacement1 = (
            r"\1" + marker1 + "\n"
            r"        if 'language_model.model.embed_tokens' in name:\n"
            r"            return True\n"
            r"        \2"
        )
        src, n = pattern1.subn(replacement1, src, count=1)
        assert n == 1, "Patch 1 failed to apply — file structure may have changed"

    # Patch 2: pixtral activation flag — use projector_hidden_act instead of hparams["hidden_act"]
    marker2 = "# --- B2CQ patch: pixtral projector activation ---"
    if marker2 not in src:
        # Locate the clip.use_silu / clip.use_gelu emission for pixtral model class.
        pattern2 = re.compile(
            r"(class LlavaVisionModel.*?)(self\.gguf_writer\.add_bool\(\"clip\.use_silu\",\s*)(self\.hparams\[[\"']hidden_act[\"']\]\s*==\s*[\"']silu[\"'])(.*?add_bool\(\"clip\.use_gelu\",\s*)(self\.hparams\[[\"']hidden_act[\"']\]\s*==\s*[\"']gelu[\"'])",
            re.DOTALL,
        )
        replacement2 = (
            r"\1" + marker2 + "\n        "
            r"_proj_act = self.hparams.get('projector_hidden_act', self.hparams.get('hidden_act', 'gelu'))\n        "
            r"\g<2>_proj_act == 'silu'\g<4>_proj_act == 'gelu'"
        )
        src, n = pattern2.subn(replacement2, src, count=1)
        assert n == 1, "Patch 2 failed to apply — file structure may have changed"

    if src != original:
        CONVERT_PY.write_text(src, encoding="utf-8")
        print(f"[patch] applied Mistral3 mmproj patches to {CONVERT_PY}")
    else:
        print(f"[patch] {CONVERT_PY} already patched — no changes")


if __name__ == "__main__":
    apply_patches()
```

**Note on regex fragility:** the patch uses regex to locate patch sites. If llama.cpp's converter refactors these sections, the patches will fail the assertions. Rebuild the image (which pulls a specific SHA) protects against upstream drift; the SHA pin in `build_llama_cpp.sh` is the trust anchor.

- [ ] **Step 2: Implement `src/b2cq/workers/mmproj.py`**

```python
"""Multimodal projector (mmproj) GGUF export."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

CONVERT_SCRIPT = "/opt/llama.cpp/convert_hf_to_gguf.py"
PATCH_SCRIPT = "/opt/patch_convert_hf_to_gguf.py"


def is_multimodal(source_dir: Path) -> bool:
    cfg = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    return cfg.get("architectures", [None])[0] == "Mistral3ForConditionalGeneration"


def export_mmproj(source_dir: Path, output_gguf: Path, log_cb: Callable[[str], None]) -> None:
    # 1. Apply patches (idempotent).
    log_cb("Applying Mistral3 mmproj patches to llama.cpp converter")
    r = subprocess.run(["python3", PATCH_SCRIPT], capture_output=True, text=True)
    log_cb(r.stdout.rstrip())
    if r.returncode != 0:
        raise RuntimeError(f"patch script failed: {r.stderr}")

    # 2. Run mmproj export.
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", CONVERT_SCRIPT, str(source_dir), "--mmproj",
           "--outtype", "f16", "--outfile", str(output_gguf)]
    log_cb(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log_cb(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"mmproj export failed with exit {proc.returncode}")

    # 3. Verify v.token_embd.img_break is present.
    verify = subprocess.run(
        ["python3", "-c",
         "import gguf; r = gguf.GGUFReader('%s'); "
         "names = [t.name for t in r.tensors]; "
         "assert 'v.token_embd.img_break' in names, 'MISSING img_break tensor: %d tensors total' %% len(names); "
         "print('mmproj OK:', len(names), 'tensors')" % str(output_gguf)],
        capture_output=True, text=True)
    log_cb(verify.stdout.rstrip())
    if verify.returncode != 0:
        raise RuntimeError(f"mmproj verification failed:\n{verify.stderr}")
```

- [ ] **Step 3: Commit**

```bash
git add scripts/patch_convert_hf_to_gguf.py src/b2cq/workers/mmproj.py
git commit -m "feat: mmproj export with runtime patch for Mistral3 llama.cpp bugs"
```

---

### Task 10: Job orchestrator (two-lane runner)

**Files:**
- Create: `src/b2cq/job_runner.py`

**Interfaces:**
- `async def run_job(job: Job, hf_client: HFClient, calibration: list[dict], progress: ProgressBus, workdir: Path) -> None`
- Orchestrates: source download → two concurrent lanes → per-quant upload → mark statuses on the Job → publish progress events
- On completion, calls `hf_client.close()` (token wipe).

- [ ] **Step 1: Sketch the orchestrator**

```python
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
import time
from datetime import datetime, timezone
from pathlib import Path

from b2cq.calibration import to_plaintext
from b2cq.hf_client import HFClient
from b2cq.job_model import Job, QuantStatus, QuantResult
from b2cq.progress import ProgressBus
from b2cq.quant_catalog import get as get_quant, QuantFamily
from b2cq.workers.safetensors import load_model_for_safetensors, quantize_safetensors
from b2cq.workers.gguf_convert import convert_to_bf16_gguf
from b2cq.workers.gguf_quantize import gguf_quantize, compute_imatrix
from b2cq.workers.mmproj import export_mmproj, is_multimodal
from b2cq.readme_updater import update_source_readme


def _mk_log_cb(job_id: str, quant_id: str, progress: ProgressBus, tail: list[str]):
    """Bounded log-tail + publish each line to the progress bus."""
    loop = asyncio.get_event_loop()
    def cb(line: str) -> None:
        tail.append(line)
        if len(tail) > 100:
            del tail[:-100]
        asyncio.run_coroutine_threadsafe(
            progress.publish(job_id, {"type": "log", "quant": quant_id, "line": line}),
            loop
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
        await _run_one_safetensors_quant(job, q, model, tokenizer, calibration, hf_client, progress, workdir, source_dir)


async def _run_one_safetensors_quant(job, q, model, tokenizer, calibration, hf_client, progress, workdir, source_dir):
    q.status = QuantStatus.RUNNING
    q.started_at = datetime.now(timezone.utc)
    t0 = time.time()
    await progress.publish(job.id, {"type": "status", "quant": q.quant_id, "status": q.status})
    tail = q.log_tail
    log_cb = _mk_log_cb(job.id, q.quant_id, progress, tail)

    try:
        output_dir = workdir / f"safetensors_{q.quant_id}"
        # Note: source_dir is passed through so quantize_safetensors can detect
        # FrndoBrain-class artifacts (by tekken.json presence) and pick the right
        # calibration render path. See SPEC §14 and Task 7 for the rationale.
        await asyncio.to_thread(
            quantize_safetensors,
            model, tokenizer, get_quant(q.quant_id).format, calibration, output_dir, source_dir, log_cb,
        )
        q.status = QuantStatus.UPLOADING
        await progress.publish(job.id, {"type": "status", "quant": q.quant_id, "status": q.status})

        repo_id = q.repo_id or f"{job.owner}/{Path(job.source_model).name}-{q.quant_id}"
        url = await asyncio.to_thread(
            hf_client.upload_folder, repo_id, output_dir,
            create_if_missing=True, private=job.private,
            commit_message=f"B2CQuantizer: {q.quant_id}",
        )
        q.upload_url = url
        q.output_size_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        q.status = QuantStatus.DONE
        # cleanup local
        import shutil; shutil.rmtree(output_dir, ignore_errors=True)
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
    shared_repo = f"{job.owner}/{Path(job.source_model).name}-GGUF"
    for q in gguf_quants:
        if q.status == QuantStatus.SKIPPED:  # from imatrix failure
            continue
        await _run_one_gguf_quant(job, q, bf16_gguf, imatrix_path, source_dir, shared_repo,
                                    hf_client, progress, workdir)

    # cleanup BF16 intermediate
    bf16_gguf.unlink(missing_ok=True)


async def _run_one_gguf_quant(job, q, bf16_gguf, imatrix_path, source_dir, shared_repo,
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
        url = await asyncio.to_thread(hf_client.upload_file, shared_repo, out_gguf, out_gguf.name,
                                        create_if_missing=True, private=job.private,
                                        commit_message=f"B2CQuantizer: {q.quant_id}")
        q.upload_url = url
        q.repo_id = shared_repo
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
    job.status = "running"
    await progress.publish(job.id, {"type": "job_started", "job_id": job.id})

    # Download source
    source_dir = workdir / "source"
    try:
        source_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(hf_client.download_snapshot, job.source_model, source_dir)
    except Exception as e:
        job.status = "failed"
        job.finished_at = datetime.now(timezone.utc)
        await progress.publish(job.id, {"type": "job_failed", "error": str(e)})
        hf_client.close()
        return

    # Run both lanes concurrently
    await asyncio.gather(
        _run_lane_a(job, source_dir, calibration, hf_client, progress, workdir),
        _run_lane_b(job, source_dir, calibration, hf_client, progress, workdir),
    )

    # README update
    if job.update_source_readme:
        try:
            update_source_readme(job, hf_client)
        except Exception as e:
            await progress.publish(job.id, {"type": "readme_failed", "error": str(e)})

    job.status = "complete"
    job.finished_at = datetime.now(timezone.utc)
    await progress.publish(job.id, {"type": "job_complete", "job_id": job.id})
    hf_client.close()
```

- [ ] **Step 2: Commit**

```bash
git add src/b2cq/job_runner.py
git commit -m "feat: two-lane job orchestrator with per-quant failure isolation"
```

---

### Task 11: README updater (source repo `## Quantizations` section)

**Files:**
- Create: `src/b2cq/readme_updater.py`
- Create: `tests/unit/test_readme_updater.py`

**Interfaces:**
- `def update_source_readme(job: Job, hf_client: HFClient) -> None` — downloads source repo's `README.md`, appends or replaces `## Quantizations` section with a canonical table of all `DONE` quants, uploads back to source repo main branch.

- [ ] **Step 1: Write the test**

```python
"""README updater: canonical ## Quantizations section replacement."""
from datetime import datetime, timezone
from b2cq.readme_updater import _render_section, _splice_section


def test_render_section_produces_table():
    from b2cq.job_model import Job, QuantResult, QuantStatus
    from b2cq.calibration import CalibrationSource
    job = Job(
        id="j1", source_model="user/model", owner="user",
        quants=[
            QuantResult(quant_id="Q4_K_M", status=QuantStatus.DONE, lane="B",
                        upload_url="https://huggingface.co/user/model-GGUF/tree/main",
                        repo_id="user/model-GGUF"),
            QuantResult(quant_id="W4A16_GPTQ", status=QuantStatus.DONE, lane="A",
                        upload_url="https://huggingface.co/user/model-W4A16_GPTQ",
                        repo_id="user/model-W4A16_GPTQ"),
            QuantResult(quant_id="NVFP4", status=QuantStatus.FAILED, lane="A"),
        ],
        calibration=CalibrationSource(type="bundled"), private=False,
        update_source_readme=True, started_at=datetime.now(timezone.utc),
    )
    section = _render_section(job)
    assert "## Quantizations" in section
    assert "user/model-GGUF" in section
    assert "user/model-W4A16_GPTQ" in section
    assert "NVFP4" not in section  # failed quants excluded


def test_splice_appends_when_absent():
    original = "# My Model\n\nSome description.\n"
    new = _splice_section(original, "## Quantizations\n\n| tbl |\n")
    assert new.endswith("## Quantizations\n\n| tbl |\n")
    assert "Some description." in new


def test_splice_replaces_when_present():
    original = "# My Model\n\n## Quantizations\n\n| old |\n\n## Other\n\nfoo\n"
    new = _splice_section(original, "## Quantizations\n\n| new |\n")
    assert "| new |" in new
    assert "| old |" not in new
    assert "## Other" in new
    assert "foo" in new
```

- [ ] **Step 2: Implement `src/b2cq/readme_updater.py`**

```python
"""Append or replace the ## Quantizations section in the source repo's README."""
from __future__ import annotations

import re
from b2cq.job_model import Job, QuantStatus
from b2cq.quant_catalog import get as get_quant, QuantFamily
from b2cq.hf_client import HFClient


def _render_section(job: Job) -> str:
    lines = ["## Quantizations", ""]
    lines.append("| Format | Repo | Notes |")
    lines.append("|---|---|---|")

    done = [q for q in job.quants if q.status == QuantStatus.DONE and q.repo_id]

    # GGUF: collapse all GGUF quants into a single "GGUF (multiple levels)" row.
    gguf_done = [q for q in done if get_quant(q.quant_id).family in
                 (QuantFamily.GGUF_K, QuantFamily.GGUF_I, QuantFamily.GGUF_MISC, QuantFamily.GGUF_MMPROJ)]
    if gguf_done:
        levels = ", ".join(sorted(set(q.quant_id for q in gguf_done)))
        repo_id = gguf_done[0].repo_id
        lines.append(f"| GGUF | [{repo_id}](https://huggingface.co/{repo_id}) | {levels} |")

    for q in done:
        spec = get_quant(q.quant_id)
        if spec.family in (QuantFamily.GGUF_K, QuantFamily.GGUF_I, QuantFamily.GGUF_MISC, QuantFamily.GGUF_MMPROJ):
            continue  # collapsed above
        lines.append(f"| {spec.name} | [{q.repo_id}](https://huggingface.co/{q.repo_id}) | {spec.notes} |")

    lines.append("")  # trailing newline
    return "\n".join(lines)


SECTION_RE = re.compile(r"^## Quantizations\s*\n.*?(?=^##\s|\Z)", re.MULTILINE | re.DOTALL)


def _splice_section(original: str, new_section: str) -> str:
    if SECTION_RE.search(original):
        return SECTION_RE.sub(new_section.rstrip() + "\n\n", original, count=1)
    if not original.endswith("\n"):
        original += "\n"
    return original + "\n" + new_section


def update_source_readme(job: Job, hf_client: HFClient) -> str:
    """Fetch source README, splice ## Quantizations, upload back. Returns commit URL."""
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    api = hf_client._api()  # uses token from HFClient
    try:
        p = hf_hub_download(repo_id=job.source_model, filename="README.md",
                            token=hf_client._token)
        original = Path(p).read_text(encoding="utf-8")
    except Exception:
        original = f"# {job.source_model}\n\n"

    new = _splice_section(original, _render_section(job))
    if new == original:
        return "no-op"  # nothing to commit
    return hf_client.update_file(
        repo_id=job.source_model,
        path_in_repo="README.md",
        content=new,
        commit_message="B2CQuantizer: update Quantizations section",
    )
```

- [ ] **Step 3: Run — expect PASS**

Run: `pytest tests/unit/test_readme_updater.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/b2cq/readme_updater.py tests/unit/test_readme_updater.py
git commit -m "feat: source-repo README ## Quantizations section updater"
```

---

### Task 12: FastAPI routes + templates (setup page)

**Files:**
- Create: `src/b2cq/web/__init__.py`, `src/b2cq/web/routes.py`, `src/b2cq/web/templates/base.html`, `src/b2cq/web/templates/setup.html`
- Modify: `src/b2cq/main.py` to include the router and mount templates

**Interfaces:**
- Routes:
  - `GET /` → setup page (HTML)
  - `POST /jobs` → creates a job, starts run_job in the background, redirects to `/jobs/<id>`
  - `GET /jobs` → history list (HTML)
  - `GET /jobs/<id>` → live job dashboard (HTML)
  - `GET /jobs/<id>/stream` → SSE endpoint for that job's progress events
  - `GET /health` (unchanged from Task 2)
  - `POST /calibration/upload` → HTMX upload handler; returns a token id the setup form uses.

- [ ] **Step 1: Write `src/b2cq/web/templates/base.html`**

```html
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>{% block title %}B2CQuantizer{% endblock %}</title>
    <script src="https://unpkg.com/htmx.org@2.0.3"></script>
    <script src="https://unpkg.com/htmx.org@2.0.3/dist/ext/sse.js"></script>
    <script defer src="https://unpkg.com/alpinejs@3.14.1/dist/cdn.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-900">
    <header class="border-b bg-white">
        <div class="max-w-5xl mx-auto px-6 py-3 flex justify-between items-baseline">
            <a href="/" class="text-lg font-semibold">B2CQuantizer</a>
            <a href="/jobs" class="text-sm text-slate-600 hover:text-slate-900">History</a>
        </div>
    </header>
    <main class="max-w-5xl mx-auto px-6 py-8">
        {% block body %}{% endblock %}
    </main>
</body>
</html>
```

- [ ] **Step 2: Write `src/b2cq/web/templates/setup.html`**

Full setup form with grouped quant checkboxes, source-model input, HF token field (password), calibration dropdown, owner/private/README toggles, launch button. Uses HTMX for the calibration upload and for the "compute owner from token" side-effect. Structure to include:

```html
{% extends "base.html" %}
{% block body %}
<form method="post" action="/jobs" class="space-y-6" x-data="{ calType: 'bundled' }">
    <div>
        <label class="block text-sm font-medium">Source model (HF repo id)</label>
        <input name="source_model" required class="w-full border rounded px-3 py-2"
               placeholder="mistralai/Mistral-Small-3.2-24B-Instruct-2506">
    </div>
    <div>
        <label class="block text-sm font-medium">HuggingFace token</label>
        <input name="hf_token" type="password" required class="w-full border rounded px-3 py-2">
        <p class="text-xs text-slate-500 mt-1">Held in memory only; wiped when the job completes.</p>
    </div>
    <div>
        <label class="block text-sm font-medium">Calibration source</label>
        <select name="calibration_type" x-model="calType" class="border rounded px-2 py-1">
            <option value="bundled">Bundled generic corpus</option>
            <option value="upload">Upload JSONL</option>
            <option value="hf_dataset">HuggingFace dataset ID</option>
        </select>
        <div x-show="calType === 'upload'" class="mt-2">
            <input name="calibration_file" type="file" accept=".jsonl">
        </div>
        <div x-show="calType === 'hf_dataset'" class="mt-2">
            <input name="calibration_dataset" placeholder="bigblueceiling/frndobrain-cal-v1"
                   class="border rounded px-2 py-1">
        </div>
    </div>
    <fieldset>
        <legend class="text-sm font-medium mb-2">GGUF K-quants</legend>
        {% for q in gguf_k %}
        <label class="inline-flex items-center mr-4">
            <input type="checkbox" name="quants" value="{{ q.id }}"> <span class="ml-1">{{ q.name }}</span>
        </label>
        {% endfor %}
    </fieldset>
    <fieldset>
        <legend class="text-sm font-medium mb-2">GGUF I-quants (require imatrix pre-step)</legend>
        {% for q in gguf_i %}
        <label class="inline-flex items-center mr-4">
            <input type="checkbox" name="quants" value="{{ q.id }}"> <span class="ml-1">{{ q.name }}</span>
        </label>
        {% endfor %}
    </fieldset>
    <fieldset>
        <legend class="text-sm font-medium mb-2">GGUF misc + mmproj</legend>
        {% for q in gguf_misc %}
        <label class="inline-flex items-center mr-4">
            <input type="checkbox" name="quants" value="{{ q.id }}"> <span class="ml-1">{{ q.name }}</span>
        </label>
        {% endfor %}
        {% for q in gguf_mmproj %}
        <label class="inline-flex items-center mr-4">
            <input type="checkbox" name="quants" value="{{ q.id }}" checked>
            <span class="ml-1">{{ q.name }} <small>(multimodal)</small></span>
        </label>
        {% endfor %}
    </fieldset>
    <fieldset>
        <legend class="text-sm font-medium mb-2">safetensors</legend>
        {% for q in safetensors %}
        <label class="inline-flex items-center mr-4">
            <input type="checkbox" name="quants" value="{{ q.id }}"
                {% if 'blackwell' in q.hardware_requirements and not blackwell_available %}disabled{% endif %}>
            <span class="ml-1">{{ q.name }}
                {% if 'blackwell' in q.hardware_requirements and not blackwell_available %}
                    <small class="text-red-600">(needs Blackwell GPU)</small>
                {% endif %}
            </span>
        </label>
        {% endfor %}
    </fieldset>
    <div>
        <label class="block text-sm font-medium">Upload owner</label>
        <input name="owner" placeholder="bigblueceiling" class="border rounded px-2 py-1">
    </div>
    <div class="flex items-center space-x-4">
        <label><input type="checkbox" name="private"> Private repos</label>
        <label><input type="checkbox" name="update_source_readme" checked> Update source README</label>
    </div>
    <button type="submit" class="bg-slate-900 text-white px-4 py-2 rounded">Launch job</button>
</form>
{% endblock %}
```

- [ ] **Step 3: Write `src/b2cq/web/routes.py`**

```python
"""FastAPI routes for setup, job submission, live job view, history."""
from __future__ import annotations

import asyncio
from pathlib import Path

import torch
from fastapi import APIRouter, Form, Request, UploadFile, File
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


def _blackwell_available() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 12
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
    from tempfile import mkdtemp
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
    job = JOB_STORE.get(job_id)
    return templates.TemplateResponse("job.html", {"request": request, "job": job})


@router.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    async def event_gen():
        async for evt in PROGRESS.subscribe(job_id):
            yield {"data": __import__("json").dumps(evt)}
    return EventSourceResponse(event_gen())
```

- [ ] **Step 4: Wire the router into `src/b2cq/main.py`**

```python
"""FastAPI app entry."""
from fastapi import FastAPI
from b2cq.web.routes import router

app = FastAPI(title="B2CQuantizer", version="0.1.0")
app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 5: Commit**

```bash
git add src/b2cq/web/ src/b2cq/main.py
git commit -m "feat: FastAPI routes + setup page (HTMX + Alpine)"
```

---

### Task 13: Job view + SSE progress + history templates

**Files:**
- Create: `src/b2cq/web/templates/job.html`, `src/b2cq/web/templates/history.html`

- [ ] **Step 1: Write `src/b2cq/web/templates/job.html`**

```html
{% extends "base.html" %}
{% block body %}
<div x-data="jobView()" hx-ext="sse" sse-connect="/jobs/{{ job.id }}/stream">
    <h1 class="text-2xl font-semibold">Job {{ job.id }}</h1>
    <p class="text-slate-600">Source: {{ job.source_model }} · Owner: {{ job.owner }}</p>
    <p x-text="'Status: ' + jobStatus" class="mt-2 font-medium"></p>

    <table class="mt-6 w-full text-sm">
        <thead class="text-left border-b">
            <tr><th>Quant</th><th>Lane</th><th>Status</th><th>Elapsed</th><th>Size</th><th>Repo</th></tr>
        </thead>
        <tbody>
        {% for q in job.quants %}
        <tr class="border-b" :class="statuses['{{ q.quant_id }}'] === 'failed' ? 'bg-red-50' : ''">
            <td class="py-2">{{ q.quant_id }}</td>
            <td>{{ q.lane }}</td>
            <td x-text="statuses['{{ q.quant_id }}'] || '{{ q.status.value }}'"></td>
            <td x-text="(elapsed['{{ q.quant_id }}'] || 0).toFixed(0) + 's'"></td>
            <td x-text="sizes['{{ q.quant_id }}'] || '—'"></td>
            <td><a x-show="urls['{{ q.quant_id }}']" :href="urls['{{ q.quant_id }}']" class="text-blue-600 underline">↗</a></td>
        </tr>
        {% endfor %}
        </tbody>
    </table>

    <div class="mt-8 grid grid-cols-2 gap-4">
        <div>
            <h3 class="font-medium">Lane A (safetensors)</h3>
            <pre class="bg-slate-900 text-slate-100 p-3 text-xs h-64 overflow-auto" x-text="laneA"></pre>
        </div>
        <div>
            <h3 class="font-medium">Lane B (GGUF)</h3>
            <pre class="bg-slate-900 text-slate-100 p-3 text-xs h-64 overflow-auto" x-text="laneB"></pre>
        </div>
    </div>
</div>

<script>
function jobView() {
    return {
        jobStatus: 'running',
        statuses: {},
        elapsed: {},
        sizes: {},
        urls: {},
        laneA: '',
        laneB: '',
        init() {
            const knownLane = { {% for q in job.quants %}'{{ q.quant_id }}': '{{ q.lane }}'{% if not loop.last %},{% endif %}{% endfor %} };
            this.$el.addEventListener('htmx:sseMessage', (e) => {
                const evt = JSON.parse(e.detail.data);
                if (evt.type === 'status') {
                    this.statuses[evt.quant] = evt.status;
                    if (evt.elapsed !== undefined) this.elapsed[evt.quant] = evt.elapsed;
                } else if (evt.type === 'log') {
                    const lane = knownLane[evt.quant] || (evt.quant.includes('lane_a') ? 'A' : 'B');
                    if (lane === 'A') this.laneA = (this.laneA + '\n[' + evt.quant + '] ' + evt.line).slice(-8000);
                    else this.laneB = (this.laneB + '\n[' + evt.quant + '] ' + evt.line).slice(-8000);
                } else if (evt.type === 'job_complete') {
                    this.jobStatus = 'complete';
                } else if (evt.type === 'job_failed') {
                    this.jobStatus = 'failed: ' + evt.error;
                }
            });
        }
    };
}
</script>
{% endblock %}
```

- [ ] **Step 2: Write `src/b2cq/web/templates/history.html`**

```html
{% extends "base.html" %}
{% block body %}
<h1 class="text-2xl font-semibold mb-4">Jobs in this session</h1>
<p class="text-slate-600 mb-4">History is in-memory only; wiped on pod restart.</p>
<table class="w-full text-sm">
    <thead class="text-left border-b">
        <tr><th>Job</th><th>Source</th><th>Started</th><th>Status</th><th>Quants</th></tr>
    </thead>
    <tbody>
    {% for job in jobs %}
    <tr class="border-b">
        <td><a href="/jobs/{{ job.id }}" class="text-blue-600 underline">{{ job.id }}</a></td>
        <td>{{ job.source_model }}</td>
        <td>{{ job.started_at.isoformat() }}</td>
        <td>{{ job.status }}</td>
        <td>{{ job.quants | length }}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
{% endblock %}
```

- [ ] **Step 3: End-to-end smoke — start the container, load `/`, submit a small job, watch job view**

Run locally (or on RunPod pod):
```bash
docker build -t b2cq:dev .
docker run --rm -d --name b2cq-e2e --gpus all -p 8000:8000 b2cq:dev
```

Manually verify in a browser:
1. `http://localhost:8000/` renders the setup form with all quant families visible.
2. NVFP4 checkbox is disabled with a "needs Blackwell GPU" note on non-Blackwell hardware.
3. Submit a job for a small model (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — note: this fails the Mistral-only check by design; use a tiny Mistral fixture if available, otherwise the smoke will fail early with the architecture-refusal error, which is itself the correct behavior).
4. Job view shows the quant table + two live log panes.
5. History page lists the job.

- [ ] **Step 4: Commit**

```bash
git add src/b2cq/web/templates/job.html src/b2cq/web/templates/history.html
git commit -m "feat: live job view (SSE + Alpine) + history page"
```

---

### Task 14: Documentation and smoke procedure

**Files:**
- Create: `docs/deployment.md`, `docs/troubleshooting.md`, `tests/smoke/README.md`

- [ ] **Step 1: Write `docs/deployment.md`**

Cover: choosing a RunPod GPU (H100 for text-only Mistral 24B, RTX PRO 6000 Blackwell for NVFP4), pod disk sizing (≥200 GB), image push instructions (ghcr.io or Docker Hub), env vars supported at container start, port exposure via RunPod's UI, expected first-run download times.

- [ ] **Step 2: Write `docs/troubleshooting.md`**

Cover: common failure modes — GPU OOM during safetensors load, disk-full during BF16 GGUF conversion, HF 401 on private-model access, mmproj `v.token_embd.img_break` missing (patch didn't apply — rebuild image), NVFP4 kernel launch failure (wrong GPU tier), llm-compressor scheme errors on architecture mismatch.

- [ ] **Step 3: Write `tests/smoke/README.md`**

Document the manual E2E smoke procedure:
1. Deploy image to a RunPod H100 80GB pod.
2. Point at a public small Mistral model (e.g., `mistralai/Mistral-7B-v0.1`).
3. Select Q4_K_M + Q8_0 + W4A16_GPTQ. No calibration override (bundled).
4. Verify all three uploads land in the expected repo layout.
5. Verify source repo README (if writable) has `## Quantizations` section.
6. Verify each quant is loadable (vLLM smoke: `MODEL_NAME=<owner>/<name>-W4A16_GPTQ`; llama.cpp smoke: `llama-cli -m <gguf>`).

- [ ] **Step 4: Commit**

```bash
git add docs/ tests/smoke/README.md
git commit -m "docs: deployment, troubleshooting, smoke procedure"
```

---

## Self-review

**Spec coverage** (per SPEC.md sections):
- §4 deployment shape → Tasks 1, 2, 14 (deployment.md)
- §5 tech stack → Tasks 1, 2 (pyproject, Dockerfile)
- §6 pipeline model → Task 10 (job_runner)
- §7 quantization catalog → Task 3
- §8 calibration model → Task 5
- §9 HuggingFace integration → Tasks 4 (client), 11 (README updater), 10 (upload calls)
- §10 UI surface → Tasks 12 (setup), 13 (job/history)
- §11 failure handling → Task 10 (orchestrator per-quant/per-lane isolation)
- §12 security → Task 4 (in-memory token, close-wipes semantics)
- §13 observability → Task 10 (progress bus + bounded log tails); log-to-disk left as an inline concern of the workers, noted in Task 14 smoke doc.
- §14 FrndoBrain-class invariants → Task 7 (safetensors worker: `is_frndobrain_class`, `build_frndobrain_tokenizer`, `_render_calibration_frndobrain` — mistral-common pin, attribute flip, hard-fails) + Task 10 (orchestrator passes source_dir through to quantize_safetensors) + Global Constraints (mistral-common pin declared).

**Placeholder scan:** none. Every module has code, every command has expected output, no "TBD"s.

**Type consistency:**
- `QuantSpec.id` matches values used across catalog, job model, worker dispatch, UI form.
- `HFClient.close()` semantics consistent between Task 4 (definition) and Task 10 (called in `run_job` finally block).
- `CalibrationSource` schema consistent across Tasks 5, 10, 12.
- `run_job` signature `(Job, HFClient, calibration, ProgressBus, workdir)` consistent between Task 10 and Task 12 (routes.py invocation).
- `quantize_safetensors(model, tokenizer, format, calibration, output_dir, source_dir, log_cb)` — signature (with `source_dir` as second-to-last positional) consistent between Task 7's Interfaces block, its implementation code block, and Task 10's `_run_one_safetensors_quant` call site.

**Scope:** single spec, single implementation. No sub-project decomposition needed.

---

**Plan complete.** Save this file at `D:\github\B2CQuantizer\PLAN.md` alongside `SPEC.md`.

**Execution options for the next agent:**

1. **Subagent-driven (recommended)** — dispatch a fresh subagent per task, review between tasks.
2. **Inline execution** — execute tasks in one session with checkpoints.

The next agent picks which; either way, the plan is self-contained (SPEC.md + PLAN.md + a fresh git repo is all they need).
