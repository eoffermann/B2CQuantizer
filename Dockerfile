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

# PyTorch with CUDA 12.8 wheel (must come from pytorch index).
# torch 2.10.* is the newest cu128 wheel within llmcompressor 0.10's supported
# range (torch>=2.9,<=2.10) and provides Blackwell/SM120 support (needs >=2.7).
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.10.*"

# llmcompressor (PyPI name is "llmcompressor", NOT "llm-compressor") and other
# heavy deps. 0.10.* ships AWQModifier + NVFP4 (0.3 predated both) and pins a
# transformers 4.56-4.57 / accelerate 1.6-1.12 stack matching pyproject.toml.
RUN python3 -m pip install "llmcompressor==0.10.*"

# Build llama.cpp (before app deps: the multi-arch CUDA compile is the most
# expensive layer and must not be invalidated by routine pyproject changes)
COPY scripts/build_llama_cpp.sh /opt/build_llama_cpp.sh
RUN chmod +x /opt/build_llama_cpp.sh && /opt/build_llama_cpp.sh

# App deps (from pyproject.toml, but copied first for cache reuse)
COPY pyproject.toml /app/pyproject.toml
COPY src/b2cq/__init__.py /app/src/b2cq/__init__.py
COPY src/b2cq_data/__init__.py /app/src/b2cq_data/__init__.py
WORKDIR /app
RUN python3 -m pip install -e ".[dev]"

# Build bundled calibration corpus
COPY scripts/build_bundled_calibration.py /tmp/build_cal.py
RUN python3 /tmp/build_cal.py && rm /tmp/build_cal.py

# Copy app source (last, so app changes don't invalidate dep layers)
COPY src/ /app/src/
COPY scripts/patch_convert_hf_to_gguf.py /opt/patch_convert_hf_to_gguf.py
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# Build-time API smoke test: catches llmcompressor/compressed_tensors API
# drift (the class of bug behind the "instant death on first real GPU run"
# incident -- Modifier kwargs that don't exist as pydantic fields, or an
# entrypoint import path that no longer re-exports what we import) at image
# build time instead of on a rented GPU pod. Placed after `COPY src/` (the
# earliest point b2cq.workers.safetensors is actually importable) rather than
# immediately after `pip install -e ".[dev]"`, since editable-install alone
# doesn't make the real module files present yet -- see the layer ordering
# comment above. Pure pydantic Modifier construction: no GPU/model weights
# needed, just the installed llmcompressor/compressed_tensors/torch stack.
#
# Written as a single JSON exec-form RUN (rather than a `RUN <<EOF` heredoc)
# so it doesn't depend on a BuildKit heredoc-capable frontend/`# syntax=`
# directive -- this Dockerfile targets plain `docker build` compatibility.
RUN ["python3", "-c", "import types\nfrom b2cq.workers.safetensors import _MULTIMODAL_ARCH, _TEXT_ARCH, _build_recipe\nSUPPORTED_FORMATS = ['W4A16_GPTQ', 'W4A16_AWQ', 'NVFP4', 'FP8_E4M3']\nUNSUPPORTED_FORMATS = ['FP8_E5M2']\ndef fake_model(arch, num_hidden_layers=2):\n    text_config = types.SimpleNamespace(num_hidden_layers=num_hidden_layers)\n    config = types.SimpleNamespace(architectures=[arch], text_config=text_config)\n    return types.SimpleNamespace(config=config)\nfor arch in (_MULTIMODAL_ARCH, _TEXT_ARCH):\n    model = fake_model(arch)\n    for fmt in SUPPORTED_FORMATS:\n        recipe = _build_recipe(fmt, arch, model)\n        assert isinstance(recipe, list) and len(recipe) >= 1, arch + '/' + fmt + ': empty recipe'\n        print('[smoke] ' + arch + ' / ' + fmt + ': OK (' + str(len(recipe)) + ' modifier(s))')\n    for fmt in UNSUPPORTED_FORMATS:\n        try:\n            _build_recipe(fmt, arch, model)\n        except ValueError as e:\n            print('[smoke] ' + arch + ' / ' + fmt + ': correctly raised ValueError: ' + str(e))\n        else:\n            raise AssertionError(arch + '/' + fmt + ': expected ValueError but got a recipe -- FP8 E5M2 is not supported by llm-compressor 0.10')\nfrom llmcompressor import oneshot\nassert callable(oneshot)\nprint('[smoke] llmcompressor.oneshot import path OK')\nprint('[smoke] safetensors _build_recipe API smoke test passed for all formats/archs')"]

EXPOSE 8000
ENTRYPOINT ["/opt/entrypoint.sh"]
