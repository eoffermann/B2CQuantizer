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

# Install the converter's runtime deps. We deliberately do NOT install
# requirements/requirements-convert_hf_to_gguf.txt verbatim: upstream pins a
# CPU-only "torch==..." there, and installing it would clobber the CUDA (cu128)
# torch wheel installed earlier in the image, silently breaking GPU quantization.
# Instead install only the converter's non-torch runtime deps explicitly.
# numpy/transformers/safetensors are already provided by the app image.
python3 -m pip install --no-cache-dir gguf sentencepiece "mistral-common>=1.5.0"

echo "llama.cpp built at commit $(git rev-parse HEAD)"
