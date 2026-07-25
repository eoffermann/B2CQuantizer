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
