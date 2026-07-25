#!/usr/bin/env bash
# Clone and build llama.cpp with CUDA support. Anchored to a specific
# commit SHA after first successful build (edit LLAMA_CPP_COMMIT below).
set -euo pipefail

# Pinned from the first successful build (2026-07-25). Bump deliberately.
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-720d7fa4097f76e5d0eade5a92c1df87c1faf9d9}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/opt/llama.cpp}"

apt-get update
apt-get install -y --no-install-recommends git cmake build-essential libcurl4-openssl-dev
rm -rf /var/lib/apt/lists/*

git clone https://github.com/ggerganov/llama.cpp.git "${INSTALL_PREFIX}"
cd "${INSTALL_PREFIX}"
git checkout "${LLAMA_CPP_COMMIT}"

# Build containers have no NVIDIA driver, so libcuda.so.1 (the driver API lib
# that libggml-cuda links against) does not exist at link time. Point the
# linker at the CUDA toolkit's driver stub for the duration of the build, then
# remove the symlink so the container runtime's real driver mount is used.
CUDA_STUBS=/usr/local/cuda/lib64/stubs
STUB_LINK=/usr/lib/x86_64-linux-gnu/libcuda.so.1
ln -sf "${CUDA_STUBS}/libcuda.so" "${STUB_LINK}"
trap 'rm -f "${STUB_LINK}"' EXIT

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
