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
