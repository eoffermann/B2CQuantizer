"""GGUF quantize wrapper (llama-quantize)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from b2cq.workers._run import run_streamed

QUANTIZE_BIN = "/opt/llama.cpp/build/bin/llama-quantize"
IMATRIX_BIN = "/opt/llama.cpp/build/bin/llama-imatrix"


def gguf_quantize(bf16_gguf: Path, output_gguf: Path, format: str,
                  log_cb: Callable[[str], None], imatrix: Path | None = None) -> None:
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [QUANTIZE_BIN]
    if imatrix is not None:
        cmd += ["--imatrix", str(imatrix)]
    cmd += [str(bf16_gguf), str(output_gguf), format]
    run_streamed(cmd, log_cb, "llama-quantize")
    if not output_gguf.exists():
        raise RuntimeError(f"llama-quantize output missing: {output_gguf}")
    log_cb(f"Wrote {format} GGUF: {output_gguf} ({output_gguf.stat().st_size / 2**30:.2f} GiB)")


def compute_imatrix(bf16_gguf: Path, calibration_text: Path, output_imatrix: Path,
                    log_cb: Callable[[str], None], n_chunks: int = 100) -> None:
    output_imatrix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [IMATRIX_BIN, "-m", str(bf16_gguf), "-f", str(calibration_text),
           "-o", str(output_imatrix), "--chunks", str(n_chunks)]
    run_streamed(cmd, log_cb, "llama-imatrix")
    log_cb(f"Wrote imatrix: {output_imatrix}")
