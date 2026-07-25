"""BF16 GGUF conversion (source safetensors -> BF16 GGUF)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from b2cq.workers._run import run_streamed

CONVERT_SCRIPT = "/opt/llama.cpp/convert_hf_to_gguf.py"


def convert_to_bf16_gguf(source_dir: Path, output_gguf: Path, log_cb: Callable[[str], None]) -> None:
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", CONVERT_SCRIPT, str(source_dir), "--outtype", "bf16",
           "--outfile", str(output_gguf)]
    run_streamed(cmd, log_cb, "convert_hf_to_gguf.py")
    if not output_gguf.exists() or output_gguf.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"BF16 GGUF suspiciously small or missing: {output_gguf}")
    log_cb(f"BF16 GGUF written: {output_gguf} ({output_gguf.stat().st_size / 2**30:.1f} GiB)")
