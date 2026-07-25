"""Multimodal projector (mmproj) GGUF export."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

from b2cq.workers._run import run_streamed

CONVERT_SCRIPT = "/opt/llama.cpp/convert_hf_to_gguf.py"
PATCH_SCRIPT = "/opt/patch_convert_hf_to_gguf.py"

# Inline verification script run as `python3 -c VERIFY_SCRIPT <gguf_path>`.
# The gguf path is passed as sys.argv[1] rather than %-formatted into the
# source string, so paths with quotes/backslashes/spaces can't break the
# generated code.
VERIFY_SCRIPT = (
    "import sys, gguf\n"
    "r = gguf.GGUFReader(sys.argv[1])\n"
    "names = [t.name for t in r.tensors]\n"
    "assert 'v.token_embd.img_break' in names, "
    "'MISSING img_break tensor: %d tensors total' % len(names)\n"
    "print('mmproj OK:', len(names), 'tensors')\n"
)


def is_multimodal(source_dir: Path) -> bool:
    cfg = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    architectures = cfg.get("architectures") or []
    return bool(architectures) and architectures[0] == "Mistral3ForConditionalGeneration"


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
    run_streamed(cmd, log_cb, "mmproj export")

    # 3. Verify v.token_embd.img_break is present.
    verify = subprocess.run(
        ["python3", "-c", VERIFY_SCRIPT, str(output_gguf)],
        capture_output=True, text=True)
    log_cb(verify.stdout.rstrip())
    if verify.returncode != 0:
        raise RuntimeError(f"mmproj verification failed:\n{verify.stderr}")
