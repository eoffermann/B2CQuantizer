"""Shared helper for running streaming subprocesses with line-by-line logging."""
from __future__ import annotations

import subprocess
from typing import Callable


def run_streamed(cmd: list[str], log_cb: Callable[[str], None], tool_name: str) -> None:
    log_cb(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log_cb(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{tool_name} failed with exit {proc.returncode}")
