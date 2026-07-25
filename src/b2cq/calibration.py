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
