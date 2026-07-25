"""Build the bundled calibration corpus at Docker-build time.

1000 samples split ~600 wikitext + ~400 openassistant-oasst1 (chat-style).
Saved as OpenAI-messages JSONL to src/b2cq_data/calibration/bundled.jsonl.
"""
import json
from pathlib import Path
from datasets import load_dataset

OUT = Path(__file__).parent.parent / "src/b2cq_data/calibration/bundled.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

samples = []

# 600 wikitext samples as single-turn "user says X" for language coverage.
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
for row in wiki:
    text = row["text"].strip()
    if len(text) > 200:
        samples.append({"messages": [{"role": "user", "content": text[:1500]}]})
    if len(samples) >= 600:
        break

# 400 conversational samples from a small open corpus.
oasst = load_dataset("OpenAssistant/oasst1", split="train")
for row in oasst:
    if row["role"] == "prompter" and row["parent_id"] is None:
        samples.append({"messages": [{"role": "user", "content": row["text"][:1500]}]})
    if len(samples) >= 1000:
        break

with OUT.open("w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Wrote {len(samples)} bundled calibration samples to {OUT}")
