"""Append or replace the ## Quantizations section in the source repo's README."""
from __future__ import annotations

import re
from b2cq.job_model import Job, QuantStatus
from b2cq.quant_catalog import get as get_quant, QuantFamily
from b2cq.hf_client import HFClient


def _render_section(job: Job) -> str:
    lines = ["## Quantizations", ""]
    lines.append("| Format | Repo | Notes |")
    lines.append("|---|---|---|")

    done = [q for q in job.quants if q.status == QuantStatus.DONE and q.repo_id]

    # GGUF: collapse all GGUF quants into a single "GGUF (multiple levels)" row.
    gguf_done = [q for q in done if get_quant(q.quant_id).family in
                 (QuantFamily.GGUF_K, QuantFamily.GGUF_I, QuantFamily.GGUF_MISC, QuantFamily.GGUF_MMPROJ)]
    if gguf_done:
        levels = ", ".join(sorted(set(q.quant_id for q in gguf_done)))
        repo_id = gguf_done[0].repo_id
        lines.append(f"| GGUF | [{repo_id}](https://huggingface.co/{repo_id}) | {levels} |")

    for q in done:
        spec = get_quant(q.quant_id)
        if spec.family in (QuantFamily.GGUF_K, QuantFamily.GGUF_I, QuantFamily.GGUF_MISC, QuantFamily.GGUF_MMPROJ):
            continue  # collapsed above
        lines.append(f"| {spec.name} | [{q.repo_id}](https://huggingface.co/{q.repo_id}) | {spec.notes} |")

    lines.append("")  # trailing newline
    return "\n".join(lines)


SECTION_RE = re.compile(r"^## Quantizations\s*\n.*?(?=^##\s|\Z)", re.MULTILINE | re.DOTALL)


def _splice_section(original: str, new_section: str) -> str:
    if SECTION_RE.search(original):
        return SECTION_RE.sub(new_section.rstrip() + "\n\n", original, count=1)
    if not original.endswith("\n"):
        original += "\n"
    return original + "\n" + new_section


def update_source_readme(job: Job, hf_client: HFClient) -> str:
    """Fetch source README, splice ## Quantizations, upload back. Returns commit URL."""
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    api = hf_client._api()  # uses token from HFClient
    try:
        p = hf_hub_download(repo_id=job.source_model, filename="README.md",
                            token=hf_client._token)
        original = Path(p).read_text(encoding="utf-8")
    except Exception:
        original = f"# {job.source_model}\n\n"

    new = _splice_section(original, _render_section(job))
    if new == original:
        return "no-op"  # nothing to commit
    return hf_client.update_file(
        repo_id=job.source_model,
        path_in_repo="README.md",
        content=new,
        commit_message="B2CQuantizer: update Quantizations section",
    )
