"""Runtime patch for llama.cpp's convert_hf_to_gguf.py.

Fixes two Mistral3 / pixtral mmproj export bugs. Idempotent — safe to
re-run. Applied on-demand by the mmproj worker before invoking the
converter, so we don't need to fork llama.cpp.
"""
from __future__ import annotations

import re
from pathlib import Path

CONVERT_PY = Path("/opt/llama.cpp/convert_hf_to_gguf.py")


def apply_patches() -> None:
    if not CONVERT_PY.exists():
        raise FileNotFoundError(f"llama.cpp converter not found at {CONVERT_PY}")
    src = CONVERT_PY.read_text(encoding="utf-8")
    original = src

    # Patch 1: MmprojModel.filter_tensors — allow language_model.model.embed_tokens through
    marker1 = "# --- B2CQ patch: allow LM embed for [IMG_BREAK] extraction ---"
    if marker1 not in src:
        # Locate the "language_model." exclusion line in MmprojModel.filter_tensors
        # and add an exemption for embed_tokens BEFORE the return-False branch.
        pattern1 = re.compile(
            r"(class MmprojModel.*?def filter_tensors.*?)(if\s+\"language_model\.\"\s+in\s+name\s*:\s*\n\s*return\s+False)",
            re.DOTALL,
        )
        replacement1 = (
            r"\1" + marker1 + "\n"
            r"        if 'language_model.model.embed_tokens' in name:\n"
            r"            return True\n"
            r"        \2"
        )
        src, n = pattern1.subn(replacement1, src, count=1)
        assert n == 1, "Patch 1 failed to apply — file structure may have changed"

    # Patch 2: pixtral activation flag — use projector_hidden_act instead of hparams["hidden_act"]
    marker2 = "# --- B2CQ patch: pixtral projector activation ---"
    if marker2 not in src:
        # Locate the clip.use_silu / clip.use_gelu emission for pixtral model class.
        pattern2 = re.compile(
            r"(class LlavaVisionModel.*?)(self\.gguf_writer\.add_bool\(\"clip\.use_silu\",\s*)(self\.hparams\[[\"']hidden_act[\"']\]\s*==\s*[\"']silu[\"'])(.*?add_bool\(\"clip\.use_gelu\",\s*)(self\.hparams\[[\"']hidden_act[\"']\]\s*==\s*[\"']gelu[\"'])",
            re.DOTALL,
        )
        replacement2 = (
            r"\1" + marker2 + "\n        "
            r"_proj_act = self.hparams.get('projector_hidden_act', self.hparams.get('hidden_act', 'gelu'))\n        "
            r"\g<2>_proj_act == 'silu'\g<4>_proj_act == 'gelu'"
        )
        src, n = pattern2.subn(replacement2, src, count=1)
        assert n == 1, "Patch 2 failed to apply — file structure may have changed"

    if src != original:
        CONVERT_PY.write_text(src, encoding="utf-8")
        print(f"[patch] applied Mistral3 mmproj patches to {CONVERT_PY}")
    else:
        print(f"[patch] {CONVERT_PY} already patched — no changes")


if __name__ == "__main__":
    apply_patches()
