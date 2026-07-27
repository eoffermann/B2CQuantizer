"""Runtime patch for llama.cpp's `conversion/base.py`.

Fixes the Mistral3 / pixtral mmproj `language_model.` tensor-drop bug in
`MmprojModel.filter_tensors`. That method drops EVERY tensor whose name
contains "language_model." before it ever reaches
`LlavaVisionModel.modify_tensors` (conversion/llava.py) -- which needs
`language_model.model.embed_tokens.weight` to survive so it can extract the
`[IMG_BREAK]` token embedding from it and emit `v.token_embd.img_break`. This
patch exempts that one tensor before the generic drop. Idempotent -- safe to
re-run. Applied on-demand by the mmproj worker before invoking the converter,
so we don't need to fork llama.cpp.

Target-path note: at our pinned llama.cpp commit (720d7fa4, see
scripts/build_llama_cpp.sh / LLAMA_CPP_COMMIT), the converter was refactored
into a `conversion/` package. `convert_hf_to_gguf.py` is now a thin CLI
wrapper that does `from conversion import (ModelBase, ModelType, ...)` --
`MmprojModel` itself lives in `conversion/base.py`, not in
`convert_hf_to_gguf.py` anymore. An earlier version of this patch script
targeted `convert_hf_to_gguf.py` directly; both of its regexes could never
match against the new layout (the class simply isn't there), which is why it
raised a loud RuntimeError on the first real run instead of silently
no-op'ing.

Historical note -- pixtral activation flag: this file used to ALSO patch
`LlavaVisionModel.set_gguf_parameters`'s pixtral activation-flag emission
(`add_bool("clip.use_silu", ...)` / `add_bool("clip.use_gelu", ...)`) to read
`projector_hidden_act` instead of `hparams["hidden_act"]`. That patch is now
obsolete and has been dropped: at this llama.cpp version,
`conversion/pixtral.py`'s `PixtralModel.set_gguf_parameters` deliberately
calls `self.gguf_writer.add_vision_use_silu(True)` unconditionally for
Mistral-format multimodal projectors -- there is no `hidden_act`-derived
boolean left to patch. The projector's actual activation behavior at
inference time is resolved via `clip.projector_type` (PIXTRAL), not via this
flag. The `v.token_embd.img_break` post-export verification in
`workers/mmproj.py` remains the real safety net for this whole export path
and is untouched by this change.
"""
from __future__ import annotations

import re
from pathlib import Path

CONVERSION_BASE_PY = Path("/opt/llama.cpp/conversion/base.py")

_MARKER = "# --- B2CQ patch: allow LM embed for [IMG_BREAK] extraction ---"

# Anchored on the real conversion/base.py shape (see
# tests/unit/fixtures/llama_cpp_conversion_base_snapshot.py for a faithful
# snapshot of the class this matches against):
#
#     class MmprojModel(ModelBase):
#         ...
#         @classmethod
#         def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
#             name, gen = item
#
#             # Skip non-multimodal tensors
#             if "language_model." in name:
#                 return None
#
#             return super().filter_tensors(item)
_PATTERN = re.compile(
    r"(class\s+MmprojModel\(ModelBase\):.*?"
    r"def\s+filter_tensors\(\s*cls,\s*item:.*?\)\s*->\s*.*?:\s*\n"
    r"\s*name,\s*gen\s*=\s*item\n\n)"
    r"(\s*)(# Skip non-multimodal tensors\n"
    r"\s*if\s+\"language_model\.\"\s+in\s+name\s*:\s*\n"
    r"\s*return\s+None)",
    re.DOTALL,
)


def apply_patches() -> None:
    if not CONVERSION_BASE_PY.exists():
        raise FileNotFoundError(
            f"llama.cpp conversion package not found at {CONVERSION_BASE_PY}"
        )
    src = CONVERSION_BASE_PY.read_text(encoding="utf-8")
    original = src

    if _MARKER not in src:
        src, n = _apply_embed_tokens_exemption(src)
        if n != 1:
            raise RuntimeError(
                "MmprojModel.filter_tensors patch failed to apply -- "
                f"{CONVERSION_BASE_PY} structure has changed since llama.cpp "
                "commit 720d7fa4; re-check the class shape and update the "
                "regex in scripts/patch_convert_hf_to_gguf.py"
            )

    if src != original:
        CONVERSION_BASE_PY.write_text(src, encoding="utf-8")
        print(f"[patch] applied Mistral3 mmproj patch to {CONVERSION_BASE_PY}")
    else:
        print(f"[patch] {CONVERSION_BASE_PY} already patched -- no changes")


def _apply_embed_tokens_exemption(src: str) -> tuple[str, int]:
    """Insert an exemption for `language_model.model.embed_tokens` BEFORE the
    generic "language_model." drop in `MmprojModel.filter_tensors`, returning
    the item unchanged (so the tensor keeps its original
    `language_model.model.embed_tokens.weight` name, which is exactly the
    name `LlavaVisionModel.modify_tensors` (conversion/llava.py) checks for
    via `embed_key = "embed_tokens.weight"` / `embed_key in name`).
    """

    def _replace(match: re.Match) -> str:
        prefix, indent, tail = match.group(1), match.group(2), match.group(3)
        exemption = (
            f"{indent}{_MARKER}\n"
            f'{indent}if "language_model.model.embed_tokens" in name:\n'
            f"{indent}    return item\n\n"
        )
        return prefix + exemption + indent + tail

    return _PATTERN.subn(_replace, src, count=1)


if __name__ == "__main__":
    apply_patches()
