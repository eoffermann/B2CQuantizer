"""Tests for scripts/patch_convert_hf_to_gguf.py against a REAL llama.cpp
conversion/base.py snapshot (see tests/unit/fixtures/
llama_cpp_conversion_base_snapshot.py -- a faithful, byte-for-byte extraction
of MmprojModel from the pinned llama.cpp commit, not a hand-invented
stand-in).

Covers the confirmed Lane B repro: the old patch script targeted
convert_hf_to_gguf.py directly, but at our pinned commit (720d7fa4) that file
is a thin CLI wrapper -- MmprojModel actually lives in conversion/base.py --
so both of its regexes could never match, and it raised a loud RuntimeError
instead of silently doing nothing.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

# scripts/ is not part of the installed `b2cq` package -- add it to sys.path
# so the module under test is importable by its plain script name, matching
# how it's actually invoked (`python3 /opt/patch_convert_hf_to_gguf.py`).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import patch_convert_hf_to_gguf as patch_module

FIXTURE = Path(__file__).parent / "fixtures" / "llama_cpp_conversion_base_snapshot.py"


@pytest.fixture
def conversion_base_copy(tmp_path, monkeypatch):
    """Copy the real MmprojModel snapshot to a tmp path shaped like
    /opt/llama.cpp/conversion/base.py, and point the patch module's
    CONVERSION_BASE_PY constant at it so apply_patches() operates on the copy
    instead of the real container path (which doesn't exist on this host)."""
    conversion_dir = tmp_path / "llama.cpp" / "conversion"
    conversion_dir.mkdir(parents=True)
    target = conversion_dir / "base.py"
    target.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(patch_module, "CONVERSION_BASE_PY", target)
    return target


def test_fixture_is_a_faithful_real_snapshot():
    """Sanity check that the fixture actually is what test_mmproj_patch.py
    claims it is: the genuine MmprojModel.filter_tensors shape from the
    pinned llama.cpp commit, unpatched."""
    src = FIXTURE.read_text(encoding="utf-8")
    assert "class MmprojModel(ModelBase):" in src
    assert 'if "language_model." in name:' in src
    assert "return None" in src
    assert patch_module._MARKER not in src


def test_apply_patches_applies_once(conversion_base_copy):
    patch_module.apply_patches()
    patched = conversion_base_copy.read_text(encoding="utf-8")

    assert patch_module._MARKER in patched
    assert patched.count(patch_module._MARKER) == 1
    assert 'if "language_model.model.embed_tokens" in name:' in patched
    # The original generic drop is preserved (comes after the exemption).
    assert 'if "language_model." in name:' in patched
    assert "return None" in patched


def test_apply_patches_is_idempotent(conversion_base_copy):
    patch_module.apply_patches()
    first = conversion_base_copy.read_text(encoding="utf-8")

    patch_module.apply_patches()
    second = conversion_base_copy.read_text(encoding="utf-8")

    assert first == second
    assert first.count(patch_module._MARKER) == 1


def test_patched_file_still_parses_as_valid_python(conversion_base_copy):
    patch_module.apply_patches()
    patched = conversion_base_copy.read_text(encoding="utf-8")

    ast.parse(patched)  # raises SyntaxError if the patch mangled the source


def test_exemption_sits_inside_filter_tensors_before_language_model_drop(conversion_base_copy):
    patch_module.apply_patches()
    patched = conversion_base_copy.read_text(encoding="utf-8")

    method_start = patched.index("def filter_tensors(")
    # Scope the search to the filter_tensors method body (up to the next
    # method definition) so we're asserting ordering *within* the method,
    # not just anywhere in the file.
    next_method = patched.index("\n    def ", method_start + 1)
    method_body = patched[method_start:next_method]

    exemption_idx = method_body.index('if "language_model.model.embed_tokens" in name:')
    drop_idx = method_body.index('if "language_model." in name:\n            return None')

    assert exemption_idx < drop_idx, (
        "the embed_tokens exemption must come BEFORE the generic "
        "language_model. drop, or embed_tokens gets dropped first"
    )


def test_missing_conversion_base_py_raises_file_not_found(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist" / "base.py"
    monkeypatch.setattr(patch_module, "CONVERSION_BASE_PY", missing)

    with pytest.raises(FileNotFoundError, match=r"conversion package not found"):
        patch_module.apply_patches()


def test_pattern_does_not_match_stale_convert_hf_to_gguf_shape():
    """Regression guard for the actual Lane B failure: a script that patches
    a plain top-level convert_hf_to_gguf.py with no MmprojModel class present
    must fail loudly (RuntimeError), not silently no-op."""
    stale_src = (
        "from conversion import ModelBase, ModelType, get_model_architecture\n"
        "\n"
        "def split_str_to_n_bytes(split_str):\n"
        "    return int(split_str)\n"
    )
    src, n = patch_module._apply_embed_tokens_exemption(stale_src)
    assert n == 0
