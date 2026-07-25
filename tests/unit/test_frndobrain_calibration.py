"""FrndoBrain-class artifact detection + calibration-rendering invariants (SPEC §14).

Covers, without requiring GPU/torch/llmcompressor:
  (a) is_frndobrain_class detection via tekken.json presence.
  (b) build_frndobrain_tokenizer hard-fails on mistral-common version mismatch
      (checked before any file I/O, so no real tekken.json needed).
  (c) The real, installed mistral-common 1.11.5 attribute contract:
      `_message_position_to_encode_tools_settings` exists on the actual
      InstructTokenizerV11 class used at runtime and accepts
      UserMessagePosition.first, exercised via the production
      build_frndobrain_tokenizer function end-to-end (no network — the
      tekken.json fixture is synthesized locally from mistral-common's own
      bundled tekken asset with its version bumped to "v11" and the full
      SpecialTokens set added, which is exactly what a real Mistral-Small-
      3.2-24B-class tekken.json contains).
  (d) quantize_safetensors' artifact-class dispatch: the FrndoBrain hard-fail
      path (corpus/tokenizer mismatch -> RuntimeError) and the HF-template
      fallback path for non-FrndoBrain artifacts -- both driven through the
      real `quantize_safetensors` entry point with `_build_recipe` and the
      `llmcompressor.transformers.oneshot` import stubbed out (llmcompressor
      is a Docker-image-only heavy dependency; see module docstring in
      b2cq/workers/safetensors.py).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from b2cq.workers.safetensors import (
    build_frndobrain_tokenizer,
    is_frndobrain_class,
    quantize_safetensors,
)


# ---------------------------------------------------------------------------
# (a) is_frndobrain_class
# ---------------------------------------------------------------------------


def test_is_frndobrain_class_true_with_tekken_json(tmp_path):
    (tmp_path / "tekken.json").write_text("{}", encoding="utf-8")
    assert is_frndobrain_class(tmp_path) is True


def test_is_frndobrain_class_false_without_tekken_json(tmp_path):
    assert is_frndobrain_class(tmp_path) is False


# ---------------------------------------------------------------------------
# (b) version-mismatch hard-fail (checked before any file I/O)
# ---------------------------------------------------------------------------


def test_build_frndobrain_tokenizer_hard_fails_on_version_mismatch(tmp_path, monkeypatch):
    import mistral_common

    monkeypatch.setattr(mistral_common, "__version__", "1.9.0")
    # No tekken.json in tmp_path -- the version check must fire before any
    # attempt to read the file, so this must NOT raise FileNotFoundError.
    with pytest.raises(RuntimeError, match="mistral-common version mismatch"):
        build_frndobrain_tokenizer(tmp_path)


# ---------------------------------------------------------------------------
# (c) real attribute-contract verification against installed mistral-common
# ---------------------------------------------------------------------------


def _write_synthetic_v11_tekken_json(dest: Path) -> None:
    """Synthesize a locally-valid v11 tekken.json from mistral-common's own
    bundled tekken asset (ships in mistral_common/data/), entirely offline.

    mistral-common's bundled fixtures are tagged tokenizer version "v3", which
    resolves to InstructTokenizerV3 rather than the V11 class Mistral-Small-
    3.2-24B ships with. Loading a v11+ tokenizer file also requires an
    explicit "special_tokens" list (v3-and-below infer a deprecated default
    set). Bumping the version field to "v11" and supplying the full
    SpecialTokens enum as the special_tokens list produces a file that
    mistral-common's own MistralTokenizer.from_file loader accepts and
    resolves to a genuine InstructTokenizerV11 instance -- no network access,
    no real trained tekken.json required.
    """
    import mistral_common
    from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, SpecialTokens

    data_dir = Path(mistral_common.__file__).parent / "data"
    src = data_dir / "tekken_240911.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload["config"]["version"] = "v11"
    all_tokens = [
        SpecialTokenInfo(rank=i, token_str=s.value, is_control=True)
        for i, s in enumerate(SpecialTokens)
    ]
    payload["special_tokens"] = [dict(t) for t in all_tokens]
    payload["config"]["default_num_special_tokens"] = max(
        payload["config"]["default_num_special_tokens"], len(all_tokens)
    )
    dest.write_text(json.dumps(payload), encoding="utf-8")


def test_real_mistral_common_v11_attribute_contract(tmp_path):
    """End-to-end against the ACTUAL installed mistral-common==1.11.5: builds
    a real InstructTokenizerV11 via the production build_frndobrain_tokenizer
    function and confirms the version-pinned attribute exists and the flip
    to UserMessagePosition.first takes effect on the live instance."""
    import mistral_common
    from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV11
    from mistral_common.tokens.tokenizers.base import UserMessagePosition

    assert mistral_common.__version__ == "1.11.5"

    _write_synthetic_v11_tekken_json(tmp_path / "tekken.json")

    tok = build_frndobrain_tokenizer(tmp_path)
    it = tok.instruct_tokenizer

    assert isinstance(it, InstructTokenizerV11)
    assert hasattr(it, "_message_position_to_encode_tools_settings")
    # build_frndobrain_tokenizer performs the flip internally; confirm it stuck.
    assert it._message_position_to_encode_tools_settings == UserMessagePosition.first


# ---------------------------------------------------------------------------
# (d) quantize_safetensors dispatch: FrndoBrain hard-fail + HF fallback
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, arch):
        self.architectures = [arch]


class _FakeModel:
    def __init__(self, arch="MistralForCausalLM"):
        self.config = _FakeConfig(arch)


def _install_fake_oneshot(monkeypatch, recorder):
    """Insert a fake `llmcompressor.transformers` module into sys.modules so
    `from llmcompressor.transformers import oneshot` succeeds without the
    real (heavy, Docker-image-only) llmcompressor package installed. The
    fake `oneshot` writes the minimal output quantize_safetensors verifies
    (config.json + a *.safetensors file) and records its call args/kwargs.
    """

    def fake_oneshot(**kwargs):
        recorder.append(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}", encoding="utf-8")
        (out / "model.safetensors").write_bytes(b"")

    fake_llmcompressor = types.ModuleType("llmcompressor")
    fake_transformers_submodule = types.ModuleType("llmcompressor.transformers")
    fake_transformers_submodule.oneshot = fake_oneshot
    fake_llmcompressor.transformers = fake_transformers_submodule

    monkeypatch.setitem(sys.modules, "llmcompressor", fake_llmcompressor)
    monkeypatch.setitem(sys.modules, "llmcompressor.transformers", fake_transformers_submodule)


def test_quantize_safetensors_frndobrain_hard_fails_on_render_error(tmp_path, monkeypatch):
    import b2cq.workers.safetensors as st

    monkeypatch.setattr(st, "_build_recipe", lambda fmt, arch, model: [])
    monkeypatch.setattr(st, "is_frndobrain_class", lambda source_dir: True)

    class _RaisingStubTokenizer:
        def encode_chat_completion(self, req):
            raise ValueError("corpus does not match training/serving invariant")

    monkeypatch.setattr(st, "build_frndobrain_tokenizer", lambda source_dir: _RaisingStubTokenizer())

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "tekken.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="FrndoBrain-class calibration failed to render"):
        st.quantize_safetensors(
            model=_FakeModel(),
            tokenizer=None,
            format="W4A16_GPTQ",
            calibration=[{"messages": [{"role": "user", "content": "hi"}]}],
            output_dir=tmp_path / "out",
            source_dir=source_dir,
            log_cb=lambda msg: None,
        )


def test_quantize_safetensors_generic_uses_hf_chat_template(tmp_path, monkeypatch):
    import b2cq.workers.safetensors as st

    monkeypatch.setattr(st, "_build_recipe", lambda fmt, arch, model: [])
    monkeypatch.setattr(st, "is_frndobrain_class", lambda source_dir: False)

    oneshot_calls = []
    _install_fake_oneshot(monkeypatch, oneshot_calls)

    apply_chat_template_calls = []

    class _FakeHFTokenizer:
        def apply_chat_template(self, messages, tokenize=False):
            apply_chat_template_calls.append(messages)
            return "<rendered via HF chat template>"

    source_dir = tmp_path / "src"
    source_dir.mkdir()  # no tekken.json -- generic Mistral artifact
    output_dir = tmp_path / "out"

    logs = []
    st.quantize_safetensors(
        model=_FakeModel(),
        tokenizer=_FakeHFTokenizer(),
        format="W4A16_GPTQ",
        calibration=[{"messages": [{"role": "user", "content": "hi"}]}],
        output_dir=output_dir,
        source_dir=source_dir,
        log_cb=logs.append,
    )

    assert len(apply_chat_template_calls) == 1
    assert len(oneshot_calls) == 1
    assert oneshot_calls[0]["dataset"] == ["<rendered via HF chat template>"]
    assert (output_dir / "config.json").exists()
    assert any(output_dir.glob("*.safetensors"))
