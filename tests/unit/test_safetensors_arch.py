"""Architecture detection/validation for the safetensors worker.

Pure-Python coverage of the refuse-on-unsupported-architecture contract --
no torch/transformers/llmcompressor/compressed_tensors/mistral_common
required. `_detect_arch` and the validation inside `_build_recipe` both run
before any heavy import in `b2cq.workers.safetensors`, so these tests must
pass even in an environment with none of those packages installed.
"""
import json
import types

import pytest

from b2cq.workers.safetensors import _MULTIMODAL_ARCH, _TEXT_ARCH, _build_recipe, _detect_arch


def _write_config(tmp_path, architectures):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": architectures}), encoding="utf-8"
    )


def test_detect_arch_multimodal(tmp_path):
    _write_config(tmp_path, ["Mistral3ForConditionalGeneration"])
    assert _detect_arch(tmp_path) == _MULTIMODAL_ARCH


def test_detect_arch_text(tmp_path):
    _write_config(tmp_path, ["MistralForCausalLM"])
    assert _detect_arch(tmp_path) == _TEXT_ARCH


def test_detect_arch_unsupported_raises(tmp_path):
    _write_config(tmp_path, ["LlamaForCausalLM"])
    with pytest.raises(ValueError, match="LlamaForCausalLM"):
        _detect_arch(tmp_path)


def test_build_recipe_rejects_unsupported_arch_before_heavy_imports():
    # A fake model object -- if _build_recipe's validation didn't run before
    # its `llmcompressor`/`compressed_tensors` imports, this would blow up
    # with ModuleNotFoundError instead of the intended ValueError in an
    # environment without those packages installed.
    fake_model = types.SimpleNamespace(config=types.SimpleNamespace())
    with pytest.raises(ValueError, match="LlamaForCausalLM"):
        _build_recipe("W4A16_GPTQ", "LlamaForCausalLM", fake_model)
