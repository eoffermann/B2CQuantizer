"""safetensors quantization worker via llm-compressor.

Handles the five safetensors QuantSpec formats (W4A16_GPTQ, W4A16_AWQ,
NVFP4, FP8_E4M3, FP8_E5M2) for Mistral/Mistral3 source models.

This module MUST import cleanly without `llm-compressor`, `torch`, or a
GPU present — those are Docker-image-only (see `Dockerfile` / RunPod pod
setup). Every import of those packages therefore lives inside a function
body, never at module scope.

Mistral3 (multimodal, `Mistral3ForConditionalGeneration`) needs special
handling that plain Mistral (`MistralForCausalLM`) doesn't:
  - it loads via `AutoModelForImageTextToText`, not `AutoModelForCausalLM`;
  - the vision tower / projector must be excluded from quantization
    (`_IGNORE_MULTIMODAL`);
  - `llm-compressor`'s default AWQ mapping glob matches all decoder layers
    into a single `match_modules_set` and errors out on the wrapped
    `language_model.model.layers.*` module tree — AWQ needs one explicit
    `AWQMapping` per decoder layer instead (`_build_awq_mappings_scoped`).
This is the reason the FrndoBrain reference implementation
(`frndobrain_quantize.py`) works and a naive llm-compressor recipe doesn't.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

_MULTIMODAL_ARCH = "Mistral3ForConditionalGeneration"
_TEXT_ARCH = "MistralForCausalLM"

_IGNORE_MULTIMODAL = ["re:.*vision_tower.*", "re:.*multi_modal_projector.*", "lm_head"]
_IGNORE_TEXT = ["lm_head"]

# Target calibration sample count (SPEC: 512). If the caller-provided
# calibration set is larger, it's truncated to this many samples before
# being handed to llm-compressor's oneshot runner.
DEFAULT_NUM_CALIBRATION_SAMPLES = 512

# Mistral-native tokenizer/runtime artifacts that llm-compressor's oneshot
# save doesn't carry forward but that downstream vLLM serving of a
# FrndoBrain-style artifact needs.
_CARRY_FORWARD_ARTIFACTS = ("tekken.json", "params.json", "chat_template.jinja")


def _detect_arch(source_dir: Path) -> str:
    cfg = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    arch = cfg.get("architectures", [None])[0]
    if arch not in (_MULTIMODAL_ARCH, _TEXT_ARCH):
        raise ValueError(
            f"Unsupported architecture: {arch!r}. B2CQuantizer v1 supports only Mistral/Mistral3."
        )
    return arch


def load_model_for_safetensors(source_dir: Path, log_cb: Callable[[str], None]):
    """Load the source model + tokenizer once per lane.

    Dispatches on the architecture recorded in `<source_dir>/config.json`:
    `Mistral3ForConditionalGeneration` (multimodal) loads via
    `AutoModelForImageTextToText`; `MistralForCausalLM` (text-only) loads via
    `AutoModelForCausalLM`. The caller (Lane A orchestrator) is expected to
    call this once and pass the returned `(model, tokenizer)` into repeated
    `quantize_safetensors` calls, one per selected safetensors format, so the
    (potentially 24B-parameter) model is only loaded onto the GPU a single
    time.
    """
    source_dir = Path(source_dir)
    arch = _detect_arch(source_dir)
    log_cb(f"Detected architecture: {arch}")

    import torch
    from transformers import AutoTokenizer

    if arch == _MULTIMODAL_ARCH:
        from transformers import AutoModelForImageTextToText as ModelClass
    else:
        from transformers import AutoModelForCausalLM as ModelClass

    model = ModelClass.from_pretrained(source_dir, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    log_cb(f"Loaded {arch} on {next(model.parameters()).device}")
    return model, tokenizer


def _build_awq_mappings_scoped(model):
    """Explicit per-decoder-layer AWQ mappings for Mistral3ForConditionalGeneration.

    llm-compressor's default mappings glob decoder-layer patterns across all
    layers into one `match_modules_set` group and error out ("AWQ needs to
    match a single smooth layer per set") on Mistral3's wrapped
    `language_model.model.layers.*` module tree. This builds one mapping set
    per decoder layer explicitly instead, covering the four smoothing/balance
    blocks llm-compressor's default AWQMapping list would otherwise cover in
    aggregate: attention QKV smoothing, MLP gate/up smoothing, attention
    output balance, and MLP down-proj balance.
    """
    from llmcompressor.modifiers.awq import AWQMapping

    n_layers = model.config.text_config.num_hidden_layers  # 40 for Mistral-Small-3.2-24B
    mappings = []
    for i in range(n_layers):
        prefix = f"language_model.model.layers.{i}"
        # Attention smoothing block
        mappings.append(
            AWQMapping(
                smooth_layer=f"{prefix}.input_layernorm",
                balance_layers=[
                    f"{prefix}.self_attn.q_proj",
                    f"{prefix}.self_attn.k_proj",
                    f"{prefix}.self_attn.v_proj",
                ],
            )
        )
        # MLP smoothing block
        mappings.append(
            AWQMapping(
                smooth_layer=f"{prefix}.post_attention_layernorm",
                balance_layers=[f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"],
            )
        )
        # Attention output balance
        mappings.append(
            AWQMapping(
                smooth_layer=f"{prefix}.self_attn.v_proj",
                balance_layers=[f"{prefix}.self_attn.o_proj"],
            )
        )
        # MLP output balance
        mappings.append(
            AWQMapping(
                smooth_layer=f"{prefix}.mlp.up_proj",
                balance_layers=[f"{prefix}.mlp.down_proj"],
            )
        )
    return mappings


def _build_recipe(format: str, arch: str, model):
    """Build the llm-compressor recipe (list of Modifiers) for `format`.

    All five recipes are built exclusively from PRESET quantization schemes
    (compressed_tensors/quantization/quant_scheme.py `PRESET_SCHEMES`) plus
    the modifier-level `targets`/`ignore`/`mappings` fields that actually
    exist on GPTQModifier/AWQModifier/QuantizationModifier. Every Modifier in
    llm-compressor 0.10 has pydantic `model_config = ConfigDict(extra="forbid")`
    (inherited via QuantizationMixin / QuantizationArgs / QuantizationScheme),
    so passing any kwarg that isn't a real field (e.g. the old `group_size=`,
    `sym=`, `weights=`, `observer_options=`) raises a ValidationError before
    any GPU work starts. See wheel citations below for each format.
    """
    if arch not in (_MULTIMODAL_ARCH, _TEXT_ARCH):
        raise ValueError(
            f"Unsupported architecture: {arch!r}. B2CQuantizer v1 supports only Mistral/Mistral3."
        )

    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    from llmcompressor.modifiers.awq import AWQModifier

    ignore = _IGNORE_MULTIMODAL if arch == _MULTIMODAL_ARCH else _IGNORE_TEXT

    if format == "W4A16_GPTQ":
        # GPTQModifier(Modifier, QuantizationMixin) has NO `group_size`/`sym`
        # fields (llmcompressor/modifiers/quantization/gptq/base.py -- its own
        # fields are sequential_targets/block_size/dampening_frac/actorder/
        # offload_hessians; the rest come from QuantizationMixin: config_groups/
        # targets/ignore/scheme/kv_cache_scheme/*_observer). The preset name
        # "W4A16" already encodes 4-bit int, symmetric, group_size=128
        # (compressed_tensors/quantization/quant_scheme.py: W4A16 = dict(
        #   weights=QuantizationArgs(num_bits=4, type=INT, strategy=GROUP,
        #   group_size=128, symmetric=True)) ) -- the preset alone is correct.
        return [GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)]

    if format == "W4A16_AWQ":
        # AWQModifier(Modifier, QuantizationMixin) (modifiers/awq/base.py)
        # takes `scheme`/`config_groups` + `mappings` directly -- it IS a
        # QuantizationMixin, so it does not need a separate QuantizationModifier
        # paired after it (see AWQModifier docstring's example recipe, which
        # shows a single AWQModifier with `mappings` + `config_groups`). Preset
        # "W4A16_ASYM" = 4-bit int, group_size=128, symmetric=False -- the
        # asymmetric counterpart of W4A16, matching AWQ's asymmetric default.
        awq_kwargs = {"scheme": "W4A16_ASYM", "ignore": ignore}
        if arch == _MULTIMODAL_ARCH:
            # Default AWQ mappings error out on Mistral3's wrapped decoder
            # layer tree -- build explicit per-layer mappings instead. For
            # text-only Mistral, leaving `mappings` unset uses llm-compressor's
            # defaults (AWQ_MAPPING_REGISTRY["MistralForCausalLM"]), which work
            # fine there.
            awq_kwargs["mappings"] = _build_awq_mappings_scoped(model)
        return [AWQModifier(**awq_kwargs)]

    if format == "NVFP4":
        # No GPTQ/AWQ pass: FP4 rounding doesn't benefit from Hessian-based
        # or activation-smoothing calibration the way integer schemes do.
        # Preset "NVFP4" exists in PRESET_SCHEMES (quant_scheme.py): 4-bit
        # float, TENSOR_GROUP strategy, group_size=16, FP8_E4M3 scale/zp dtype,
        # with a matching LOCAL-dynamic input_activations entry.
        return [QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)]

    if format == "FP8_E4M3":
        # Preset "FP8" (quant_scheme.py) is 8-bit float TENSOR-strategy
        # weights + static TENSOR-strategy input_activations. compressed_tensors
        # 0.10's QuantizationArgs has NO `observer_options` field (extra=forbid)
        # and its float8 dtype is hardcoded to FP8_E4M3_DATA.dtype everywhere
        # (pytorch_dtype()/round_to_quantized_type_args in quant_args.py) --
        # there is no dtype-selection knob, so "FP8" *is* E4M3.
        return [QuantizationModifier(targets="Linear", scheme="FP8", ignore=ignore)]

    if format == "FP8_E5M2":
        # compressed_tensors/quantization/quant_args.py defines FP8_E4M3_DATA
        # only -- no FP8_E5M2_DATA class exists, and both pytorch_dtype() and
        # round_to_quantized_type_args() hardcode FP8_E4M3_DATA.dtype for any
        # 8-bit float QuantizationArgs. There is no preset, no dtype param, and
        # no code path in llm-compressor 0.10 / compressed-tensors that can
        # produce e5m2 weights. This is a genuine gap, not a naming mismatch.
        raise ValueError("FP8 E5M2 is not supported by llm-compressor 0.10; use FP8 E4M3")

    raise ValueError(f"unsupported safetensors format: {format!r}")


def is_frndobrain_class(source_dir: Path) -> bool:
    """Detect FrndoBrain-class artifact by presence of tekken.json in source.

    FrndoBrain merges carry tekken.json byte-identical to the base Mistral repo
    per the tokenizer-invariant training contract. Its presence marks an artifact
    whose calibration MUST go through mistral-common with the tool-placement flip.
    See SPEC §14 for the full invariant set.
    """
    return (Path(source_dir) / "tekken.json").exists()


# Pin locked to what the FrndoBrain serving stack ships. The private attribute
# below is version-fragile -- it was `_user_message_position_to_encode_tools` in
# mistral-common 1.9.x and renamed to `_message_position_to_encode_tools_settings`
# in 1.11.x. Silent divergence between calibration-time and serving-time
# tokenization gets baked into the quantized weights; hence the hard-fails.
_EXPECTED_MISTRAL_COMMON = "1.11.5"
_TOOL_PLACEMENT_ATTR = "_message_position_to_encode_tools_settings"


def build_frndobrain_tokenizer(source_dir: Path):
    """Build a mistral-common MistralTokenizer with the tools-at-first-user-turn
    flip, pinned to the version the serving stack ships. Fails loudly if
    version or attribute has drifted."""
    import mistral_common
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.tokens.tokenizers.base import UserMessagePosition
    from mistral_common.protocol.instruct.validator import ValidationMode

    if mistral_common.__version__ != _EXPECTED_MISTRAL_COMMON:
        raise RuntimeError(
            f"mistral-common version mismatch: got {mistral_common.__version__}, "
            f"expected {_EXPECTED_MISTRAL_COMMON}. FrndoBrain-class calibration "
            "is version-locked to match training + serving; audit the InstructTokenizer "
            "source and update the pin (see SPEC §14 for the drift history)."
        )
    # Load from the local source_dir (tekken.json lives here).
    # NOTE: `serving` mode (not `finetuning`) is deliberate -- confirmed repro:
    # in `finetuning` mode, mistral-common's MistralRequestValidator rejects
    # ANY calibration sample that doesn't end on an assistant turn ("Expected
    # last role Assistant for finetuning but got user" --
    # mistral_common/protocol/instruct/validator.py:_validate_last_message),
    # which is the entire bundled corpus's user-only samples. `serving` mode
    # rendering is serving-time tokenization (what the model actually sees at
    # inference), and correctly accepts user-terminated calibration samples.
    # See `_render_calibration_frndobrain` for the corresponding
    # `model=`/`continue_final_message=` handling this mode switch requires.
    tok = MistralTokenizer.from_file(str(Path(source_dir) / "tekken.json"),
                                      mode=ValidationMode.serving)
    it = tok.instruct_tokenizer
    if not hasattr(it, _TOOL_PLACEMENT_ATTR):
        raise RuntimeError(
            f"expected attribute {_TOOL_PLACEMENT_ATTR!r} not present on "
            f"{type(it).__name__}; attribute has moved again -- audit InstructTokenizer "
            "source and update the pin."
        )
    setattr(it, _TOOL_PLACEMENT_ATTR, UserMessagePosition.first)
    return tok


def _render_calibration_frndobrain(mtok, sample: dict) -> str:
    """Render one calibration sample via mistral-common. Sample shape is
    {messages: [...], tools: [...] or omitted}. Returns plain text -- llm-compressor
    will re-tokenize with the model's HF tokenizer, which for FrndoBrain artifacts
    produces the same IDs since tokenizer.json is regenerated from tekken.json
    at training time. What matters is the CONVERSATION STRUCTURE is rendered by
    mistral-common (so [AVAILABLE_TOOLS] lands at first user turn, tool_calls
    render with V11's [CALL_ID] format, etc.)."""
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    messages = sample["messages"]
    # `serving`-mode validation (see build_frndobrain_tokenizer) requires
    # `model` to be set (MistralRequestValidator.validate_request raises
    # "Model name parameter is required for serving mode" otherwise -- the
    # value itself is not otherwise validated/used for rendering). It also
    # requires the request to end on a user/tool turn UNLESS
    # `continue_final_message=True` is set, in which case an assistant-
    # terminated conversation renders as a continuation of that assistant
    # turn (no closing tag after the assistant content) rather than being
    # rejected -- so both user-terminated and assistant-terminated
    # calibration samples render correctly.
    continue_final_message = bool(messages) and messages[-1].get("role") == "assistant"
    req = ChatCompletionRequest(
        messages=messages,
        tools=sample.get("tools"),
        model="b2cq-frndobrain-calibration",
        continue_final_message=continue_final_message,
    )
    return mtok.encode_chat_completion(req).text


def _render_calibration_hf(tokenizer, sample: dict) -> str:
    """Fallback render for non-FrndoBrain Mistral models. Uses HF chat template."""
    return tokenizer.apply_chat_template(sample["messages"], tokenize=False)


def quantize_safetensors(
    model,
    tokenizer,
    format: str,
    calibration: list[dict],
    output_dir: Path,
    source_dir: Path,
    log_cb: Callable[[str], None],
) -> None:
    """Run one llm-compressor oneshot quantization pass for `format`.

    `model`/`tokenizer` come from `load_model_for_safetensors`, loaded once
    per lane and reused across formats. `source_dir` is the on-disk source
    model directory and is REQUIRED -- it is where artifact-class detection
    (`is_frndobrain_class`) and the mistral-common tekken.json tokenizer are
    read from, and where post-quant tokenizer/runtime-artifact carry-forward
    pulls from. See SPEC §14 for why the calibration render path depends on
    artifact class rather than best-effort import success.
    """
    if not calibration:
        raise ValueError("calibration is empty")

    source_dir = Path(source_dir)

    arch = model.config.architectures[0]
    # _build_recipe validates `arch` against the supported set before doing
    # any of its own heavy imports, so this raises ValueError early (and
    # cheaply) for an unsupported architecture rather than failing deep
    # inside llm-compressor after calibration rendering / model prep.
    recipe = _build_recipe(format, arch, model)

    samples = calibration[:DEFAULT_NUM_CALIBRATION_SAMPLES]
    log_cb(
        f"Preparing {len(samples)} calibration samples "
        f"(of {len(calibration)} provided, target {DEFAULT_NUM_CALIBRATION_SAMPLES})"
    )

    # Detect artifact class and pick the calibration render path. Per SPEC
    # §14, this dispatch is NOT best-effort/try-except-on-import: the render
    # path is determined by whether the artifact IS FrndoBrain-class, not by
    # whether mistral-common happens to be importable.
    if is_frndobrain_class(source_dir):
        log_cb(f"FrndoBrain-class artifact detected (tekken.json present in {source_dir})")
        log_cb("Calibration rendering via mistral-common with tools-at-first-user-turn flip")
        mtok = build_frndobrain_tokenizer(source_dir)
        # If ANY calibration sample fails to render, fail loudly -- the
        # user's calibration corpus does not match the training invariant
        # and would produce calibration/inference divergence baked into the
        # quantized weights.
        try:
            texts = [_render_calibration_frndobrain(mtok, s) for s in samples]
        except Exception as e:
            raise RuntimeError(
                f"FrndoBrain-class calibration failed to render at least one sample "
                f"via mistral-common: {type(e).__name__}: {e}. This means your "
                "calibration corpus does not match the training/serving invariant. "
                "For best results, calibrate on the same data (or a same-shape subset) "
                "as was used for training. See SPEC §14 for the invariants that must hold."
            ) from e
    else:
        log_cb(f"Generic Mistral artifact (no tekken.json in {source_dir})")
        log_cb("Calibration rendering via HF chat template")
        texts = [_render_calibration_hf(tokenizer, s) for s in samples]

    # `llmcompressor.transformers` no longer re-exports `oneshot` in 0.10
    # (llmcompressor/transformers/__init__.py only does `from .utils import *`
    # + `from .data import TextGenerationDataset`) -- the verified path is the
    # top-level package, which re-exports it via
    # llmcompressor/entrypoints/__init__.py -> llmcompressor/__init__.py
    # (`from llmcompressor.entrypoints import Oneshot, oneshot, model_free_ptq`).
    from llmcompressor import oneshot
    from datasets import Dataset

    # oneshot()'s `dataset` param is typed `str | Dataset | DatasetDict | None`
    # (llmcompressor/entrypoints/oneshot.py:266) -- a bare list[str] is not one
    # of those. Internally, `get_processed_dataset` -> `TextGenerationDataset.
    # __call__` (llmcompressor/transformers/data/base.py:95-101) only calls
    # `self.load_dataset()` (which needs a HF dataset id/path) when
    # `isinstance(dataset, str)`; otherwise it treats the passed object
    # directly as a `datasets.Dataset` and looks for `dataset_args.text_column`
    # (default "text") to feed the tokenizer. So we wrap our rendered strings
    # in a one-column `datasets.Dataset` under the "text" column.
    log_cb(f"Starting oneshot quantization: {format}")
    oneshot(
        model=model,
        recipe=recipe,
        dataset=Dataset.from_dict({"text": texts}),
        output_dir=str(output_dir),
        num_calibration_samples=len(texts),
        max_seq_length=2048,
    )

    output_dir = Path(output_dir)
    if not (output_dir / "config.json").exists() or not any(output_dir.glob("*.safetensors")):
        raise RuntimeError(
            f"{format} quantization produced no valid output in {output_dir} "
            "(missing config.json and/or *.safetensors files)"
        )

    if source_dir.exists():
        from shutil import copy

        for name in _CARRY_FORWARD_ARTIFACTS:
            src = source_dir / name
            if src.exists():
                copy(src, output_dir / name)
                log_cb(f"Carried forward {name}")

    log_cb(f"Wrote {format} quant to {output_dir}")
