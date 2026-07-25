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
    if arch not in (_MULTIMODAL_ARCH, _TEXT_ARCH):
        raise ValueError(
            f"Unsupported architecture: {arch!r}. B2CQuantizer v1 supports only Mistral/Mistral3."
        )

    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    from llmcompressor.modifiers.awq import AWQModifier
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationScheme,
        QuantizationStrategy,
        QuantizationType,
    )

    ignore = _IGNORE_MULTIMODAL if arch == _MULTIMODAL_ARCH else _IGNORE_TEXT

    if format == "W4A16_GPTQ":
        return [GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore, group_size=128, sym=True)]

    if format == "W4A16_AWQ":
        awq_kwargs = {"ignore": ignore}
        if arch == _MULTIMODAL_ARCH:
            # Default AWQ mappings error out on Mistral3's wrapped decoder
            # layer tree -- build explicit per-layer mappings instead. For
            # text-only Mistral, leaving `mappings` unset uses llm-compressor's
            # defaults, which work fine there.
            awq_kwargs["mappings"] = _build_awq_mappings_scoped(model)
        weight_quant = QuantizationModifier(
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=4,
                        type=QuantizationType.INT,
                        symmetric=False,
                        strategy=QuantizationStrategy.GROUP,
                        group_size=128,
                        actorder=None,
                    ),
                )
            }
        )
        return [AWQModifier(**awq_kwargs), weight_quant]

    if format == "NVFP4":
        # No GPTQ/AWQ pass: FP4 rounding doesn't benefit from Hessian-based
        # or activation-smoothing calibration the way integer schemes do.
        return [QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)]

    if format in ("FP8_E4M3", "FP8_E5M2"):
        # E4M3 vs E5M2 differ only in the float8 dtype passed through
        # observer_options; exact param name/values should be re-verified
        # against the installed llm-compressor version in the GPU image.
        dtype = "float8_e4m3fn" if format == "FP8_E4M3" else "float8_e5m2"
        weights_args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.CHANNEL,
            observer_options={"dtype": dtype},
        )
        return [QuantizationModifier(targets="Linear", scheme="FP8", ignore=ignore, weights=weights_args)]

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
    tok = MistralTokenizer.from_file(str(Path(source_dir) / "tekken.json"),
                                      mode=ValidationMode.finetuning)
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

    req = ChatCompletionRequest(messages=sample["messages"], tools=sample.get("tools"))
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

    from llmcompressor.transformers import oneshot

    log_cb(f"Starting oneshot quantization: {format}")
    oneshot(
        model=model,
        recipe=recipe,
        dataset=texts,
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
