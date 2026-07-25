"""Declarative catalog of every quant variant the app can produce.

Adding a new quant here does NOT automatically wire it into a worker; the
worker code needs the format string to know what to invoke. Kept declarative
so the UI can render selection groups without importing the worker code.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel


class QuantFamily(str, Enum):
    GGUF_K = "gguf_k"
    GGUF_I = "gguf_i"
    GGUF_MISC = "gguf_misc"
    GGUF_MMPROJ = "gguf_mmproj"
    SAFETENSORS = "safetensors"


class QuantSpec(BaseModel):
    id: str
    family: QuantFamily
    name: str
    format: str
    needs_calibration: bool
    needs_imatrix: bool = False
    notes: str = ""
    hardware_requirements: list[str] = []


CATALOG: list[QuantSpec] = [
    # ---- GGUF K-quants ----
    QuantSpec(id="Q2_K",     family=QuantFamily.GGUF_K, name="Q2_K",     format="Q2_K",     needs_calibration=False),
    QuantSpec(id="Q3_K_S",   family=QuantFamily.GGUF_K, name="Q3_K_S",   format="Q3_K_S",   needs_calibration=False),
    QuantSpec(id="Q3_K_M",   family=QuantFamily.GGUF_K, name="Q3_K_M",   format="Q3_K_M",   needs_calibration=False),
    QuantSpec(id="Q3_K_L",   family=QuantFamily.GGUF_K, name="Q3_K_L",   format="Q3_K_L",   needs_calibration=False),
    QuantSpec(id="Q4_K_S",   family=QuantFamily.GGUF_K, name="Q4_K_S",   format="Q4_K_S",   needs_calibration=False),
    QuantSpec(id="Q4_K_M",   family=QuantFamily.GGUF_K, name="Q4_K_M",   format="Q4_K_M",   needs_calibration=False),
    QuantSpec(id="Q5_K_S",   family=QuantFamily.GGUF_K, name="Q5_K_S",   format="Q5_K_S",   needs_calibration=False),
    QuantSpec(id="Q5_K_M",   family=QuantFamily.GGUF_K, name="Q5_K_M",   format="Q5_K_M",   needs_calibration=False),
    QuantSpec(id="Q6_K",     family=QuantFamily.GGUF_K, name="Q6_K",     format="Q6_K",     needs_calibration=False),

    # ---- GGUF I-quants (all need imatrix) ----
    QuantSpec(id="IQ1_S",    family=QuantFamily.GGUF_I, name="IQ1_S",    format="IQ1_S",    needs_calibration=True, needs_imatrix=True, notes="Extreme compression; noticeable quality drop"),
    QuantSpec(id="IQ1_M",    family=QuantFamily.GGUF_I, name="IQ1_M",    format="IQ1_M",    needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ2_XXS",  family=QuantFamily.GGUF_I, name="IQ2_XXS",  format="IQ2_XXS",  needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ2_XS",   family=QuantFamily.GGUF_I, name="IQ2_XS",   format="IQ2_XS",   needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ2_S",    family=QuantFamily.GGUF_I, name="IQ2_S",    format="IQ2_S",    needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ2_M",    family=QuantFamily.GGUF_I, name="IQ2_M",    format="IQ2_M",    needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ3_XXS",  family=QuantFamily.GGUF_I, name="IQ3_XXS",  format="IQ3_XXS",  needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ3_XS",   family=QuantFamily.GGUF_I, name="IQ3_XS",   format="IQ3_XS",   needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ3_S",    family=QuantFamily.GGUF_I, name="IQ3_S",    format="IQ3_S",    needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ3_M",    family=QuantFamily.GGUF_I, name="IQ3_M",    format="IQ3_M",    needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ4_XS",   family=QuantFamily.GGUF_I, name="IQ4_XS",   format="IQ4_XS",   needs_calibration=True, needs_imatrix=True),
    QuantSpec(id="IQ4_NL",   family=QuantFamily.GGUF_I, name="IQ4_NL",   format="IQ4_NL",   needs_calibration=True, needs_imatrix=True),

    # ---- GGUF misc (no calibration) ----
    QuantSpec(id="Q8_0",     family=QuantFamily.GGUF_MISC, name="Q8_0",  format="Q8_0",     needs_calibration=False, notes="Reference-quality 8-bit"),
    QuantSpec(id="F16",      family=QuantFamily.GGUF_MISC, name="F16",   format="F16",      needs_calibration=False, notes="Repack; no quantization"),
    QuantSpec(id="BF16",     family=QuantFamily.GGUF_MISC, name="BF16",  format="BF16",     needs_calibration=False, notes="Repack; no quantization"),

    # ---- GGUF mmproj (multimodal projector) ----
    QuantSpec(id="mmproj-f16", family=QuantFamily.GGUF_MMPROJ, name="mmproj-f16", format="mmproj", needs_calibration=False, notes="Vision projector for multimodal models"),

    # ---- safetensors ----
    QuantSpec(id="W4A16_GPTQ", family=QuantFamily.SAFETENSORS, name="W4A16 GPTQ",  format="W4A16_GPTQ", needs_calibration=True, notes="Symmetric g128; vLLM gptq_marlin kernel"),
    QuantSpec(id="W4A16_AWQ",  family=QuantFamily.SAFETENSORS, name="W4A16 AWQ",   format="W4A16_AWQ",  needs_calibration=True, notes="Asymmetric g128; vLLM awq_marlin kernel"),
    QuantSpec(id="NVFP4",      family=QuantFamily.SAFETENSORS, name="NVFP4",       format="NVFP4",      needs_calibration=True, hardware_requirements=["blackwell"], notes="4-bit weights + activations; Blackwell tensor cores only"),
    QuantSpec(id="FP8_E4M3",   family=QuantFamily.SAFETENSORS, name="FP8 E4M3",    format="FP8_E4M3",   needs_calibration=True, notes="8-bit float; Hopper+ compute, Ampere storage"),
    QuantSpec(id="FP8_E5M2",   family=QuantFamily.SAFETENSORS, name="FP8 E5M2",    format="FP8_E5M2",   needs_calibration=True, notes="8-bit float, larger exponent range"),
]

_BY_ID = {q.id: q for q in CATALOG}


def get(quant_id: str) -> QuantSpec:
    if quant_id not in _BY_ID:
        raise KeyError(f"unknown quant id: {quant_id!r}")
    return _BY_ID[quant_id]


def by_family(family: QuantFamily) -> list[QuantSpec]:
    return [q for q in CATALOG if q.family == family]
