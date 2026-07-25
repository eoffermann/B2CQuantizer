"""Catalog must expose the full v1 quant list with correct metadata."""
import pytest
from b2cq.quant_catalog import CATALOG, QuantFamily, get, by_family


def test_catalog_has_expected_counts():
    assert len(by_family(QuantFamily.GGUF_K)) == 9   # Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K
    assert len(by_family(QuantFamily.GGUF_I)) == 12  # IQ1_S, IQ1_M, IQ2_XXS/XS/S/M, IQ3_XXS/XS/S/M, IQ4_XS/NL
    assert len(by_family(QuantFamily.GGUF_MISC)) == 3   # Q8_0, F16, BF16
    assert len(by_family(QuantFamily.GGUF_MMPROJ)) == 1  # mmproj-f16
    assert len(by_family(QuantFamily.SAFETENSORS)) == 5  # W4A16_GPTQ, W4A16_AWQ, NVFP4, FP8_E4M3, FP8_E5M2
    assert len(CATALOG) == 30


def test_calibration_flags_correct():
    assert not get("Q4_K_M").needs_calibration
    assert not get("Q8_0").needs_calibration
    assert get("IQ4_XS").needs_calibration and get("IQ4_XS").needs_imatrix
    assert get("W4A16_GPTQ").needs_calibration and not get("W4A16_GPTQ").needs_imatrix
    assert get("NVFP4").needs_calibration
    assert get("FP8_E4M3").needs_calibration


def test_hardware_gates():
    assert "blackwell" in get("NVFP4").hardware_requirements
    assert "blackwell" not in get("W4A16_GPTQ").hardware_requirements


def test_ids_unique():
    ids = [q.id for q in CATALOG]
    assert len(ids) == len(set(ids)), f"duplicate ids: {[i for i in ids if ids.count(i) > 1]}"


def test_get_raises_on_unknown():
    with pytest.raises(KeyError):
        get("BOGUS_QUANT_ID")
