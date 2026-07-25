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


def test_catalog_family_ids_and_properties_exact():
    """Pin complete id set per family and verify calibration + hardware_requirements properties."""
    # Exact id sets per family (catch any typos or drifts)
    gguf_k_ids = {"Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K"}
    gguf_i_ids = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL"}
    gguf_misc_ids = {"Q8_0", "F16", "BF16"}
    gguf_mmproj_ids = {"mmproj-f16"}
    safetensors_ids = {"W4A16_GPTQ", "W4A16_AWQ", "NVFP4", "FP8_E4M3", "FP8_E5M2"}

    # Verify exact id sets per family
    actual_gguf_k = {q.id for q in by_family(QuantFamily.GGUF_K)}
    actual_gguf_i = {q.id for q in by_family(QuantFamily.GGUF_I)}
    actual_gguf_misc = {q.id for q in by_family(QuantFamily.GGUF_MISC)}
    actual_gguf_mmproj = {q.id for q in by_family(QuantFamily.GGUF_MMPROJ)}
    actual_safetensors = {q.id for q in by_family(QuantFamily.SAFETENSORS)}

    assert actual_gguf_k == gguf_k_ids, f"GGUF_K ids must match exactly; expected {gguf_k_ids}, got {actual_gguf_k}"
    assert actual_gguf_i == gguf_i_ids, f"GGUF_I ids must match exactly; expected {gguf_i_ids}, got {actual_gguf_i}"
    assert actual_gguf_misc == gguf_misc_ids, f"GGUF_MISC ids must match exactly; expected {gguf_misc_ids}, got {actual_gguf_misc}"
    assert actual_gguf_mmproj == gguf_mmproj_ids, f"GGUF_MMPROJ ids must match exactly; expected {gguf_mmproj_ids}, got {actual_gguf_mmproj}"
    assert actual_safetensors == safetensors_ids, f"SAFETENSORS ids must match exactly; expected {safetensors_ids}, got {actual_safetensors}"

    # Every GGUF_I spec must have needs_calibration and needs_imatrix True
    for spec in by_family(QuantFamily.GGUF_I):
        assert spec.needs_calibration, f"GGUF_I spec {spec.id} must have needs_calibration=True"
        assert spec.needs_imatrix, f"GGUF_I spec {spec.id} must have needs_imatrix=True"

    # Every GGUF_K spec must have needs_calibration False
    for spec in by_family(QuantFamily.GGUF_K):
        assert not spec.needs_calibration, f"GGUF_K spec {spec.id} must have needs_calibration=False"

    # Every GGUF_MISC spec must have needs_calibration False
    for spec in by_family(QuantFamily.GGUF_MISC):
        assert not spec.needs_calibration, f"GGUF_MISC spec {spec.id} must have needs_calibration=False"

    # Every SAFETENSORS spec must have needs_calibration True and needs_imatrix False
    for spec in by_family(QuantFamily.SAFETENSORS):
        assert spec.needs_calibration, f"SAFETENSORS spec {spec.id} must have needs_calibration=True"
        assert not spec.needs_imatrix, f"SAFETENSORS spec {spec.id} must have needs_imatrix=False"

    # Only NVFP4 has nonempty hardware_requirements
    with_hardware_reqs = [q for q in CATALOG if q.hardware_requirements]
    assert len(with_hardware_reqs) == 1, f"Exactly 1 spec must have hardware_requirements; found {len(with_hardware_reqs)}"
    assert with_hardware_reqs[0].id == "NVFP4", f"Only NVFP4 should have hardware_requirements; got {with_hardware_reqs[0].id}"
