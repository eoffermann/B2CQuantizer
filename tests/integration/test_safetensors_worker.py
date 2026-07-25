"""GPU integration test — skipped without GPU_INTEGRATION set.

Uses a tiny stub model (TinyLlama or a synthetic Mistral3-shaped test model)
so the test runs in reasonable time. Full-model verification is manual
via the smoke procedure (tests/smoke/README.md)."""
import pytest
import os
from pathlib import Path

pytestmark = pytest.mark.skipif(
    "GPU_INTEGRATION" not in os.environ,
    reason="Set GPU_INTEGRATION=1 to run GPU integration tests",
)


def test_quantize_w4a16_gptq_smoke(tmp_path):
    from b2cq.workers.safetensors import load_model_for_safetensors, quantize_safetensors
    # Use a tiny known-Mistral model for smoke — swap for a permanently
    # available small mistral-family model when this test is enabled.
    source = Path("/workspace/hf-cache/mistralai/Mistral-7B-v0.1-tiny")  # fixture-provided
    if not source.exists():
        pytest.skip("tiny Mistral fixture not available")

    logs = []
    model, tok = load_model_for_safetensors(source, log_cb=logs.append)
    output = tmp_path / "gptq"
    quantize_safetensors(model, tok, "W4A16_GPTQ",
                         calibration=[{"messages": [{"role": "user", "content": "hi"}]}] * 8,
                         output_dir=output, log_cb=logs.append)
    assert (output / "config.json").exists()
    assert any(f.name.endswith(".safetensors") for f in output.iterdir())
