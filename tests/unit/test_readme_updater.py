"""README updater: canonical ## Quantizations section replacement."""
from datetime import datetime, timezone
from b2cq.readme_updater import _render_section, _splice_section


def test_render_section_produces_table():
    from b2cq.job_model import Job, QuantResult, QuantStatus
    from b2cq.calibration import CalibrationSource
    job = Job(
        id="j1", source_model="user/model", owner="user",
        quants=[
            QuantResult(quant_id="Q4_K_M", status=QuantStatus.DONE, lane="B",
                        upload_url="https://huggingface.co/user/model-GGUF/tree/main",
                        repo_id="user/model-GGUF"),
            QuantResult(quant_id="W4A16_GPTQ", status=QuantStatus.DONE, lane="A",
                        upload_url="https://huggingface.co/user/model-W4A16_GPTQ",
                        repo_id="user/model-W4A16_GPTQ"),
            QuantResult(quant_id="NVFP4", status=QuantStatus.FAILED, lane="A"),
        ],
        calibration=CalibrationSource(type="bundled"), private=False,
        update_source_readme=True, started_at=datetime.now(timezone.utc),
    )
    section = _render_section(job)
    assert "## Quantizations" in section
    assert "user/model-GGUF" in section
    assert "user/model-W4A16_GPTQ" in section
    assert "NVFP4" not in section  # failed quants excluded


def test_splice_appends_when_absent():
    original = "# My Model\n\nSome description.\n"
    new = _splice_section(original, "## Quantizations\n\n| tbl |\n")
    assert new.endswith("## Quantizations\n\n| tbl |\n")
    assert "Some description." in new


def test_splice_replaces_when_present():
    original = "# My Model\n\n## Quantizations\n\n| old |\n\n## Other\n\nfoo\n"
    new = _splice_section(original, "## Quantizations\n\n| new |\n")
    assert "| new |" in new
    assert "| old |" not in new
    assert "## Other" in new
    assert "foo" in new
