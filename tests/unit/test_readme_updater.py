"""README updater: canonical ## Quantizations section replacement."""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from huggingface_hub.errors import EntryNotFoundError

from b2cq.hf_client import HFClient
from b2cq.readme_updater import _render_section, _splice_section, update_source_readme


def _make_job():
    from b2cq.job_model import Job, QuantResult, QuantStatus
    from b2cq.calibration import CalibrationSource
    return Job(
        id="j1", source_model="user/model", owner="user",
        quants=[
            QuantResult(quant_id="Q4_K_M", status=QuantStatus.DONE, lane="B",
                        upload_url="https://huggingface.co/user/model-GGUF/tree/main",
                        repo_id="user/model-GGUF"),
        ],
        calibration=CalibrationSource(type="bundled"), private=False,
        update_source_readme=True, started_at=datetime.now(timezone.utc),
    )


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


def test_update_source_readme_splices_existing_readme(tmp_path):
    job = _make_job()
    readme_path = tmp_path / "README.md"
    readme_path.write_text("# My Model\n\nSome description.\n", encoding="utf-8")

    hf_client = MagicMock(spec=HFClient)
    hf_client.download_file.return_value = readme_path
    hf_client.update_file.return_value = "https://huggingface.co/user/model/commit/abc"

    result = update_source_readme(job, hf_client)

    hf_client.download_file.assert_called_once_with("user/model", "README.md")
    hf_client.update_file.assert_called_once()
    _, kwargs = hf_client.update_file.call_args
    assert kwargs["repo_id"] == "user/model"
    assert kwargs["path_in_repo"] == "README.md"
    assert "Some description." in kwargs["content"]
    assert "## Quantizations" in kwargs["content"]
    assert result == "https://huggingface.co/user/model/commit/abc"


def test_update_source_readme_falls_back_to_stub_on_entry_not_found():
    job = _make_job()

    hf_client = MagicMock(spec=HFClient)
    hf_client.download_file.side_effect = EntryNotFoundError("README.md not found")
    hf_client.update_file.return_value = "https://huggingface.co/user/model/commit/abc"

    result = update_source_readme(job, hf_client)

    hf_client.update_file.assert_called_once()
    _, kwargs = hf_client.update_file.call_args
    assert "# user/model" in kwargs["content"]
    assert "## Quantizations" in kwargs["content"]
    assert result == "https://huggingface.co/user/model/commit/abc"


def test_update_source_readme_propagates_generic_errors():
    job = _make_job()

    hf_client = MagicMock(spec=HFClient)
    hf_client.download_file.side_effect = ConnectionError("network is down")

    with pytest.raises(ConnectionError, match="network is down"):
        update_source_readme(job, hf_client)

    hf_client.update_file.assert_not_called()


def test_update_source_readme_is_noop_when_section_already_current(tmp_path):
    job = _make_job()
    # Must match exactly what _splice_section would produce (it always leaves a
    # blank line before EOF), or the "no change" comparison won't hold.
    current_content = "# My Model\n\n" + _render_section(job).rstrip() + "\n\n"
    readme_path = tmp_path / "README.md"
    readme_path.write_text(current_content, encoding="utf-8")

    hf_client = MagicMock(spec=HFClient)
    hf_client.download_file.return_value = readme_path

    result = update_source_readme(job, hf_client)

    assert result == "no-op"
    hf_client.update_file.assert_not_called()
