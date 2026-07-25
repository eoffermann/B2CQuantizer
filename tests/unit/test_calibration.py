"""Calibration loader — all three sources produce a list of message dicts."""
import json
from unittest.mock import patch

import pytest
from pathlib import Path
from b2cq.calibration import load_calibration, CalibrationSource
from b2cq.calibration import _BUNDLED_PATH


@pytest.fixture
def upload_jsonl(tmp_path) -> Path:
    p = tmp_path / "cal.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(50):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"msg {i}"}]}) + "\n")
    return p


@pytest.mark.skipif(
    not _BUNDLED_PATH.exists(),
    reason="bundled corpus not generated; run scripts/build_bundled_calibration.py",
)
def test_bundled_loads_default():
    samples = load_calibration(CalibrationSource(type="bundled"))
    assert isinstance(samples, list)
    assert len(samples) >= 500
    assert all("messages" in s for s in samples)


def test_upload_loads_from_file(upload_jsonl):
    samples = load_calibration(CalibrationSource(type="upload", local_path=upload_jsonl))
    assert len(samples) == 50
    assert samples[0]["messages"][0]["content"] == "msg 0"


def test_upload_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibration(CalibrationSource(type="upload", local_path=tmp_path / "nonexistent.jsonl"))


def test_hf_dataset_maps_text_and_passes_through_messages():
    fake_rows = [
        {"text": "hello world"},
        {"messages": [{"role": "user", "content": "already chat-shaped"}]},
    ]
    with patch("datasets.load_dataset", return_value=fake_rows) as mock_load:
        samples = load_calibration(
            CalibrationSource(type="hf_dataset", hf_dataset_id="some/dataset", hf_token="tok")
        )
    mock_load.assert_called_once_with("some/dataset", split="train", token="tok")
    assert samples == [
        {"messages": [{"role": "user", "content": "hello world"}]},
        {"messages": [{"role": "user", "content": "already chat-shaped"}]},
    ]


def test_hf_dataset_unreachable_propagates():
    with patch("datasets.load_dataset", side_effect=ConnectionError("no network")):
        with pytest.raises(ConnectionError):
            load_calibration(CalibrationSource(type="hf_dataset", hf_dataset_id="some/dataset"))
