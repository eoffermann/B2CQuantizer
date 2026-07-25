"""Calibration loader — all three sources produce a list of message dicts."""
import json, pytest
from pathlib import Path
from b2cq.calibration import load_calibration, CalibrationSource


@pytest.fixture
def upload_jsonl(tmp_path) -> Path:
    p = tmp_path / "cal.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(50):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"msg {i}"}]}) + "\n")
    return p


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
