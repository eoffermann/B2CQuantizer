"""HFClient thin wrapper — token handling + method signatures."""
from unittest.mock import MagicMock, patch
import pytest
from b2cq.hf_client import HFClient


def test_repr_never_leaks_token():
    c = HFClient(token="hf_supersecret_xyz")
    r = repr(c)
    assert "hf_supersecret" not in r
    assert "xyz" not in r


def test_close_wipes_token():
    c = HFClient(token="hf_supersecret")
    assert c._token is not None
    c.close()
    assert c._token is None


def test_calls_after_close_raise():
    c = HFClient(token="hf_supersecret")
    c.close()
    with pytest.raises(RuntimeError, match="closed"):
        c.whoami()


@patch("b2cq.hf_client.HfApi")
def test_whoami_forwards_token(HfApi):
    inst = MagicMock()
    inst.whoami.return_value = {"name": "eddie"}
    HfApi.return_value = inst
    c = HFClient(token="hf_x")
    result = c.whoami()
    HfApi.assert_called_once_with(token="hf_x")
    assert result == {"name": "eddie"}


@patch("b2cq.hf_client.HfApi")
def test_upload_folder_forwards_args(HfApi):
    inst = MagicMock()
    inst.upload_folder.return_value = "https://huggingface.co/user/repo/commit/abc"
    HfApi.return_value = inst
    c = HFClient(token="hf_x")
    url = c.upload_folder("user/repo", "/local/dir", create_if_missing=True, private=False)
    assert url == "https://huggingface.co/user/repo/commit/abc"
    inst.create_repo.assert_called_once_with(repo_id="user/repo", private=False, exist_ok=True)
    inst.upload_folder.assert_called_once()
