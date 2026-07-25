"""HFClient thin wrapper — token handling + method signatures."""
import io
from pathlib import Path
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


CLOSED_METHOD_CASES = [
    ("whoami", (), {}),
    ("download_snapshot", ("user/repo", "/tmp/local-dir"), {}),
    ("download_file", ("user/repo", "README.md"), {}),
    ("upload_folder", ("user/repo", "/tmp/local-dir"), {}),
    ("upload_file", ("user/repo", "/tmp/file.bin", "weights/file.bin"), {}),
    ("update_file", ("user/repo", "README.md", "content", "commit msg"), {}),
]


@pytest.mark.parametrize(
    "method_name,args,kwargs",
    CLOSED_METHOD_CASES,
    ids=[case[0] for case in CLOSED_METHOD_CASES],
)
@patch("huggingface_hub.hf_hub_download")
@patch("huggingface_hub.snapshot_download")
@patch("b2cq.hf_client.HfApi")
def test_every_public_method_raises_after_close(HfApi, snapshot_download, hf_hub_download, method_name, args, kwargs):
    c = HFClient(token="hf_supersecret")
    c.close()
    method = getattr(c, method_name)
    with pytest.raises(RuntimeError, match="closed"):
        method(*args, **kwargs)
    HfApi.assert_not_called()
    snapshot_download.assert_not_called()
    hf_hub_download.assert_not_called()


@patch("b2cq.hf_client.HfApi")
def test_whoami_forwards_token(HfApi):
    inst = MagicMock()
    inst.whoami.return_value = {"name": "eddie"}
    HfApi.return_value = inst
    c = HFClient(token="hf_x")
    result = c.whoami()
    HfApi.assert_called_once_with(token="hf_x")
    assert result == {"name": "eddie"}


@patch("huggingface_hub.hf_hub_download")
def test_download_file_forwards_args(hf_hub_download):
    hf_hub_download.return_value = "/cache/user/repo/README.md"
    c = HFClient(token="hf_x")
    result = c.download_file("user/repo", "README.md")
    hf_hub_download.assert_called_once_with(repo_id="user/repo", filename="README.md", token="hf_x")
    assert result == Path("/cache/user/repo/README.md")


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


@patch("b2cq.hf_client.HfApi")
def test_upload_file_forwards_args(HfApi):
    inst = MagicMock()
    inst.upload_file.return_value = "https://huggingface.co/user/repo/blob/main/weights/file.bin"
    HfApi.return_value = inst
    c = HFClient(token="hf_x")
    url = c.upload_file(
        "user/repo", "/local/file.bin", "weights/file.bin",
        create_if_missing=True, private=False, commit_message="custom upload msg",
    )
    assert url == "https://huggingface.co/user/repo/blob/main/weights/file.bin"
    inst.create_repo.assert_called_once_with(repo_id="user/repo", private=False, exist_ok=True)
    inst.upload_file.assert_called_once()
    _, kwargs = inst.upload_file.call_args
    assert kwargs["repo_id"] == "user/repo"
    assert kwargs["path_in_repo"] == "weights/file.bin"
    assert kwargs["commit_message"] == "custom upload msg"
    assert kwargs["path_or_fileobj"] == "/local/file.bin"


@patch("b2cq.hf_client.HfApi")
def test_update_file_forwards_args_and_encodes_utf8(HfApi):
    inst = MagicMock()
    inst.upload_file.return_value = "https://huggingface.co/user/repo/blob/main/README.md"
    HfApi.return_value = inst
    c = HFClient(token="hf_x")
    url = c.update_file("user/repo", "README.md", "hello café", "update readme")
    assert url == "https://huggingface.co/user/repo/blob/main/README.md"
    inst.upload_file.assert_called_once()
    _, kwargs = inst.upload_file.call_args
    assert kwargs["repo_id"] == "user/repo"
    assert kwargs["path_in_repo"] == "README.md"
    assert kwargs["commit_message"] == "update readme"
    fileobj = kwargs["path_or_fileobj"]
    assert isinstance(fileobj, io.BytesIO)
    assert fileobj.read() == "hello café".encode("utf-8")


@patch("b2cq.hf_client.HfApi")
def test_upload_folder_private_flag_threads_to_create_repo(HfApi):
    inst = MagicMock()
    inst.upload_folder.return_value = "https://huggingface.co/user/repo/commit/abc"
    HfApi.return_value = inst
    c = HFClient(token="hf_x")
    c.upload_folder("user/repo", "/local/dir", create_if_missing=True, private=True)
    inst.create_repo.assert_called_once_with(repo_id="user/repo", private=True, exist_ok=True)
