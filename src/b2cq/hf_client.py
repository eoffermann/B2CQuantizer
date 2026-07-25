"""Thin wrapper around huggingface_hub with careful token handling.

Token is held only in an instance attribute — cleared on close(). Never
logged, never included in __repr__, never written to disk. Instance
methods forward operations to a fresh HfApi with the token attached; no
long-lived HfApi is stored (each call constructs one) so revocation via
close() is immediate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


class HFClient:
    def __init__(self, token: str):
        if not token or not token.startswith(("hf_", "hf-")):
            raise ValueError("HF token must start with 'hf_' or 'hf-'")
        self._token: str | None = token

    def __repr__(self) -> str:
        state = "closed" if self._token is None else "open"
        return f"<HFClient state={state}>"

    def close(self) -> None:
        self._token = None

    def _api(self) -> HfApi:
        if self._token is None:
            raise RuntimeError("HFClient is closed")
        return HfApi(token=self._token)

    def whoami(self) -> dict[str, Any]:
        return self._api().whoami()

    def download_snapshot(self, repo_id: str, local_dir: str | Path) -> Path:
        if self._token is None:
            raise RuntimeError("HFClient is closed")
        from huggingface_hub import snapshot_download
        p = snapshot_download(repo_id=repo_id, local_dir=str(local_dir), token=self._token)
        return Path(p)

    def upload_folder(self, repo_id: str, folder_path: str | Path, *,
                      create_if_missing: bool = True, private: bool = False,
                      commit_message: str | None = None) -> str:
        api = self._api()
        if create_if_missing:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        return api.upload_folder(
            repo_id=repo_id,
            folder_path=str(folder_path),
            commit_message=commit_message or "B2CQuantizer: upload quantized weights",
        )

    def upload_file(self, repo_id: str, file_path: str | Path, path_in_repo: str, *,
                    create_if_missing: bool = True, private: bool = False,
                    commit_message: str | None = None) -> str:
        api = self._api()
        if create_if_missing:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        return api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            commit_message=commit_message or f"B2CQuantizer: upload {path_in_repo}",
        )

    def update_file(self, repo_id: str, path_in_repo: str, content: str,
                    commit_message: str) -> str:
        """Upload a text file (README, etc.) directly from a string."""
        import io
        api = self._api()
        return api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=io.BytesIO(content.encode("utf-8")),
            path_in_repo=path_in_repo,
            commit_message=commit_message,
        )
