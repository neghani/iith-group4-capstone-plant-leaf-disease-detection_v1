"""Final-test guard: one small JSON per model revision, kept off the Colab disk.

    hub:<repo_id>   final_test_guard/<revision>.json on the repo's `stage2-runs` branch
    drive:<dir>     <dir>/<revision>.json (e.g. /content/drive/MyDrive/plant-disease/final_test_guard)

A record means the final test was already run for that revision; final_test.py refuses a second
run unless --force. Smoke mode never constructs a guard.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

GUARD_BRANCH = "stage2-runs"
GUARD_FOLDER = "final_test_guard"


def _file_name(revision: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", revision or ""):
        raise ValueError(f"guard revision must be a plain tag like v1.1, got {revision!r}")
    return f"{revision}.json"


class DirGuard:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def describe(self, revision: str) -> str:
        return str(self.root / _file_name(revision))

    def read(self, revision: str) -> dict | None:
        path = self.root / _file_name(revision)
        return json.loads(path.read_text()) if path.exists() else None

    def write(self, revision: str, record: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / _file_name(revision)).write_text(json.dumps(record, indent=2) + "\n")


class HubGuard:
    def __init__(self, repo_id: str, token: str | None = None, branch: str = GUARD_BRANCH, api=None):
        self.repo_id = repo_id
        self.branch = branch
        self.token = token
        self._api = api

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
        return self._api

    def _path(self, revision: str) -> str:
        return f"{GUARD_FOLDER}/{_file_name(revision)}"

    def describe(self, revision: str) -> str:
        return f"hf://{self.repo_id}@{self.branch}/{self._path(revision)}"

    def read(self, revision: str) -> dict | None:
        branches = {b.name for b in self.api.list_repo_refs(self.repo_id).branches}
        if self.branch not in branches:
            return None
        if not self.api.file_exists(self.repo_id, self._path(revision), revision=self.branch):
            return None
        local = self.api.hf_hub_download(self.repo_id, self._path(revision), revision=self.branch,
                                         force_download=True)
        return json.loads(Path(local).read_text())

    def write(self, revision: str, record: dict) -> None:
        self.api.create_branch(self.repo_id, branch=self.branch, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(record, indent=2) + "\n")
            tmp = f.name
        self.api.upload_file(path_or_fileobj=tmp, path_in_repo=self._path(revision), repo_id=self.repo_id,
                             revision=self.branch, commit_message=f"Final-test guard for {revision}")


def make_guard(spec: str, token: str | None = None):
    """'hub:<repo>' or 'drive:<dir>' (any 'dir:<path>' works too)."""
    kind, _, target = spec.partition(":")
    if not target:
        raise ValueError(f"guard spec must be hub:<repo> or drive:<dir>, got {spec!r}")
    if kind == "hub":
        return HubGuard(target, token=token)
    if kind in ("drive", "dir"):
        return DirGuard(target)
    raise ValueError(f"unknown guard kind {kind!r} in {spec!r}")


def check(guard, revision: str, model_key: str, force: bool) -> None:
    """Refuse a second final test for `revision`. Unreadable guard is an error, never a pass."""
    try:
        prev = guard.read(revision)
    except Exception as exc:
        raise SystemExit(f"Cannot read the final-test guard {guard.describe(revision)} ({exc}). "
                         "Refusing to run: the guard must be readable to know this is the first run.") from exc
    if prev and not force:
        extra = "" if prev.get("model_key") == model_key else (
            f" Those were different weights ({prev.get('model_key')}); a new candidate needs a new revision name.")
        raise SystemExit(f"The final test for {revision} already ran on {prev.get('date')} "
                         f"({guard.describe(revision)}).{extra} Use --force only to regenerate the same run.")
