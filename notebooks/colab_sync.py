#!/usr/bin/env python3
"""Always replace repo files in Colab from GitHub. Never use files.upload() for code.

Why this exists
---------------
Colab's `files.upload()` writes a *new* copy next to the clone. Re-uploading the same
name does not reliably overwrite, and `git pull --ff-only` then fails because the
working tree is dirty. The result is you re-test stale `model/*.py`.

Do this instead
---------------
Local (CPU, iterate on code):

    python model/train.py --data_dir "$DATA_DIR" --crop_filter Tomato \\
      --max_samples_per_class 40 --output_dir model/outputs/smoke \\
      --do_train --do_eval --num_train_epochs 1 --save_strategy no --report_to none
    git add model && git commit && git push

Colab (GPU, full run). After every local push, re-run this helper -- it hard-resets
tracked files to origin/main and leaves `model/outputs/` and real-world images alone:

    %run notebooks/colab_sync.py

Replace one file only (still from GitHub, still overwrite):

    %run notebooks/colab_sync.py --file model/train.py

Real-world photos are git-ignored. Put a zip on Drive *once*, then always replace:

    %run notebooks/colab_sync.py --real-world-from /content/drive/MyDrive/plant-disease/real_world.zip
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

DEFAULT_REPO = (
    "https://github.com/neghani/iiith-group4-capstone-plat-leaf-decease-detection.git"
)
DEFAULT_BRANCH = "main"
REPO_NAME = "iiith-group4-capstone-plat-leaf-decease-detection"
KEEP_UNTRACKED = ("model/outputs", "model/data/real_world")
DEFAULT_DRIVE_ZIP = "/content/drive/MyDrive/plant-disease/real_world.zip"


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SystemExit(f"$ {' '.join(cmd)}\n{detail}")
    return (result.stdout or "").strip()


def find_repo_root() -> Path | None:
    candidates = [Path.cwd(), Path.cwd() / REPO_NAME]
    if Path("/content").exists():
        candidates.append(Path("/content") / REPO_NAME)
    for path in candidates:
        if (path / ".git").exists() and (path / "model" / "train.py").exists():
            return path.resolve()
    return None


def clone_or_find(repo: str, dest_parent: Path) -> Path:
    existing = find_repo_root()
    if existing:
        return existing
    dest = dest_parent / REPO_NAME
    print(f"Cloning {repo} -> {dest}")
    run(["git", "clone", repo, str(dest)])
    return dest.resolve()


def reset_to_origin(root: Path, branch: str) -> str:
    """Always replace tracked files with origin/<branch>. Keep outputs and images."""
    run(["git", "fetch", "origin", branch], cwd=root)
    # reset --hard even if Colab has leftover uploads or local edits
    run(["git", "reset", "--hard", f"origin/{branch}"], cwd=root)
    sha = run(["git", "rev-parse", "--short", "HEAD"], cwd=root)
    print(f"Replaced tracked files with origin/{branch} @ {sha}")
    print(f"Left untracked: {', '.join(KEEP_UNTRACKED)}")
    return sha


def replace_files(root: Path, branch: str, rel_paths: list[str]) -> None:
    run(["git", "fetch", "origin", branch], cwd=root)
    for rel in rel_paths:
        dest = (root / rel).resolve()
        if not str(dest).startswith(str(root.resolve())):
            raise SystemExit(f"Refusing path outside repo: {rel}")
        run(["git", "checkout", f"origin/{branch}", "--", rel], cwd=root)
        print(f"Replaced {rel} from origin/{branch}")


def _class_folders(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))


def replace_real_world(src: Path, dest: Path) -> None:
    """Wipe class folders under dest, then copy from a zip or folder. Always replace."""
    dest.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise SystemExit(f"Real-world source not found: {src}")

    with tempfile.TemporaryDirectory() as tmp:
        extracted = Path(tmp) / "src"
        extracted.mkdir()
        if src.is_file() and src.suffix.lower() == ".zip":
            with zipfile.ZipFile(src) as zf:
                zf.extractall(extracted)
        elif src.is_dir():
            shutil.copytree(src, extracted, dirs_exist_ok=True)
        else:
            raise SystemExit(f"Need a .zip or a folder, got {src}")

        # Accept zip layouts: <Class>/img, real_world/<Class>/img, or nested once more.
        payload = extracted
        for _ in range(2):
            kids = _class_folders(payload)
            if len(kids) == 1 and kids[0].name in {"real_world", "data"}:
                payload = kids[0]
                continue
            break

        removed = 0
        for folder in _class_folders(dest):
            shutil.rmtree(folder)
            removed += 1
        copied = 0
        for folder in _class_folders(payload):
            shutil.copytree(folder, dest / folder.name, dirs_exist_ok=False)
            copied += 1
        for extra in ("sources.csv",):
            candidate = payload / extra
            if candidate.exists():
                shutil.copy2(candidate, dest / extra)

    print(f"Replaced real_world: removed {removed} folders, copied {copied} from {src}")
    print("Class folders:", [p.name for p in _class_folders(dest)] or "(none)")


def maybe_mount_drive() -> None:
    if not in_colab():
        return
    if Path("/content/drive/MyDrive").is_dir():
        return
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def install_requirements(root: Path) -> None:
    req = root / "model" / "requirements.txt"
    if not req.exists():
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)]
    )
    print("Installed model/requirements.txt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Always replace Colab files from GitHub. Do not files.upload() code."
    )
    p.add_argument("--repo", default=os.environ.get("REPO_URL", DEFAULT_REPO))
    p.add_argument("--branch", default=os.environ.get("REPO_BRANCH", DEFAULT_BRANCH))
    p.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="PATH",
        help="Replace only this repo-relative file from origin (repeatable).",
    )
    p.add_argument(
        "--real-world-from",
        default="",
        help=f"Zip or folder of class images. Default on Colab if present: {DEFAULT_DRIVE_ZIP}",
    )
    p.add_argument("--skip-install", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow a hard reset outside Colab (destructive to local edits).",
    )
    # `%run` / Jupyter may append -f kernel.json; ignore those.
    args, unknown = p.parse_known_args(argv)
    extra = [a for a in unknown if a not in {"-f"} and "kernel" not in a]
    if extra:
        print("Ignoring extra args:", extra)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    colab = in_colab()

    if not colab and not args.force and not args.file and not args.real_world_from:
        print("Local checkout detected. Edit files here, smoke-test, then git push.")
        print("In Colab, re-run this script to always replace tracked files from GitHub.")
        print("Refusing to reset your working tree. Pass --force only if you mean it.")
        return 0

    dest_parent = Path("/content") if Path("/content").exists() else Path.cwd()
    if colab or args.force:
        root = clone_or_find(args.repo, dest_parent)
    else:
        root = find_repo_root()
        if root is None:
            raise SystemExit("Run this from the repo (or pass --force to clone).")

    os.chdir(root)
    # Visible to `%run` callers so later notebook cells share the same cwd.
    os.environ["REPO_ROOT"] = str(root)

    if args.file:
        replace_files(root, args.branch, args.file)
    elif colab or args.force:
        reset_to_origin(root, args.branch)

    if colab:
        maybe_mount_drive()
    src = args.real_world_from
    if not src and Path(DEFAULT_DRIVE_ZIP).exists():
        src = DEFAULT_DRIVE_ZIP
    if src:
        replace_real_world(Path(src).expanduser(), root / "model" / "data" / "real_world")

    if (colab or args.force) and not args.skip_install and not args.file:
        install_requirements(root)

    print("REPO_ROOT", root)
    print("cwd", Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
