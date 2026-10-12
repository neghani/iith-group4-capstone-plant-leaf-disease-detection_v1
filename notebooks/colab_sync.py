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

    python -m model.train --smoke
    git add model && git commit && git push

Colab (GPU, full run). After every local push, re-run this helper -- it hard-resets
tracked files to a pinned commit (or origin/main) and leaves `model/outputs/` and
real-world images alone. It prints CODE_COMMIT (the full SHA) for run_info.json:

    %run notebooks/colab_sync.py --commit <sha>

Replace one file only (still from GitHub, still overwrite):

    %run notebooks/colab_sync.py --file model/train.py

Real-world photos are git-ignored. Put a zip on Drive *once*, then always replace:

    %run notebooks/colab_sync.py --real-world-from /content/drive/MyDrive/plant-disease/real_world.zip

The labelled real-world test set (labels.csv + images/, see model/data/real_world_set/README.md)
works the same way; a new or changed zip is what triggers a new Hub publish in the notebook:

    %run notebooks/colab_sync.py --real-world-set-from /content/drive/MyDrive/plant-disease/real_world_set.zip

PlantVillage training images: put one zip on Drive, then copy+unzip onto the Colab SSD
(do not train from the Drive mount). The notebook section 3 cell calls stage_zip_locally().
"""

from __future__ import annotations

import argparse
import csv
import json
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
KEEP_UNTRACKED = (
    "model/outputs",
    "model/data/real_world",
    "model/data/real_world_set/images",
    "model/data/plantdoc_src",
    "model/data/plantdoc_train",
    "model/data/plantdoc_test",
)
DEFAULT_DRIVE_ZIP = "/content/drive/MyDrive/plant-disease/real_world.zip"
DEFAULT_DRIVE_SET_ZIP = "/content/drive/MyDrive/plant-disease/real_world_set.zip"
DEFAULT_DRIVE_PV_ZIP = "/content/drive/MyDrive/plant-disease/plantvillage.zip"
DEFAULT_LOCAL_PV_ZIP = "/content/plantvillage.zip"
DEFAULT_LOCAL_PV_DIR = "/content/plantvillage"


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


def reset_to_origin(root: Path, branch: str, commit: str = "") -> str:
    """Replace tracked files with `commit` (a SHA) or origin/<branch>. Keep outputs and images."""
    run(["git", "fetch", "origin", branch], cwd=root)
    target = commit.strip() if commit else f"origin/{branch}"
    if commit and commit not in (branch, f"origin/{branch}"):
        # SHA or other ref: fetch it so reset can see it even if it is not the branch tip.
        fetched = subprocess.run(["git", "fetch", "origin", commit], cwd=root, text=True, capture_output=True)
        if fetched.returncode != 0:
            print(f"[warn] git fetch origin {commit} failed; trying reset anyway.\n{fetched.stderr}")
    run(["git", "reset", "--hard", target], cwd=root)
    sha = run(["git", "rev-parse", "HEAD"], cwd=root)
    short = run(["git", "rev-parse", "--short", "HEAD"], cwd=root)
    os.environ["CODE_COMMIT"] = sha
    print(f"Replaced tracked files with {target} @ {short}")
    print(f"CODE_COMMIT {sha}")
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


def _find_set_root(path: Path, depth: int = 3) -> Path | None:
    if (path / "labels.csv").is_file():
        return path
    if depth == 0:
        return None
    for kid in _class_folders(path):
        found = _find_set_root(kid, depth - 1)
        if found:
            return found
    return None


def replace_real_world_set(src: Path, dest: Path) -> None:
    """Replace the labelled set (labels.csv, images/, sources.csv, ...) from a zip or folder."""
    if not src.exists():
        raise SystemExit(f"Real-world set source not found: {src}")
    with tempfile.TemporaryDirectory() as tmp:
        extracted = Path(tmp) / "src"
        if src.is_file() and src.suffix.lower() == ".zip":
            with zipfile.ZipFile(src) as zf:
                zf.extractall(extracted)
        elif src.is_dir():
            shutil.copytree(src, extracted)
        else:
            raise SystemExit(f"Need a .zip or a folder, got {src}")
        payload = _find_set_root(extracted)
        if payload is None:
            raise SystemExit(f"No labels.csv found in {src}")
        with open(payload / "labels.csv", newline="") as f:
            files = [row["file"] for row in csv.DictReader(f)]
        missing = [p for p in files if not (payload / p).is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} images listed in labels.csv are missing, e.g. {missing[:3]}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(payload, dest)
    print(f"Replaced real_world_set from {src}: {len(files)} images")


def _zip_stamp(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def stage_zip_locally(src: Path, dest: Path, local_zip: Path | None = None) -> Path:
    """Copy a Drive zip onto local disk, then unzip that copy. Skip when size+mtime match.

    Reading tens of thousands of small files through the Drive mount is slow. The zip is
    copied to `local_zip` (default: dest.parent / dest.name.zip) and extracted from there.
    """
    src = Path(src)
    dest = Path(dest)
    if not src.exists():
        raise FileNotFoundError(f"Dataset zip not found: {src}")
    if not src.is_file() or src.suffix.lower() != ".zip":
        raise SystemExit(f"Need a .zip file, got {src}")
    local_zip = Path(local_zip) if local_zip is not None else dest.parent / f"{dest.name}.zip"
    stamp_path = dest / ".zip_stamp.json"
    src_stamp = _zip_stamp(src)

    already = False
    if dest.exists() and local_zip.exists() and stamp_path.exists():
        try:
            already = json.loads(stamp_path.read_text()) == src_stamp
        except (OSError, json.JSONDecodeError):
            already = False
    if already:
        print(f"Local copy of {src} already staged at {dest}; skipping copy and unzip")
        return dest

    if src.resolve() != local_zip.resolve():
        local_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_zip)
        print(f"Copied {src} -> {local_zip}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(dest)
    stamp_path.write_text(json.dumps(src_stamp))
    print(f"Unzipped {local_zip} -> {dest}")
    return dest


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
        "--commit",
        default=os.environ.get("REPO_COMMIT", ""),
        help="Checkout this SHA (or other git ref). Default: origin/<branch>. Printed as CODE_COMMIT.",
    )
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
    p.add_argument(
        "--real-world-set-from",
        default="",
        help=f"Zip or folder with labels.csv + images/. Default on Colab if present: {DEFAULT_DRIVE_SET_ZIP}",
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

    if not colab and not args.force and not args.file and not args.real_world_from and not args.real_world_set_from:
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
    os.environ["PYTHONPATH"] = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if args.file:
        replace_files(root, args.branch, args.file)
        sha = run(["git", "rev-parse", "HEAD"], cwd=root)
        os.environ["CODE_COMMIT"] = sha
        print(f"CODE_COMMIT {sha} (working tree not reset; --file only)")
    elif colab or args.force:
        reset_to_origin(root, args.branch, args.commit)
    else:
        sha = run(["git", "rev-parse", "HEAD"], cwd=root)
        os.environ["CODE_COMMIT"] = sha
        print(f"CODE_COMMIT {sha}")

    if colab:
        maybe_mount_drive()
    src = args.real_world_from
    if not src and Path(DEFAULT_DRIVE_ZIP).exists():
        src = DEFAULT_DRIVE_ZIP
    if src:
        replace_real_world(Path(src).expanduser(), root / "model" / "data" / "real_world")
    set_src = args.real_world_set_from
    if not set_src and Path(DEFAULT_DRIVE_SET_ZIP).exists():
        set_src = DEFAULT_DRIVE_SET_ZIP
    if set_src:
        replace_real_world_set(Path(set_src).expanduser(), root / "model" / "data" / "real_world_set")

    if (colab or args.force) and not args.skip_install and not args.file:
        install_requirements(root)

    print("REPO_ROOT", root)
    print("cwd", Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
