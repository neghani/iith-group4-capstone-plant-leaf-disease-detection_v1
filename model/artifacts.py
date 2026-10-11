"""Upload the small artifacts of a Colab step to the Hub repo and/or Drive.

    python -m model.artifacts --step prep --run stage2 --hub neghani/plantvillage-mobilenetv2 \
        --drive /content/drive/MyDrive/plant-disease/runs model/outputs/data_manifest.json ...

Hub files go to runs/<run>/<step>/ on the `stage2-runs` branch, so `main` and the v1.0 tag are
untouched. Folders are walked; checkpoints, weights and files over --max_mb are skipped
(checkpoints go to Drive through train.py's output_dir instead).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BRANCH = "stage2-runs"
SKIP_PARTS = ("checkpoint-", "runs/", "__pycache__")
SKIP_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".npz", ".ckpt")


def collect(paths: list[str], max_mb: float = 5.0) -> list[tuple[Path, str]]:
    """(file, relative name) for each small file; folders keep their inner layout."""
    out = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            print(f"[artifacts] missing, skipped: {p}")
            continue
        files = [(p, p.name)] if p.is_file() else [(f, f"{p.name}/{f.relative_to(p).as_posix()}")
                                                     for f in sorted(p.rglob("*")) if f.is_file()]
        for f, rel in files:
            if any(s in rel for s in SKIP_PARTS) or f.suffix in SKIP_SUFFIXES:
                continue
            if f.stat().st_size > max_mb * 1024 * 1024:
                print(f"[artifacts] over {max_mb} MB, skipped: {rel}")
                continue
            out.append((f, rel))
    return out


def to_drive(files: list[tuple[Path, str]], root: Path, run: str, step: str) -> Path:
    dest = root / run / step
    for f, rel in files:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / rel)
    return dest


def to_hub(files: list[tuple[Path, str]], repo_id: str, run: str, step: str, token: str | None = None, api=None) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = api or HfApi(token=token)
    api.create_branch(repo_id, branch=BRANCH, exist_ok=True)
    ops = [CommitOperationAdd(f"runs/{run}/{step}/{rel}", str(f)) for f, rel in files]
    api.create_commit(repo_id=repo_id, operations=ops, revision=BRANCH,
                      commit_message=f"{run}: {step} artifacts ({len(files)} files)")
    return f"hf://{repo_id}@{BRANCH}/runs/{run}/{step}/"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--step", required=True, help="prep | diagnostics | train | calibrate | final_test | publish")
    ap.add_argument("--run", default="stage2")
    ap.add_argument("--hub", default=None, help="Hub model repo id")
    ap.add_argument("--drive", default=None, help="Drive folder (Colab: /content/drive/MyDrive/...)")
    ap.add_argument("--max_mb", type=float, default=5.0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)

    files = collect(a.paths, a.max_mb)
    print(f"[artifacts] {a.step}: {len(files)} files")
    for _, rel in files:
        print(f"  {rel}")
    if a.dry_run or not files:
        print("[artifacts] dry run, nothing uploaded" if a.dry_run else "[artifacts] nothing to upload")
        return
    if not a.hub and not a.drive:
        raise SystemExit("Pass --hub and/or --drive.")
    if a.drive:
        print(f"[artifacts] Drive: {to_drive(files, Path(a.drive), a.run, a.step)}")
    if a.hub:
        from common import get_hf_token

        print(f"[artifacts] Hub: {to_hub(files, a.hub, a.run, a.step, token=get_hf_token())}")


if __name__ == "__main__":
    main()
