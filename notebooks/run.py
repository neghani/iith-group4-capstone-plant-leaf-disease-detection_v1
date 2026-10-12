"""Colab helpers: always run from the repo, always fail the cell on a nonzero exit.

    from notebooks.run import run, run_mod, ensure_repo

`!python ...` does not raise on failure and often uses cwd=/content, which is why
prep/eval look like they ran and training gets re-executed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = "/content/iiith-group4-capstone-plat-leaf-decease-detection"


def ensure_repo() -> Path:
    root = Path(os.environ.get("REPO_ROOT") or DEFAULT_ROOT)
    if not (root / "model" / "train.py").exists():
        here = Path.cwd()
        if (here / "model" / "train.py").exists():
            root = here
        elif (here.parent / "model" / "train.py").exists():
            root = here.parent
    if not (root / "model" / "train.py").exists():
        raise SystemExit(
            f"Repo not found at {root}. Re-run section 1 with REPO_BRANCH=main."
        )
    os.chdir(root)
    os.environ["REPO_ROOT"] = str(root)
    os.environ["PYTHONPATH"] = str(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def run(*argv: str) -> None:
    """sys.executable + args, cwd=repo, PYTHONPATH=repo. Raises on failure."""
    root = ensure_repo()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    cmd = [sys.executable, *map(str, argv)]
    print("$", " ".join(cmd))
    try:
        subprocess.check_call(cmd, cwd=root, env=env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Command exited {exc.returncode}: {' '.join(cmd)}\n"
            "The script's own error is in the cell output above this message "
            "(missing weights, splits.json, DATA_DIR, etc.)."
        ) from None


def run_mod(module: str, *args: str) -> None:
    run("-m", module, *args)


def require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"Missing env {missing}. Re-run the cells that set them (section 2/3).")


def flag_on(name: str) -> bool:
    """True when the notebook or env set this to 1/true/yes."""
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes")
