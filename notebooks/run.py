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

DEFAULT_ROOT = "/content/iith-group4-capstone-plant-leaf-disease-detection_v1"

KNOB_DEFAULTS = {
    "RUN_STAGE1": "0",
    "RUN_STAGE2": "0",
    "STAGE1_EPOCHS": "5",
    "STAGE1_LR": "5e-4",
    "STAGE1_BATCH": "64",
    "PD_FRACTION": "0.5",
    "EPOCH_SAMPLES": "4000",
    "STAGE2_EPOCHS": "8",
    "STAGE2_LRS": "1e-4,5e-5",
    "STAGE2_BATCH": "64",
}

DEFAULT_HF_MODEL_ID = "neghani/plantvillage-resnet18"


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
    """sys.executable -u + args, cwd=repo, PYTHONPATH=repo. Raises on failure.

    `-u` and PYTHONUNBUFFERED keep Trainer / EDA prints streaming into the
    notebook cell while a long job is still running.
    """
    root = ensure_repo()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-u", *map(str, argv)]
    print("$", " ".join(cmd), flush=True)
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


def knob(name: str, default: str | None = None) -> str:
    """Read a notebook knob. Defaults match the Colab knobs cell."""
    if default is None:
        default = KNOB_DEFAULTS.get(name, "")
    return str(os.environ.get(name, default)).strip()


def apply_knobs(**values: object) -> None:
    """Write knobs into the environment for later cells."""
    for key, value in values.items():
        if isinstance(value, bool):
            os.environ[key] = "1" if value else "0"
        elif isinstance(value, (list, tuple)):
            os.environ[key] = ",".join(str(v) for v in value)
        else:
            os.environ[key] = str(value)


def stage2_lrs() -> list[str]:
    return [s.strip() for s in knob("STAGE2_LRS").split(",") if s.strip()]


def print_session_plan() -> None:
    """What this kernel will train, skip, and tag. Does not start a run."""
    s1 = flag_on("RUN_STAGE1")
    s2 = flag_on("RUN_STAGE2")
    hub = os.environ.get("HF_MODEL_ID") or DEFAULT_HF_MODEL_ID
    print("This session", flush=True)
    print("  Hub:", f"https://huggingface.co/{hub}", flush=True)
    print("  Stage 1 PlantVillage train (section 5):",
          "YES" if s1 else "SKIP (v1.0 already on the Hub)", flush=True)
    print("  Stage 1 tag v1.0 (section 8):", "YES" if s1 else "SKIP", flush=True)
    print("  Stage 2 mixed PlantVillage+PlantDoc train (section 7b):",
          "YES" if s2 else "SKIP (v1.1 already on the Hub)", flush=True)
    print("  Stage 2 tag v1.1 (section 7b):", "YES" if s2 else "SKIP", flush=True)
    tags = [t for t, on in (("v1.0", s1), ("v1.1", s2)) if on]
    print("  Hub tag this session would write:", ", ".join(tags) or "(none)", flush=True)
    print("  shared/release.json: unchanged by this notebook", flush=True)
    print(f"  PlantVillage: {knob('STAGE1_EPOCHS')} epochs, lr {knob('STAGE1_LR')}", flush=True)
    print(
        f"  PlantDoc mix: fraction {knob('PD_FRACTION')}, "
        f"{knob('EPOCH_SAMPLES')} draws/epoch, lrs {knob('STAGE2_LRS')}",
        flush=True,
    )


def print_publish_checklist() -> None:
    """Name the three Hub upload steps before a publish cell runs."""
    s1 = flag_on("RUN_STAGE1")
    s2 = flag_on("RUN_STAGE2")
    hub = os.environ.get("HF_MODEL_ID") or DEFAULT_HF_MODEL_ID
    tags = [t for t, on in (("v1.0", s1), ("v1.1", s2)) if on]
    print("Hugging Face publish checklist", flush=True)
    print(f"  Repo: https://huggingface.co/{hub}", flush=True)
    print("  1. Weights: section 5 Trainer --push_to_hub (v1.0) or section 7b local save then publish.py (v1.1)", flush=True)
    print("  2. reports/ + model card: evaluate.py --push (v1.0) or evaluate then publish.py (v1.1)", flush=True)
    print("  3. Release tag: publish.py --tag v1.0 (section 8) or --tag v1.1 (section 7b)", flush=True)
    print("  This session would tag:", ", ".join(tags) or "(none; RUN_STAGE1/RUN_STAGE2 are off)", flush=True)
    print("  shared/release.json is unchanged by this notebook. The live API pin stays until that file is edited.", flush=True)
