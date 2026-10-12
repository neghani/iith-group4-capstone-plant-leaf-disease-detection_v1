"""Notebook helpers: skip flags and PlantVillage staging names."""

import json
from pathlib import Path

import pytest

from notebooks.colab_sync import DEFAULT_BRANCH, assert_pv_staging_helper, parse_args
from notebooks.run import apply_knobs, flag_on, knob, print_publish_checklist, print_session_plan, run, stage2_lrs

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "phase1_plantvillage_mobilenetv2.ipynb"


def test_flag_on_reads_env(monkeypatch):
    monkeypatch.delenv("RUN_STAGE2", raising=False)
    assert flag_on("RUN_STAGE2") is False
    monkeypatch.setenv("RUN_STAGE2", "1")
    assert flag_on("RUN_STAGE2") is True
    monkeypatch.setenv("RUN_STAGE2", "true")
    assert flag_on("RUN_STAGE2") is True
    monkeypatch.setenv("RUN_STAGE2", "no")
    assert flag_on("RUN_STAGE2") is False


def test_notebooks_package_exposes_pv_staging():
    assert DEFAULT_BRANCH == "main"
    assert_pv_staging_helper()


def test_stale_helper_exits():
    class Stale:
        pass

    with pytest.raises(SystemExit, match="Re-run section 1"):
        assert_pv_staging_helper(Stale())


def _notebook_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(cell.get("source") or []) for cell in nb["cells"]]


def _notebook_source() -> str:
    return "\n".join(_notebook_cells())


def test_notebook_section1_defaults_to_main_not_env():
    src = _notebook_source()
    assert 'REPO_BRANCH = "main"' in src
    assert "os.environ.pop(\"REPO_BRANCH\"" in src
    assert "ALLOW_BRANCH_OVERRIDE" in src
    assert 'os.environ.get("REPO_BRANCH", "main")' not in src.split("if _OVERRIDE:")[0]
    assert "Phase4" not in src or "leftover Colab REPO_BRANCH" in src


def test_notebook_zip_cell_does_not_hard_reset():
    zip_cell = next(c for c in _notebook_cells() if "real-world-set-from" in c and "check_call" in c)
    assert "--no-reset" in zip_cell
    assert '"--branch", "main"' in zip_cell


def test_colab_sync_no_reset_flag():
    args = parse_args(["--no-reset", "--skip-install", "--real-world-set-from", "/tmp/x.zip"])
    assert args.no_reset is True
    assert args.skip_install is True


def test_notebook_stage2_and_publish_cells_honor_skip_flags():
    src = _notebook_source()
    assert src.count('flag_on("RUN_STAGE2")') >= 4
    assert src.count('flag_on("RUN_STAGE1")') >= 4  # section 5 train, section 6 eval+reports, section 8
    assert "apply_knobs" in src
    assert "print_session_plan" in src
    assert "print_publish_checklist" in src
    eval_cell = next(c for c in _notebook_cells() if "model/evaluate.py" in c and "--push" in c)
    assert 'flag_on("RUN_STAGE1")' in eval_cell


def test_notebook_knobs_cell_owns_stage_flags_and_shows_work():
    cells = _notebook_cells()
    knobs = next(c for c in cells if "apply_knobs" in c and "PD_FRACTION" in c)
    assert "RUN_STAGE1 = False" in knobs
    assert "RUN_STAGE2 = False" in knobs
    assert "STAGE2_LRS" in knobs
    train = next(c for c in cells if "--push_to_hub" in c and "google/mobilenet_v2")
    assert "RUN_STAGE1 = False" not in train
    assert "PlantDoc is not in this run" in train
    assert "knob(\"STAGE1_EPOCHS\")" in train
    mix = next(c for c in cells if "class_weight_summary" in c)
    assert "uniform PlantVillage" in mix
    loop = next(c for c in cells if "--extra_data" in c and "plantdoc" in c and "stage2_lrs()" in c)
    assert "for lr_s, out in zip(stage2_lrs()" in loop
    eda = next(c for c in cells if "plantvillage_eda.json" in c and "per_class" in c)
    assert "PlantVillage class counts" in eda
    assert "shared/release.json" in "\n".join(cells)


def test_run_wraps_nonzero_exit_as_systemexit():
    with pytest.raises(SystemExit, match="exited 1"):
        run("-c", "raise SystemExit(1)")


def test_run_launches_unbuffered_python(monkeypatch):
    import subprocess
    import sys

    seen = {}

    def fake_check_call(cmd, cwd=None, env=None):
        seen["cmd"] = cmd
        seen["env"] = env

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    run("-c", "print(1)")
    assert seen["cmd"][0] == sys.executable
    assert seen["cmd"][1] == "-u"
    assert seen["env"]["PYTHONUNBUFFERED"] == "1"


def test_apply_knobs_and_publish_checklist(monkeypatch, capsys):
    for key in (
        "RUN_STAGE1", "RUN_STAGE2", "PD_FRACTION", "STAGE2_LRS",
        "STAGE1_EPOCHS", "STAGE1_LR", "EPOCH_SAMPLES", "HF_MODEL_ID",
    ):
        monkeypatch.setenv(key, "x")
    apply_knobs(RUN_STAGE1=False, RUN_STAGE2=True, PD_FRACTION=0.5, STAGE2_LRS=["1e-4", "5e-5"])
    monkeypatch.setenv("HF_MODEL_ID", "neghani/plantvillage-mobilenetv2")
    assert flag_on("RUN_STAGE1") is False
    assert flag_on("RUN_STAGE2") is True
    assert knob("PD_FRACTION") == "0.5"
    assert stage2_lrs() == ["1e-4", "5e-5"]
    print_session_plan()
    print_publish_checklist()
    out = capsys.readouterr().out
    assert "v1.1" in out
    assert "shared/release.json" in out
    assert "https://huggingface.co/" in out


def test_evaluate_rejects_empty_local_dir(tmp_path):
    empty = tmp_path / "stage1"
    empty.mkdir()
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "model/evaluate.py", "--model", str(empty)],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not a trained model" in (result.stderr + result.stdout)
