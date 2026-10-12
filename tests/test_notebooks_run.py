"""Notebook helpers: skip flags and PlantVillage staging names."""

import json
from pathlib import Path

import pytest

from notebooks.colab_sync import DEFAULT_BRANCH, assert_pv_staging_helper, parse_args
from notebooks.run import flag_on

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
    assert src.count('flag_on("RUN_STAGE1")') >= 2
    assert 'os.environ["RUN_STAGE1"]' in src
