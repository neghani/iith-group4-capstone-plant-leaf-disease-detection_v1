"""Notebook helpers: skip flags and PlantVillage staging names."""

import ast
import json
from pathlib import Path

import pytest

from notebooks.colab_sync import DEFAULT_BRANCH, assert_pv_staging_helper, parse_args
from notebooks.run import (
    apply_knobs,
    flag_on,
    knob,
    print_publish_checklist,
    print_session_plan,
    require_data_dir,
    run,
    stage2_lrs,
)

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "phase1_plantvillage_resnet18.ipynb"
MOBILENET_NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "phase1_plantvillage_mobilenetv2.ipynb"


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


def test_mobilenet_notebook_still_present():
    assert MOBILENET_NOTEBOOK.is_file()
    src = MOBILENET_NOTEBOOK.read_text()
    assert "google/mobilenet_v2" in src


def test_notebook_section1_defaults_to_main_not_env():
    src = _notebook_source()
    assert 'REPO_BRANCH = "main"' in src
    assert "os.environ.pop(\"REPO_BRANCH\"" in src
    assert "ALLOW_BRANCH_OVERRIDE" in src
    assert 'os.environ.get("REPO_BRANCH", "main")' not in src.split("if _OVERRIDE:")[0]
    assert "Phase4" not in src or "leftover Colab REPO_BRANCH" in src


def test_notebook_zip_cell_does_not_hard_reset():
    zip_cell = next(c for c in _notebook_cells() if "real-world-set-from" in c and "colab_sync.py" in c)
    assert "--no-reset" in zip_cell
    assert '"--branch", "main"' in zip_cell
    assert "reset --hard" not in zip_cell
    assert "check_call" not in zip_cell


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
    train = next(c for c in cells if "--push_to_hub" in c and "microsoft/resnet-18")
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


def test_require_data_dir_missing(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    with pytest.raises(SystemExit, match="not set"):
        require_data_dir()


def test_require_data_dir_rejects_drive_path(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/content/drive/MyDrive/plant-disease/color")
    with pytest.raises(SystemExit, match="Google Drive"):
        require_data_dir()


def test_require_data_dir_rejects_missing_kaggle_path(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/kaggle/input/plantvillage-dataset/plantvillage dataset/color")
    with pytest.raises(SystemExit, match="does not exist"):
        require_data_dir()


def test_require_data_dir_accepts_class_folder(tmp_path, monkeypatch):
    root = tmp_path / "color" / "Tomato___healthy"
    root.mkdir(parents=True)
    (root / "a.jpg").write_bytes(b"not-a-real-jpeg")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "color"))
    assert require_data_dir() == tmp_path / "color"


def test_data_dir_cells_call_preflight():
    cells = _notebook_cells()
    for needle in ("model/eda.py", "tomato_smoke", "microsoft/resnet-18", "--extra_data"):
        cell = next(c for c in cells if needle in c and "DATA_DIR" in c)
        assert "require_data_dir()" in cell


def test_run_wraps_nonzero_exit_as_systemexit():
    with pytest.raises(SystemExit, match="exited 1"):
        run("-c", "raise SystemExit(1)")


def test_run_launches_unbuffered_python(monkeypatch):
    import subprocess
    import sys

    seen = {}

    class FakeProc:
        stdout = iter(())

        def __init__(self, cmd, cwd=None, env=None, **kwargs):
            seen["cmd"] = cmd
            seen["env"] = env
            seen["stdout"] = kwargs.get("stdout")
            seen["stderr"] = kwargs.get("stderr")

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    run("-c", "print(1)")
    assert seen["cmd"][0] == sys.executable
    assert seen["cmd"][1] == "-u"
    assert seen["env"]["PYTHONUNBUFFERED"] == "1"
    assert seen["stdout"] == subprocess.PIPE
    assert seen["stderr"] == subprocess.STDOUT


def _parse_code_cells() -> list[tuple[int, str, ast.AST]]:
    nb = json.loads(NOTEBOOK.read_text())
    parsed = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source") or [])
        py = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%")))
        if not py.strip():
            continue
        parsed.append((i, src, ast.parse(py)))
    return parsed


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _test_calls_gate(test: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("hub_writes_on", "flag_on")
        for n in ast.walk(test)
    )


def _inside_gate(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, ast.If) and _test_calls_gate(cur.test):
            return True
    return False


def _environ_keys(tree: ast.AST, ctx: type) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ctx):
            continue
        value = node.value
        if not (isinstance(value, ast.Attribute) and value.attr == "environ" and isinstance(value.value, ast.Name) and value.value.id == "os"):
            continue
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            keys.add(sl.value)
    return keys


def _require_env_keys(tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "require_env"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                keys.add(arg.value)
    return keys


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def test_resnet_notebook_audit():
    """Fail if a ResNet notebook cell can write the Hub, read env, or call helpers unsafely."""
    parsed = _parse_code_cells()
    defined: dict[str, list[int]] = {}
    for idx, _src, tree in parsed:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined.setdefault(node.name, []).append(idx)
    findings = []
    for idx, src, tree in parsed:
        parents = _parents(tree)
        imported = _imported_names(tree)
        for node in ast.walk(tree):
            hub_call = isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "upload_artifacts"
            hub_flag = isinstance(node, ast.Constant) and node.value in ("--push", "--hub")
            if (hub_call or hub_flag) and not _inside_gate(node, parents):
                findings.append(f"cell {idx}: Hub write is not inside hub_writes_on() or flag_on()")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name in defined and idx not in defined[name] and name not in imported:
                    findings.append(f"cell {idx}: calls {name}(), defined only in cell(s) {defined[name]}")
        missing_env = _environ_keys(tree, ast.Load) - _require_env_keys(tree) - _environ_keys(tree, ast.Store)
        if missing_env:
            findings.append(f"cell {idx}: os.environ read without require_env: {sorted(missing_env)}")
        if "subprocess" in src and "urllib.request.urlretrieve" not in src:
            findings.append(f"cell {idx}: calls subprocess directly; use run() or run_mod()")
    assert not findings, "\n".join(findings)


def test_hub_writes_on_follows_stage_flags(monkeypatch):
    monkeypatch.setenv("RUN_STAGE1", "0")
    monkeypatch.setenv("RUN_STAGE2", "0")
    from notebooks.run import hub_writes_on

    assert hub_writes_on() is False
    monkeypatch.setenv("RUN_STAGE2", "1")
    assert hub_writes_on() is True


def test_run_surfaces_child_error_text():
    with pytest.raises(SystemExit, match="v1.1 card check failed"):
        run(
            "-c",
            "import sys; sys.stderr.write('v1.1 card check failed; missing sections: latency\\n'); raise SystemExit(1)",
        )


def test_apply_knobs_and_publish_checklist(monkeypatch, capsys):
    for key in (
        "RUN_STAGE1", "RUN_STAGE2", "PD_FRACTION", "STAGE2_LRS",
        "STAGE1_EPOCHS", "STAGE1_LR", "EPOCH_SAMPLES", "HF_MODEL_ID",
    ):
        monkeypatch.setenv(key, "x")
    apply_knobs(RUN_STAGE1=False, RUN_STAGE2=True, PD_FRACTION=0.5, STAGE2_LRS=["1e-4", "5e-5"])
    monkeypatch.setenv("HF_MODEL_ID", "neghani/plantvillage-resnet18")
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
