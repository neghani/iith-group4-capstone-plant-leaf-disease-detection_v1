"""Fill the Hugging Face model-card README from artifacts already on disk.

Called by publish.py. Does not train, re-run the final test, or measure latency.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
LATENCY_PATH = HERE / "latency.json"
NOT_RECORDED = "Not recorded."

WEAK_CLASSES = (
    ("Potato___healthy", "Potato healthy"),
    ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn Cercospora"),
)

SECTIONS = ("epochs", "weak_classes", "before_after", "calibration", "latency", "strategies")
MARKERS = {name: (f"<!-- {name.upper()}_START -->", f"<!-- {name.upper()}_END -->") for name in SECTIONS}


def _first(*paths: Path | None) -> Path | None:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def find_artifact(model_dir: Path, outputs: Path, *rels: str) -> Path | None:
    for rel in rels:
        found = _first(model_dir / rel, outputs / rel)
        if found is not None:
            return found
    return None


def replace_block(text: str, start: str, end: str, body: str) -> str:
    block = f"{start}\n{body.rstrip()}\n{end}"
    if start in text and end in text:
        return re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, text, flags=re.S)
    return text.rstrip() + "\n\n" + block + "\n"


def _pick(rec: dict, *keys: str):
    for k in keys:
        if rec.get(k) is not None:
            return rec[k]
    return None


def epoch_rows(log_history: list[dict]) -> list[dict]:
    """One row per epoch from Trainer log_history (eval_strategy=epoch)."""
    by_epoch: dict[int, dict] = {}
    for rec in log_history:
        loss = _pick(rec, "eval_plantdoc_loss", "eval_loss", "eval_plantvillage_loss")
        acc = _pick(rec, "eval_plantdoc_accuracy", "eval_accuracy", "eval_plantvillage_accuracy")
        if loss is None and acc is None:
            continue
        epoch = rec.get("epoch")
        if epoch is None:
            continue
        row = {"epoch": float(epoch), "eval_loss": loss, "eval_accuracy": acc}
        if rec.get("loss") is not None:
            row["train_loss"] = rec["loss"]
        by_epoch[round(float(epoch))] = row
    return [by_epoch[k] for k in sorted(by_epoch)]


def _fmt_epoch(value: float) -> str:
    return str(int(value)) if abs(value - round(value)) < 1e-6 else f"{value:.2f}"


def _fmt_num(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "-"
    return f"{100 * float(value):.1f}%"


def fill_epochs(model_dir: Path, outputs: Path) -> str | None:
    path = find_artifact(model_dir, outputs, "trainer_state.json")
    if path is None:
        return None
    state = json.loads(path.read_text())
    rows = epoch_rows(state.get("log_history") or [])
    if not rows:
        return None
    has_train = any("train_loss" in r for r in rows)
    header = ["Epoch", "Val loss", "Val accuracy"] + (["Train loss"] if has_train else [])
    lines = [
        "Rising validation loss in later epochs is the overfitting check.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for r in rows:
        cells = [_fmt_epoch(r["epoch"]), _fmt_num(r["eval_loss"]), _fmt_pct(r["eval_accuracy"])]
        if has_train:
            cells.append(_fmt_num(r.get("train_loss")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fill_weak_classes(model_dir: Path, outputs: Path) -> str | None:
    path = find_artifact(model_dir, outputs, "reports/metrics.json", "metrics.json")
    if path is None:
        return None
    metrics = json.loads(path.read_text())
    by_name = {r["class"]: r for r in metrics.get("per_class") or []}
    rows = []
    for class_id, display in WEAK_CLASSES:
        rec = by_name.get(class_id)
        if rec is None:
            return None
        rows.append((class_id, display, rec))
    lines = [
        "The weakest PlantVillage test classes are **Potato healthy** and **Corn Cercospora**.",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for class_id, display, rec in rows:
        lines.append(
            f"| {display} (`{class_id}`) | {_fmt_pct(rec.get('precision'))} | "
            f"{_fmt_pct(rec.get('recall'))} | {_fmt_pct(rec.get('f1'))} | {rec.get('support', '')} |"
        )
    lines += [
        "",
        "Confusion matrix: [`reports/confusion_matrix.png`](reports/confusion_matrix.png).",
    ]
    return "\n".join(lines)


def fill_before_after(model_dir: Path, outputs: Path) -> str | None:
    path = find_artifact(model_dir, outputs, "final_summary.json", "reports/final_summary.json")
    if path is None:
        return None
    data = json.loads(path.read_text())
    v10 = (data.get("v1.0") or {}).get("real_world_set") or {}
    v11 = (data.get("v1.1") or {}).get("real_world_set") or {}
    if not v10.get("label") or not v11.get("label"):
        return None
    dest = model_dir / "final_summary.json"
    if path.resolve() != dest.resolve():
        dest.write_text(path.read_text())
    return "\n".join([
        "Same real-world set, live-app chain (gate + strategy B). Each cell is "
        "passes/n (rate), 95% Wilson CI.",
        "",
        "| Release | End-to-end pass rate |",
        "|---|---|",
        f"| v1.0 (before fine-tune) | {v10['label']} |",
        f"| v1.1 (after fine-tune) | {v11['label']} |",
    ])


def fill_calibration(model_dir: Path, outputs: Path) -> str | None:
    path = find_artifact(model_dir, outputs, "calibration.json")
    if path is None:
        return None
    calib = json.loads(path.read_text())
    before, after = calib.get("ece_before"), calib.get("ece_after")
    if before is None or after is None:
        return None
    plot = find_artifact(model_dir, outputs, "reports/reliability.png", "reliability.png")
    reports = model_dir / "reports"
    if plot is not None:
        dest = reports / "reliability.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if plot.resolve() != dest.resolve():
            shutil.copy2(plot, dest)
    lines = [
        "Temperature scaling fitted on PlantDoc validation (15 bins).",
        "",
        "| | Expected calibration error |",
        "|---|---|",
        f"| Before | {_fmt_pct(before)} |",
        f"| After | {_fmt_pct(after)} |",
    ]
    if plot is not None:
        lines += ["", "Reliability (before and after on the same axes):", "",
                  "![Reliability](reports/reliability.png)"]
    return "\n".join(lines)


def fill_latency(latency_path: Path) -> str | None:
    if not latency_path.exists():
        return None
    data = json.loads(latency_path.read_text())
    forward = data.get("model_forward_ms")
    warm = data.get("predict_warm_label") or data.get("predict_warm_s")
    if forward is None or warm is None:
        return None
    warm_s = warm if isinstance(warm, str) else f"about {warm} s"
    note = data.get("note") or (
        "The gap is the app path (quality gate, crop, HTTP), not the forward pass."
    )
    return "\n".join([
        "| Step | Time |",
        "|---|---|",
        f"| Model forward | {forward} ms |",
        f"| `/predict` (warm) | {warm_s} |",
        "",
        note,
    ])


def _strategy_table(md: str) -> str | None:
    lines = md.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("| Strategy")), None)
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    table = "\n".join(lines[start:end]).strip()
    if not table:
        return None
    letters = {ch for ch in "ABCDE" if re.search(rf"\b{ch}\b", table)}
    if letters < set("ABCDE"):
        return None
    return table


def fill_strategies(model_dir: Path, outputs: Path) -> str | None:
    path = find_artifact(model_dir, outputs, "strategy_comparison.md",
                         "reports/strategy_comparison.md")
    if path is None:
        return None
    table = _strategy_table(path.read_text())
    if table is None:
        return None
    return "Tune-split comparison from `eval_strategies.py`. Each strategy at its own best threshold.\n\n" + table


FILLERS = {
    "epochs": lambda model_dir, outputs, latency_path: fill_epochs(model_dir, outputs),
    "weak_classes": lambda model_dir, outputs, latency_path: fill_weak_classes(model_dir, outputs),
    "before_after": lambda model_dir, outputs, latency_path: fill_before_after(model_dir, outputs),
    "calibration": lambda model_dir, outputs, latency_path: fill_calibration(model_dir, outputs),
    "latency": lambda model_dir, outputs, latency_path: fill_latency(latency_path),
    "strategies": lambda model_dir, outputs, latency_path: fill_strategies(model_dir, outputs),
}


def assemble_card(
    model_dir: Path,
    outputs: Path | None = None,
    latency_path: Path | None = None,
) -> dict[str, list[str]]:
    """Fill marker sections in model_dir/README.md. Returns filled vs missing section names."""
    model_dir = Path(model_dir)
    outputs = Path(outputs) if outputs is not None else OUTPUTS
    latency_path = Path(latency_path) if latency_path is not None else LATENCY_PATH
    card_path = model_dir / "README.md"
    text = card_path.read_text() if card_path.exists() else ""

    filled, missing = [], []
    for name in SECTIONS:
        start, end = MARKERS[name]
        body = FILLERS[name](model_dir, outputs, latency_path)
        if body:
            filled.append(name)
            text = replace_block(text, start, end, body)
        else:
            missing.append(name)
            print(f"Warning: model card section '{name}' not recorded "
                  f"(missing artifact under {model_dir} or {outputs}).")
            text = replace_block(text, start, end, NOT_RECORDED)

    card_path.write_text(text if text.endswith("\n") else text + "\n")
    return {"filled": filled, "missing": missing}
