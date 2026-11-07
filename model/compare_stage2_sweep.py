"""Print a comparison table for the single-factor stage-2 sweep (sweep_mobilenet_stage2.sh).

Reads each variant's eval_results.json (written by train.py after the best checkpoint is loaded)
and calibration.json (written by calibrate.py, optional). Selection metric is PlantDoc VALIDATION
macro-F1; PlantVillage accuracy is the forgetting check (publish rule: drop <= 2 points).

    python model/compare_stage2_sweep.py --root model/outputs/exp_mobilenet_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The v1.0 starting point (PlantVillage only). Filled from v1.0's own published eval if available.
DEFAULT_ORDER = ["ctrl", "warm", "robust", "field", "frac70"]


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def row(root: Path, name: str) -> dict | None:
    run = root / name
    ev = _load(run / "eval_results.json")
    if ev is None:
        return None
    cal = _load(run / "calibration.json") or {}
    chosen = (_load(run / "threshold.json") or {}).get("chosen") or {}
    return {
        "variant": name,
        "pd_val_macro_f1": ev.get("eval_plantdoc_macro_f1"),
        "pd_val_acc": ev.get("eval_plantdoc_accuracy"),
        "pd_val_top3": ev.get("eval_plantdoc_top3_accuracy"),
        "pd_val_ece_after": cal.get("ece_after"),
        "pd_val_temperature": cal.get("temperature"),
        "pv_val_acc": ev.get("eval_plantvillage_accuracy"),
        "pass_rate_at_threshold": chosen.get("pass_rate") if chosen else None,
        "threshold": (_load(run / "threshold.json") or {}).get("threshold"),
    }


def fmt(x, pct=True) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent / "outputs" / "exp_mobilenet_v2"))
    ap.add_argument("--variants", nargs="*", default=DEFAULT_ORDER)
    args = ap.parse_args(argv)
    root = Path(args.root)
    rows = [r for r in (row(root, n) for n in args.variants) if r]
    if not rows:
        raise SystemExit(f"no finished variants under {root}")
    base = rows[0]
    print("| variant | PD val macro-F1 | Δ vs ctrl | PD val acc | PD val top-3 | PV val acc | PD val ECE (after T) | pass rate @thr | thr |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        d = None if r["pd_val_macro_f1"] is None or base["pd_val_macro_f1"] is None else r["pd_val_macro_f1"] - base["pd_val_macro_f1"]
        delta = "" if d is None else f"{100 * d:+.1f} pts"
        print(
            f"| {r['variant']} | {fmt(r['pd_val_macro_f1'])} | {delta} | {fmt(r['pd_val_acc'])} | "
            f"{fmt(r['pd_val_top3'])} | {fmt(r['pv_val_acc'])} | {fmt(r['pd_val_ece_after'])} | "
            f"{fmt(r['pass_rate_at_threshold'])} | {fmt(r['threshold'], pct=False)} |"
        )


if __name__ == "__main__":
    main()
