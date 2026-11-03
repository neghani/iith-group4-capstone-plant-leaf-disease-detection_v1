"""Three-model bake-off on the shared-label field set.

Candidates:
  A  neghani/plantvillage-mobilenetv2@v1.1
  B  neghani/plantvillage-resnet18@v1.1
  C  neghani/plant-leaf-masterdataset-resnet18@v1.0.1
     (local model/outputs/masterdataset-resnet18 used when present)

Protocol:
  1. Build/load model/data/shared_set (shared labels only).
  2. Run each model once at threshold=0 (no-gate) to capture confidences.
  3. On tune only: pick lowest threshold with precision>=0.97 and coverage>=0.50;
     fall back to published threshold if none qualify. Write threshold.json.
  4. Apply chosen threshold on final once. Report top-1, pass, not-sure, CI.
  5. Winner = highest final top-1; paired bootstrap vs runner-up; tie-break
     not-sure then crop coverage.

    python model/bakeoff_shared.py
    python model/bakeoff_shared.py --set_dir model/data/shared_set --out model/outputs/bakeoff
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from app_predictor import WirePredictor, diagnose_bytes  # noqa: E402
from build_shared_set import build as build_shared_set  # noqa: E402
from common import crop_from_class  # noqa: E402
from eval_real_world_set import norm  # noqa: E402
from predict import get_predictor  # noqa: E402
from stats import (  # noqa: E402
    WIDE_CI_NOTE,
    WIDE_N,
    SWEEP,
    frac,
    frac_ci,
    paired_bootstrap,
    threshold_sweep,
    wilson,
)

CANDIDATES = [
    {
        "key": "A_mobilenet_v1.1",
        "name": "PlantVillage MobileNetV2 @v1.1",
        "model": "neghani/plantvillage-mobilenetv2",
        "revision": "v1.1",
        "label_field": "label_pv",
        "published_threshold": 0.50,
        "crop_coverage": 14,
    },
    {
        "key": "B_resnet_pv_v1.1",
        "name": "PlantVillage ResNet-18 @v1.1",
        "model": "neghani/plantvillage-resnet18",
        "revision": "v1.1",
        "label_field": "label_pv",
        "published_threshold": 0.55,
        "crop_coverage": 14,
    },
    {
        "key": "C_master_v1.0.1",
        "name": "MasterDataset ResNet-18 @v1.0.1",
        "model": "neghani/plant-leaf-masterdataset-resnet18",
        "revision": "v1.0.1",
        "local": str(HERE / "outputs" / "masterdataset-resnet18"),
        "label_field": "label_master",
        "published_threshold": 0.30,
        "crop_coverage": 19,
    },
]

MIN_PRECISION = 0.97
MIN_COVERAGE = 0.50
MIN_FINAL_N = WIDE_N  # 100


def load_rows(set_dir: Path, split: str | None = None) -> list[dict]:
    rows = list(csv.DictReader(open(set_dir / "labels.csv", newline="", encoding="utf-8")))
    if split:
        rows = [r for r in rows if r["split"] == split]
    missing = [r["file"] for r in rows if not (set_dir / r["file"]).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} image(s) missing under {set_dir}, e.g. {missing[0]}")
    return rows


def select_threshold_plan(table: list[dict], published: float) -> dict:
    """Lowest threshold with precision>=97% and coverage>=50%; else published."""
    eligible = [
        r for r in table
        if r["ok_n"] > 0
        and r["precision"] is not None
        and r["precision"] >= MIN_PRECISION
        and r["coverage"] >= MIN_COVERAGE
    ]
    if eligible:
        chosen = min(eligible, key=lambda r: r["threshold"])
        return {
            "threshold": chosen["threshold"],
            "source": "tune_rule",
            "rule": f"lowest thr with precision>={MIN_PRECISION} and coverage>={MIN_COVERAGE}",
            "row": chosen,
        }
    # Prefer published if it appears in the sweep; else nearest.
    published_row = next((r for r in table if abs(r["threshold"] - published) < 1e-9), None)
    return {
        "threshold": published,
        "source": "published_fallback",
        "rule": (
            f"no thr met precision>={MIN_PRECISION} & coverage>={MIN_COVERAGE} on tune; "
            f"using published {published}"
        ),
        "row": published_row,
    }


def resolve_model(cand: dict) -> tuple[str, str | None]:
    local = cand.get("local")
    if local and Path(local).is_dir() and (Path(local) / "config.json").is_file():
        return local, None
    return cand["model"], cand["revision"]


def expected(row: dict, cand: dict) -> str:
    return row[cand["label_field"]]


def run_raw(rows: list[dict], set_dir: Path, cand: dict, strategy: str = "B") -> list[dict]:
    """One forward pass per image at threshold=0; keep raw status/conf/pred/top_k."""
    model_id, revision = resolve_model(cand)
    get_predictor.cache_clear()
    pred = WirePredictor(model_id, revision)
    out = []
    for i, r in enumerate(rows, 1):
        res = diagnose_bytes(
            (set_dir / r["file"]).read_bytes(),
            strategy=strategy,
            threshold=0.0,
            predictor=pred,
        )
        true = expected(r, cand)
        status = res.get("status", "ok")
        disease = res.get("disease") or ""
        conf = float(res.get("confidence") or 0.0)
        top = [t.get("disease") for t in (res.get("top_k") or []) if t.get("disease")]
        reached = status != "retake_photo"
        top1 = bool(disease) and reached and norm(disease) == norm(true)
        top3 = reached and any(norm(x) == norm(true) for x in top)
        crop_ok = reached and bool(disease) and crop_from_class(disease) == crop_from_class(true)
        out.append({
            "file": r["file"],
            "split": r["split"],
            "source": r["source"],
            "true": true,
            "pred": disease,
            "confidence": conf,
            "status_raw": status,
            "reached": reached,
            "top1": top1,
            "top3": top3,
            "crop_match": crop_ok,
            "top_k": top,
        })
        if i % 25 == 0 or i == len(rows):
            print(f"  [{cand['key']}] {i}/{len(rows)}", flush=True)
    return out


def apply_threshold(raw: list[dict], threshold: float) -> list[dict]:
    rows = []
    for r in raw:
        if not r["reached"]:
            status = "retake_photo"
        elif r["confidence"] < threshold:
            status = "not_sure"
        else:
            status = "ok"
        passed = status == "ok" and r["top1"]
        rows.append({**r, "status": status, "pass": passed, "threshold": threshold})
    return rows


def summarize(scored: list[dict], threshold: float) -> dict:
    n = len(scored)
    reached = [r for r in scored if r["reached"]]
    top1 = sum(r["top1"] for r in reached)
    top3 = sum(r["top3"] for r in reached)
    crop = sum(r["crop_match"] for r in reached)
    passes = sum(r["pass"] for r in scored)
    not_sure = sum(r["status"] == "not_sure" for r in scored)
    gate = sum(r["status"] == "retake_photo" for r in scored)
    ok_n = sum(r["status"] == "ok" for r in scored)
    ok_wrong = sum(r["status"] == "ok" and not r["top1"] for r in scored)
    lo, hi = wilson(passes, n)
    lo1, hi1 = wilson(top1, len(reached))
    return {
        "n": n,
        "threshold": threshold,
        "top1": top1,
        "top1_n": len(reached),
        "top1_rate": top1 / len(reached) if reached else 0.0,
        "top1_ci": [lo1, hi1],
        "top3": top3,
        "top3_rate": top3 / len(reached) if reached else 0.0,
        "crop_match": crop,
        "crop_match_rate": crop / len(reached) if reached else 0.0,
        "passes": passes,
        "pass_rate": passes / n if n else 0.0,
        "pass_ci": [lo, hi],
        "not_sure": not_sure,
        "not_sure_rate": not_sure / n if n else 0.0,
        "gate": gate,
        "ok_n": ok_n,
        "ok_wrong": ok_wrong,
        "pass_vector": [1.0 if r["pass"] else 0.0 for r in scored],
        "top1_vector": [1.0 if r["top1"] else 0.0 for r in scored],
    }


def pick_winner(results: dict[str, dict]) -> dict:
    """Highest final top-1; if paired CI includes 0, lower not-sure then crop coverage."""
    by_top1 = sorted(results.values(), key=lambda m: m["final"]["top1_rate"], reverse=True)
    lead, second = by_top1[0], by_top1[1]
    delta, lo, hi = paired_bootstrap(
        lead["final"]["top1_vector"],
        second["final"]["top1_vector"],
    )
    decisive = not (lo <= 0.0 <= hi)
    tie_break_used = False
    if decisive:
        winner, runner = lead, second
        note = None
    else:
        tie_break_used = True
        # Top-1 gap not decisive: prefer lower not-sure, then higher crop coverage.
        tied = [lead, second]
        ranked = sorted(
            tied,
            key=lambda m: (m["final"]["not_sure_rate"], -m["crop_coverage"]),
        )
        winner = ranked[0]
        runner = ranked[1]
        note = (
            "Paired top-1 CI includes 0; applied tie-break "
            "(lower not-sure rate, then higher crop coverage)."
        )
        delta, lo, hi = paired_bootstrap(
            winner["final"]["top1_vector"],
            runner["final"]["top1_vector"],
        )
    return {
        "winner": winner["key"],
        "winner_name": winner["name"],
        "runner_up": runner["key"],
        "runner_up_name": runner["name"],
        "metric": "final_top1" if not tie_break_used else "tie_break_not_sure",
        "winner_top1": winner["final"]["top1_rate"],
        "runner_top1": runner["final"]["top1_rate"],
        "paired_delta": delta,
        "paired_ci": [lo, hi],
        "decisive": decisive,
        "tie_break_used": tie_break_used,
        "note": note,
    }


def write_report(out: Path, payload: dict) -> None:
    lines = [
        "# Shared-label bake-off",
        "",
        f"Set: `{payload['set_dir']}` (n={payload['n']}, "
        f"tune={payload['n_tune']}, final={payload['n_final']}, "
        f"classes={payload['n_classes']}).",
        "",
        "Decision metric: **final-split top-1** on images that reached the model. "
        "Threshold frozen on tune only. MasterDataset lab 96% is not used here.",
        "",
        WIDE_CI_NOTE if payload["n_final"] < MIN_FINAL_N else "",
        "",
        "| Model | Final top-1 | Pass (ok+correct) | Not sure | Crop match | Thr |",
        "|---|---|---|---|---|---|",
    ]
    for m in payload["models"]:
        f = m["final"]
        lines.append(
            f"| {m['name']} | {frac_ci(f['top1'], f['top1_n'])} | "
            f"{frac_ci(f['passes'], f['n'])} | {frac(f['not_sure'], f['n'])} | "
            f"{frac(f['crop_match'], f['top1_n'])} | {f['threshold']:.2f} |"
        )
    lines += ["", "## No-gate (threshold 0) on final", ""]
    lines += [
        "| Model | Top-1 | Top-3 | Crop match |",
        "|---|---|---|---|",
    ]
    for m in payload["models"]:
        g = m["final_nogate"]
        lines.append(
            f"| {m['name']} | {frac(g['top1'], g['top1_n'])} | "
            f"{frac(g['top3'], g['top1_n'])} | {frac(g['crop_match'], g['top1_n'])} |"
        )
    w = payload["winner"]
    lines += [
        "",
        "## Winner",
        "",
        f"**{w['winner_name']}** (`{w['winner']}`).",
        f"Paired Δ top-1 vs {w['runner_up_name']}: "
        f"{100 * w['paired_delta']:.1f} pts "
        f"(95% CI {100 * w['paired_ci'][0]:.1f} to {100 * w['paired_ci'][1]:.1f}%).",
    ]
    if w.get("note"):
        lines.append(w["note"])
    if payload["n_final"] < MIN_FINAL_N:
        lines += [
            "",
            f"**Pinning caution:** final n={payload['n_final']} < {MIN_FINAL_N}. "
            "Prefer growing the shared set before changing the live pin.",
        ]
    (out / "bakeoff_report.md").write_text("\n".join(lines).strip() + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set_dir", default=str(HERE / "data" / "shared_set"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "bakeoff"))
    ap.add_argument("--strategy", default="B")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)

    set_dir = Path(args.set_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.rebuild or not (set_dir / "labels.csv").is_file():
        print("[build] shared set …", flush=True)
        build_shared_set(
            set_dir,
            HERE / "data" / "plantdoc_src",
            HERE / "data" / "real_world_set",
            HERE / "outputs" / "masterdataset-resnet18" / "plantdoc_ood.json",
        )

    all_rows = load_rows(set_dir)
    tune_rows = [r for r in all_rows if r["split"] == "tune"]
    final_rows = [r for r in all_rows if r["split"] == "final"]
    print(f"[set] n={len(all_rows)} tune={len(tune_rows)} final={len(final_rows)}", flush=True)

    model_results = []
    for cand in CANDIDATES:
        print(f"\n=== {cand['name']} ===", flush=True)
        raw = run_raw(all_rows, set_dir, cand, strategy=args.strategy)
        raw_tune = [r for r in raw if r["split"] == "tune"]
        raw_final = [r for r in raw if r["split"] == "final"]

        conf = [r["confidence"] for r in raw_tune]
        correct = [r["top1"] for r in raw_tune]
        reached = [r["reached"] for r in raw_tune]
        table = threshold_sweep(conf, correct, reached, SWEEP)
        choice = select_threshold_plan(table, cand["published_threshold"])
        thr = float(choice["threshold"])
        print(f"  threshold={thr:.2f} ({choice['source']})", flush=True)

        thr_path = out / f"threshold_{cand['key']}.json"
        thr_path.write_text(json.dumps({
            "model": cand["model"],
            "revision": cand["revision"],
            "threshold": thr,
            "published_threshold": cand["published_threshold"],
            "selection": choice,
            "sweep": table,
            "min_precision": MIN_PRECISION,
            "min_coverage": MIN_COVERAGE,
            "n_tune": len(raw_tune),
        }, indent=2) + "\n")

        scored_tune = apply_threshold(raw_tune, thr)
        scored_final = apply_threshold(raw_final, thr)
        scored_final_nogate = apply_threshold(raw_final, 0.0)

        # Per-image CSV
        with open(out / f"per_image_{cand['key']}.csv", "w", newline="", encoding="utf-8") as f:
            fields = [
                "file", "split", "source", "true", "pred", "confidence",
                "status", "reached", "top1", "top3", "crop_match", "pass",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in apply_threshold(raw, thr):
                w.writerow({k: r.get(k) for k in fields})

        entry = {
            "key": cand["key"],
            "name": cand["name"],
            "model": resolve_model(cand)[0],
            "revision": resolve_model(cand)[1],
            "label_field": cand["label_field"],
            "crop_coverage": cand["crop_coverage"],
            "threshold_choice": {
                "threshold": thr,
                "source": choice["source"],
                "rule": choice["rule"],
            },
            "tune": summarize(scored_tune, thr),
            "final": summarize(scored_final, thr),
            "final_nogate": summarize(scored_final_nogate, 0.0),
        }
        # Drop large vectors from nested copies used only for winner pick — keep in memory
        model_results.append(entry)
        fsum = entry["final"]
        print(
            f"  final top-1 {frac(fsum['top1'], fsum['top1_n'])}  "
            f"pass {frac(fsum['passes'], fsum['n'])}  "
            f"not_sure {frac(fsum['not_sure'], fsum['n'])}",
            flush=True,
        )

    by_key = {m["key"]: m for m in model_results}
    winner = pick_winner(by_key)

    # Strip pass vectors from JSON (keep in winner calc already done)
    serialisable = []
    for m in model_results:
        sm = json.loads(json.dumps(m))
        for part in ("tune", "final", "final_nogate"):
            sm[part].pop("pass_vector", None)
            sm[part].pop("top1_vector", None)
        serialisable.append(sm)

    payload = {
        "set_dir": str(set_dir),
        "n": len(all_rows),
        "n_tune": len(tune_rows),
        "n_final": len(final_rows),
        "n_classes": len({r["label_pv"] for r in all_rows}),
        "min_final_n_for_pin": MIN_FINAL_N,
        "pin_recommended": len(final_rows) >= MIN_FINAL_N,
        "models": serialisable,
        "winner": winner,
    }
    (out / "bakeoff.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_report(out, payload)
    print(f"\nWinner: {winner['winner_name']} ({winner['winner']})", flush=True)
    print(f"Wrote {out}/bakeoff.json and bakeoff_report.md", flush=True)


if __name__ == "__main__":
    main()
