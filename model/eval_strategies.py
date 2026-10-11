"""Compare crop strategies A-E (shared/vision/strategies.py) on the real-world sets.

    python model/eval_strategies.py --split tune
    python model/eval_strategies.py --split final --strategy B --threshold 0.5

Datasets (both optional, at least one must have images):
    model/data/real_world_set/  Cropped-PlantDoc, one leaf per image
    model/data/full_photos/     our own uncropped phone photos, several leaves per frame

Every image goes through the live chain: original file bytes -> shared decode + gate ->
strategy crops -> 256x256 JPEG q95 (what the API sends) -> local copy of the Hub weights.

Pass = correct disease AND status "ok"; pass rate = passes / all images (gate rejections count
as misses).

Selection rule, fixed before any results were seen (do not change it after looking):
  1. Tune split only. Per strategy, sweep the threshold 0.30..0.90 (step 0.05) and keep the one
     with the highest pass rate among thresholds where >= 70% of "ok" answers are correct
     (needs at least one "ok"). Ties go to the higher threshold. A strategy with no such
     threshold is marked as missing the precision bar and cannot be selected.
  2. Pick the eligible strategy with the highest tune pass rate (ties: B, then A..E order).
  3. If that is not B and its paired gain over B is under 5 points, or the 95% paired bootstrap
     interval of the difference includes 0, keep B (simplest wins). Keep B as well if no
     strategy is eligible.
  4. Run the chosen strategy + threshold on the final split exactly once (enforced with
     outputs/final_run.json). Never tune on final.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from eval_real_world_set import norm  # noqa: E402
from shared.vision.strategies import DESCRIPTIONS, STRATEGY_IDS, decide_status  # noqa: E402
from stats import MIN_PRECISION, N_BOOT, SEED, WIDE_CI_NOTE, ci, frac, paired_bootstrap, wilson  # noqa: E402,F401
from stats import model_key as _model_key  # noqa: E402

THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(13)]
MIN_GAIN = 0.05
REFERENCE = "B"
DEFAULT_SETS = [HERE / "data" / "real_world_set", HERE / "data" / "full_photos"]


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def load_rows(set_dirs: list[Path], split: str | None) -> list[dict]:
    rows = []
    for root in set_dirs:
        labels = root / "labels.csv"
        if not labels.exists():
            print(f"[data] {root.name}: no labels.csv, skipped")
            continue
        sources = {}
        if (root / "sources.csv").exists():
            sources = {r["file"]: r.get("original_path", "") for r in csv.DictReader(open(root / "sources.csv"))}
        found = 0
        for r in csv.DictReader(open(labels)):
            if split and r["split"] != split:
                continue
            path = root / r["file"]
            if not path.exists():
                print(f"[data] {root.name}: missing image {r['file']}, skipped")
                continue
            rows.append({
                "dataset": root.name, "file": r["file"], "path": path, "label": r["label"],
                "split": r["split"], "plant_id": (r.get("plant_id") or "").strip(),
                "notes": r.get("notes", ""), "source": sources.get(r["file"], ""),
            })
            found += 1
        print(f"[data] {root.name}: {found} images" + (f" in split '{split}'" if split else ""))
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dhash(path: Path) -> int:
    img = ImageOps.exif_transpose(Image.open(path)).convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def leakage_check(set_dirs: list[Path]) -> list[str]:
    """Problems that would let tune information reach final: shared plants, duplicate images."""
    rows = load_rows(set_dirs, None)
    problems = []
    by_split: dict[str, set] = {"tune": set(), "final": set()}
    for r in rows:
        if r["plant_id"] and r["split"] in by_split:
            by_split[r["split"]].add((r["dataset"], r["plant_id"]))
    for ds, pid in sorted(by_split["tune"] & by_split["final"]):
        problems.append(f"plant_id '{pid}' ({ds}) is in both tune and final")
    full = [r for r in rows if r["dataset"] == "full_photos" and not r["plant_id"]]
    if full:
        problems.append(f"{len(full)} full_photos rows have no plant_id, so plant overlap cannot be checked")

    tune = [r for r in rows if r["split"] == "tune"]
    final = [r for r in rows if r["split"] == "final"]
    sha_t = {_sha(r["path"]): r["file"] for r in tune}
    for r in final:
        h = _sha(r["path"])
        if h in sha_t:
            problems.append(f"identical image in tune and final: {sha_t[h]} == {r['file']}")
    src_t = {r["source"]: r["file"] for r in tune if r["source"]}
    for r in final:
        if r["source"] and r["source"] in src_t:
            problems.append(f"same source photo in tune and final: {src_t[r['source']]} / {r['file']}")
    dh_t = [(_dhash(r["path"]), r["file"]) for r in tune]
    for r in final:
        h = _dhash(r["path"])
        for ht, ft in dh_t:
            if bin(h ^ ht).count("1") <= 4:
                problems.append(f"near-duplicate images in tune and final (dHash): {ft} ~ {r['file']}")
    return problems


def fingerprint(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: (r["dataset"], r["file"])):
        h.update(f"{r['dataset']}/{r['file']}\0{r['label']}\0{_sha(r['path'])}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------

def run_all(rows: list[dict], strategies: list[str], predictor, tta: bool) -> dict[str, list[dict]]:
    from app_predictor import diagnose_bytes

    out: dict[str, list[dict]] = {s: [] for s in strategies}
    rejected = []
    for i, r in enumerate(rows, 1):
        data = r["path"].read_bytes()
        for s in strategies:
            t0 = time.perf_counter()
            res = diagnose_bytes(data, strategy=s, threshold=0.5, predictor=predictor, tta=tta)
            res["seconds"] = time.perf_counter() - t0
            out[s].append(res)
        reason = out[strategies[0]][-1]["gate_reason"]
        if reason:
            rejected.append((r["file"], reason))
            print(f"[gate] {r['dataset']}/{r['file']}: rejected ({reason})")
        if i % 10 == 0 or i == len(rows):
            print(f"[run] {i}/{len(rows)} images")
    return out


def outcome(res: dict, label: str, threshold: float) -> tuple[str, bool]:
    if res["status"] == "retake_photo":
        return "retake_photo", False
    status = decide_status(res["confidence"], res["agreement"], threshold)
    return status, norm(res["disease"]) == norm(label)


def summarize(results: list[dict], rows: list[dict], threshold: float) -> dict:
    n = len(rows)
    st = [outcome(res, r["label"], threshold) for res, r in zip(results, rows)]
    passes = [s == "ok" and c for s, c in st]
    reached = [s != "retake_photo" for s, _ in st]
    ok_n = sum(s == "ok" for s, _ in st)
    ok_right = sum(passes)
    top1 = sum(c for (s, c), rch in zip(st, reached) if rch)
    lo, hi = wilson(sum(passes), n)
    return {
        "threshold": threshold, "n": n, "passes": sum(passes), "pass_rate": sum(passes) / n if n else 0.0,
        "pass_ci": (lo, hi), "reached": sum(reached), "top1_correct": top1,
        "ok_n": ok_n, "ok_wrong": ok_n - ok_right, "precision": ok_right / ok_n if ok_n else None,
        "not_sure_n": sum(s == "not_sure" for s, _ in st), "gate_n": sum(s == "retake_photo" for s, _ in st),
        "sec_per_image": float(np.mean([res["seconds"] for res in results])) if results else 0.0,
        "pass_vector": passes, "statuses": [s for s, _ in st],
    }


def sweep(results: list[dict], rows: list[dict]) -> tuple[list[dict], dict, bool]:
    table = [summarize(results, rows, t) for t in THRESHOLDS]
    eligible = [m for m in table if m["ok_n"] > 0 and m["precision"] >= MIN_PRECISION]
    if eligible:
        return table, max(eligible, key=lambda m: (m["pass_rate"], m["threshold"])), True
    best = max(table, key=lambda m: (m["precision"] or 0.0, m["pass_rate"], m["threshold"]))
    return table, best, False


# ---------------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------------

def per_dataset(m: dict, rows: list[dict]) -> dict[str, tuple[int, int]]:
    out: dict[str, list[int]] = {}
    for p, r in zip(m["pass_vector"], rows):
        k = out.setdefault(r["dataset"], [0, 0])
        k[0] += p
        k[1] += 1
    return {ds: (k, n) for ds, (k, n) in out.items()}


def metrics_rows(sid: str, m: dict, rows: list[dict]) -> list[str]:
    return [
        f"| Pass rate (correct and ok, of all images) | {frac(m['passes'], m['n'])}, 95% CI {ci(*m['pass_ci'])} |",
        f"| Top-1 on images that reached the model | {frac(m['top1_correct'], m['reached'])} |",
        f"| Wrong but confident (of ok answers) | {frac(m['ok_wrong'], m['ok_n'])} |",
        f"| Not sure | {frac(m['not_sure_n'], m['n'])} |",
        f"| Rejected by the gate | {frac(m['gate_n'], m['n'])} |",
        f"| Seconds per image | {m['sec_per_image']:.3f} |",
    ] + [f"| Pass rate, {ds} | {frac(k, n)} |" for ds, (k, n) in per_dataset(m, rows).items()]


def gate_section(results: list[dict], rows: list[dict]) -> list[str]:
    rej = [(r, res["gate_reason"]) for r, res in zip(rows, results) if res["gate_reason"]]
    lines = ["## Quality-gate rejections", ""]
    if not rej:
        return lines + [f"None of the {len(rows)} images were rejected.", ""]
    counts = Counter(reason for _, reason in rej)
    lines += [f"{len(rej)} of {len(rows)} images rejected: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()), "",
              "| Image | Reason |", "|---|---|"]
    lines += [f"| {r['dataset']}/{r['file']} | {reason} |" for r, reason in rej]
    return lines + [""]


def write_per_image(path: Path, rows: list[dict], results: dict[str, list[dict]], chosen: dict[str, float]) -> None:
    fields = ["dataset", "file", "split", "true_label"]
    for s in results:
        fields += [f"{s}_pred", f"{s}_status", f"{s}_confidence", f"{s}_n_crops", f"{s}_agreement"]
    fields += ["gate_reason", "leaf_found", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            rec = {"dataset": r["dataset"], "file": r["file"], "split": r["split"], "true_label": r["label"],
                   "notes": r["notes"]}
            for s, res_list in results.items():
                res = res_list[i]
                status, _ = outcome(res, r["label"], chosen[s])
                rec.update({
                    f"{s}_pred": res["disease"] or "", f"{s}_status": status,
                    f"{s}_confidence": "" if res["confidence"] is None else round(res["confidence"], 4),
                    f"{s}_n_crops": res["n_crops"],
                    f"{s}_agreement": "" if res["agreement"] is None else round(res["agreement"], 3),
                })
                rec["gate_reason"] = res["gate_reason"] or ""
                rec["leaf_found"] = res["leaf_found"]
            w.writerow(rec)


# ---------------------------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------------------------

def run_tune(a, rows, predictor, out: Path, leaks: list[str]) -> None:
    strategies = [s for s in STRATEGY_IDS if s in (a.strategy or STRATEGY_IDS) or s == REFERENCE]
    results = run_all(rows, strategies, predictor, a.tta)

    sweeps, best, meets = {}, {}, {}
    for s in strategies:
        sweeps[s], best[s], meets[s] = sweep(results[s], rows)
    diffs = {s: paired_bootstrap(best[s]["pass_vector"], best[REFERENCE]["pass_vector"]) for s in strategies}

    # Selection rule (module docstring).
    eligible = [s for s in strategies if meets[s]]
    order = [REFERENCE] + [s for s in STRATEGY_IDS if s != REFERENCE]
    if not eligible:
        top, chosen = None, REFERENCE
        why = (f"No strategy reaches {MIN_PRECISION:.0%} correct among 'ok' answers at any threshold on tune, "
               f"so B is kept (rule 3). B's threshold is the one with the highest precision.")
    else:
        top = max(eligible, key=lambda s: (best[s]["pass_rate"], -order.index(s)))
        gain, lo, hi = diffs[top]
        if top == REFERENCE:
            chosen, why = REFERENCE, "B has the highest tune pass rate."
        elif gain < MIN_GAIN or lo <= 0.0 <= hi:
            chosen = REFERENCE
            why = (f"{top} has the highest tune pass rate, but its paired gain over B is {100 * gain:+.1f} points "
                   f"(95% bootstrap CI {100 * lo:+.1f} to {100 * hi:+.1f}); under {100 * MIN_GAIN:.0f} points or "
                   f"the interval includes 0, so B is kept (simplest wins).")
        else:
            chosen = top
            why = (f"{top} beats B by {100 * gain:+.1f} points (95% bootstrap CI {100 * lo:+.1f} to "
                   f"{100 * hi:+.1f}), which clears the {100 * MIN_GAIN:.0f}-point bar and excludes 0.")
    thr = best[chosen]["threshold"]

    # Table.
    n = len(rows)
    header = ["strategy", "description", "threshold", "meets_precision_bar", "n", "passes", "pass_rate",
              "pass_ci_low", "pass_ci_high", "reached_model", "top1_correct", "top1_rate", "ok_answers",
              "ok_wrong", "wrong_confident_rate", "not_sure", "not_sure_rate", "gate_rejected", "gate_rate",
              "sec_per_image", "diff_vs_B", "diff_ci_low", "diff_ci_high"]
    datasets = sorted({r["dataset"] for r in rows})
    header += [f"pass_{ds}" for ds in datasets]
    with open(out / "strategy_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in strategies:
            m = best[s]
            pd_ = per_dataset(m, rows)
            w.writerow([s, DESCRIPTIONS[s], m["threshold"], meets[s], n, m["passes"], round(m["pass_rate"], 4),
                        round(m["pass_ci"][0], 4), round(m["pass_ci"][1], 4), m["reached"], m["top1_correct"],
                        round(m["top1_correct"] / max(1, m["reached"]), 4), m["ok_n"], m["ok_wrong"],
                        round(m["ok_wrong"] / m["ok_n"], 4) if m["ok_n"] else "", m["not_sure_n"],
                        round(m["not_sure_n"] / n, 4), m["gate_n"], round(m["gate_n"] / n, 4),
                        round(m["sec_per_image"], 4), *(round(x, 4) for x in diffs[s])]
                       + [f"{pd_.get(ds, (0, 0))[0]}/{pd_.get(ds, (0, 0))[1]}" for ds in datasets])
    with open(out / "strategy_threshold_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "threshold", "passes", "n", "ok_answers", "ok_correct", "precision", "not_sure", "gate"])
        for s in strategies:
            for m in sweeps[s]:
                w.writerow([s, m["threshold"], m["passes"], n, m["ok_n"], m["ok_n"] - m["ok_wrong"],
                            "" if m["precision"] is None else round(m["precision"], 4), m["not_sure_n"], m["gate_n"]])
    write_per_image(out / "per_image.csv", rows, results, {s: best[s]["threshold"] for s in strategies})

    counts = Counter(r["dataset"] for r in rows)
    table = ["| Strategy | Thr | Pass rate (95% CI) | Top-1 reached | Wrong-confident (of ok) | Not sure | Gate | s/img | vs B (95% CI) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in strategies:
        m = best[s]
        g, lo, hi = diffs[s]
        flag = "" if meets[s] else " (misses 70% bar)"
        table.append(
            f"| {s} {DESCRIPTIONS[s]}{flag} | {m['threshold']:.2f} | {frac(m['passes'], n)}, {ci(*m['pass_ci'])} | "
            f"{frac(m['top1_correct'], m['reached'])} | {frac(m['ok_wrong'], m['ok_n'])} | {frac(m['not_sure_n'], n)} | "
            f"{frac(m['gate_n'], n)} | {m['sec_per_image']:.3f} | "
            + ("reference" if s == REFERENCE else f"{100 * g:+.1f} pts ({100 * lo:+.1f} to {100 * hi:+.1f})") + " |"
        )
    surprising = [s for s in strategies if best[s]["pass_rate"] >= 0.8 or diffs[s][0] >= 0.3]
    md = [
        "# Crop strategy comparison (tune split)", "",
        f"Run {datetime.datetime.now().isoformat(timespec='seconds')}. Images: {n} "
        f"({', '.join(f'{ds} {k}' for ds, k in sorted(counts.items()))}). Tune fingerprint `{fingerprint(rows)[:12]}`."
        + (" Horizontal-flip TTA on." if a.tta else ""), "",
        "Each strategy is shown at its own best threshold (highest pass rate with at least 70% of 'ok' answers "
        "correct). 'vs B' is the paired difference in pass rate with a 1000-resample bootstrap interval "
        f"(seed {SEED}). {WIDE_CI_NOTE}", "",
        *table, "",
        "## Decision", "",
        f"**Chosen: strategy {chosen} ({DESCRIPTIONS[chosen]}) at threshold {thr:.2f}.** {why}", "",
        "Run it on the final split once:", "",
        "```bash",
        f"python model/eval_strategies.py --split final --strategy {chosen} --threshold {thr:.2f}"
        + (" --tta" if a.tta else ""),
        "```", "",
        "## Pass rate by dataset (at each strategy's threshold)", "",
        "| Strategy | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets),
    ]
    for s in strategies:
        pd_ = per_dataset(best[s], rows)
        md.append(f"| {s} | " + " | ".join(frac(*pd_.get(ds, (0, 0))) for ds in datasets) + " |")
    md += ["", "## Threshold sweep (passes / ok answers / precision)", "",
           "| Thr | " + " | ".join(strategies) + " |", "|---|" + "---|" * len(strategies)]
    for i, t in enumerate(THRESHOLDS):
        cells = []
        for s in strategies:
            m = sweeps[s][i]
            p = "-" if m["precision"] is None else f"{100 * m['precision']:.0f}%"
            cells.append(f"{m['passes']} / {m['ok_n']} / {p}")
        md.append(f"| {t:.2f} | " + " | ".join(cells) + " |")
    md += [""] + gate_section(results[strategies[0]], rows)
    md += ["## Leakage check (tune vs final)", ""]
    md += [f"- {p}" for p in leaks] if leaks else ["No shared plant_id, identical image, shared source photo or near-duplicate found."]
    if surprising:
        md += ["", f"**Check before trusting:** {', '.join(surprising)} look unusually good on tune; "
               "see the leakage check above and the per-image file."]
    if counts.get("full_photos", 0) == 0:
        md += ["", "**Note:** full_photos has no images yet, so this compares strategies on pre-cropped "
               "PlantDoc leaves only. Cropping strategies matter most on multi-leaf phone photos; rerun "
               "tune once full_photos is collected, before running final."]
    md += ["", "Files: `strategy_comparison.csv`, `strategy_threshold_sweep.csv`, `per_image.csv`."]
    (out / "strategy_comparison.md").write_text("\n".join(md) + "\n")
    (out / "selection.json").write_text(json.dumps({
        "strategy": chosen, "threshold": thr, "tta": a.tta, "tune_fingerprint": fingerprint(rows), "n_tune": n,
        "tune_pass_rate": best[chosen]["pass_rate"], "top_by_pass_rate": top, "reason": why,
        "date": datetime.date.today().isoformat(),
    }, indent=2) + "\n")

    print()
    print("\n".join(table))
    print(f"\nChosen: {chosen} @ {thr:.2f}. {why}")
    if counts.get("full_photos", 0) == 0:
        print("Note: full_photos is empty; this tune result covers pre-cropped PlantDoc images only.")
    if leaks:
        print(f"Leakage check found {len(leaks)} problem(s); see strategy_comparison.md.")
    print(f"Wrote {out}/strategy_comparison.csv, strategy_comparison.md, per_image.csv")


LEGACY_MODEL_KEY = "neghani/plantvillage-mobilenetv2@v1.0"  # locks written before model_key existed


def model_key() -> str:
    """Identity of the weights under test: the file hash for a local run dir, else id@revision."""
    from app_predictor import MODEL, REVISION

    return _model_key(MODEL, REVISION)


def run_final(a, rows, predictor, out: Path, leaks: list[str]) -> None:
    if not a.strategy or len(a.strategy) != 1 or a.threshold is None:
        sys.exit("--split final needs exactly one --strategy and a --threshold (from the tune run).")
    if leaks:
        sys.exit("Leakage between tune and final; fix the split before running final:\n  " + "\n  ".join(leaks))
    s = a.strategy[0]
    fp = fingerprint(rows)
    mk = model_key()
    lock = out / "final_run.json"
    if lock.exists() and not a.force:
        prev = json.loads(lock.read_text())
        if prev.get("final_fingerprint") == fp and prev.get("model_key", LEGACY_MODEL_KEY) == mk:
            sys.exit(f"The final split was already evaluated for this model on {prev.get('date')} with strategy "
                     f"{prev.get('strategy')} @ {prev.get('threshold')}; see {out / 'final_report.md'}. "
                     "Final is run once per model. Use --force only to regenerate the same run.")
        if prev.get("final_fingerprint") == fp:
            print(f"[note] final was already used for another model ({prev.get('model_key', LEGACY_MODEL_KEY)}); "
                  f"this is a different model ({mk}). Use a separate --out so the earlier report is kept.")
    sel_path = out / "selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    if sel and (sel.get("strategy") != s or abs(sel.get("threshold", -1) - a.threshold) > 1e-9):
        print(f"[warn] tune selection was {sel.get('strategy')} @ {sel.get('threshold')}; running {s} @ {a.threshold}.")

    results = run_all(rows, [s], predictor, a.tta)[s]
    m = summarize(results, rows, a.threshold)
    write_per_image(out / "final_per_image.csv", rows, {s: results}, {s: a.threshold})
    n = len(rows)
    counts = Counter(r["dataset"] for r in rows)
    lines = [
        "# Final split report", "",
        f"Strategy **{s}** ({DESCRIPTIONS[s]}) at confidence threshold **{a.threshold:.2f}**"
        + (" with horizontal-flip TTA" if a.tta else "") + ", chosen on the tune split"
        + (f" (tune pass rate {100 * sel['tune_pass_rate']:.1f}% on {sel['n_tune']} images)" if sel else "")
        + ". This split was evaluated once and not used for any choice.", "",
        f"Images: {n} ({', '.join(f'{ds} {k}' for ds, k in sorted(counts.items()))}). "
        f"Run {datetime.datetime.now().isoformat(timespec='seconds')}. Final fingerprint `{fp[:12]}`. Model `{mk}`.", "",
        "| Metric | Value |", "|---|---|", *metrics_rows(s, m, rows), "",
        WIDE_CI_NOTE + " Pass = correct disease and status 'ok'; gate rejections count as misses.", "",
    ]
    if sel and m["pass_rate"] > sel["tune_pass_rate"] + 0.15:
        lines += [f"**Final is {100 * (m['pass_rate'] - sel['tune_pass_rate']):.0f} points above tune.** "
                  "The leakage check passed, but review `final_per_image.csv` before quoting this number.", ""]
    lines += gate_section(results, rows)
    lines += ["## Leakage check", "", "No shared plant_id, identical image, shared source photo or near-duplicate "
              "between tune and final.", "", "Per-image results: `final_per_image.csv`."]
    if counts.get("full_photos", 0) == 0:
        lines += ["", "full_photos had no images, so this covers pre-cropped PlantDoc leaves only."]
    (out / "final_report.md").write_text("\n".join(lines) + "\n")
    lock.write_text(json.dumps({"strategy": s, "threshold": a.threshold, "tta": a.tta, "final_fingerprint": fp,
                                "model_key": mk,
                                "n": n, "pass_rate": m["pass_rate"], "date": datetime.date.today().isoformat()},
                               indent=2) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}/final_report.md, final_per_image.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["tune", "final"])
    ap.add_argument("--strategy", nargs="+", choices=list(STRATEGY_IDS), default=None,
                    help="tune: subset to compare (B is always included); final: the chosen strategy")
    ap.add_argument("--threshold", type=float, default=None, help="final: the threshold chosen on tune")
    ap.add_argument("--set_dir", action="append", default=None, help="dataset folder (repeatable)")
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--model", default=None, help="Hub id or local run dir (default HF_MODEL_ID)")
    ap.add_argument("--revision", default=None, help="default HF_MODEL_REVISION")
    ap.add_argument("--device", default="cpu", help="cpu keeps results bit-for-bit reproducible")
    ap.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation")
    ap.add_argument("--force", action="store_true", help="final: regenerate an existing final run")
    a = ap.parse_args()

    if a.model:
        os.environ["EVAL_MODEL"] = a.model
    if a.revision:
        os.environ["EVAL_REVISION"] = a.revision
    random.seed(SEED)
    np.random.seed(SEED)

    set_dirs = [Path(p) for p in a.set_dir] if a.set_dir else DEFAULT_SETS
    rows = load_rows(set_dirs, a.split)
    if not rows:
        sys.exit(f"No images found for split '{a.split}' in {', '.join(str(p) for p in set_dirs)}")
    leaks = leakage_check(set_dirs)
    for p in leaks:
        print(f"[leakage] {p}")

    import torch
    from app_predictor import MODEL, REVISION, WirePredictor

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.perf_counter()
    predictor = WirePredictor(MODEL, REVISION, a.device)
    predictor.predict_proba([Image.new("RGB", (256, 256), (90, 140, 60))])  # warm-up, not timed per image
    print(f"[model] {MODEL}@{REVISION} on {a.device}, loaded in {time.perf_counter() - t0:.1f}s")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (run_tune if a.split == "tune" else run_final)(a, rows, predictor, out, leaks)


if __name__ == "__main__":
    main()
