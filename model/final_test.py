"""One-shot final test of a candidate vs a baseline. Same weights skip; different weights are refused. --force regenerates.

    # Historical v1.1 vs v1.0 (defaults):
    python -m model.final_test --v11 /content/drive/.../stage2/best \
        --revision_name v1.1 --guard hub:neghani/plantvillage-mobilenetv2

    # v1.2 vs published v1.1 (writes under a dedicated out dir, never model/outputs):
    python -m model.final_test --v11 /content/drive/.../v1.2/stage2 \
        --revision_name v1.2 --out /content/drive/.../v1.2/final \
        --baseline_revision v1.1 --baseline_threshold 0.5 --baseline_temperature 0.885 \
        --guard hub:neghani/plantvillage-mobilenetv2

    python -m model.final_test --smoke     # synthetic, under outputs/smoke, no guard

Requires calibration.json and threshold.json (decisions frozen before this runs).
Never used for any choice. Writes <out>/final_report_<revision>.md. The guard record
(<revision>.json) lives in the Hub repo or on Drive, not on the Colab disk (see guard.py).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from eval_real_world_set import norm  # noqa: E402
from stats import (  # noqa: E402
    WIDE_CI_NOTE,
    ece,
    frac,
    frac_ci,
    macro_f1_present,
    model_key,
    paired_bootstrap,
    reliability,
    wilson,
)

# Defaults match the published MobileNet v1.0 / v1.1 pins (shared/release.json history).
DEFAULT_BASELINE_REVISION = "v1.0"
DEFAULT_BASELINE_THRESHOLD = 0.85
DEFAULT_BASELINE_TEMPERATURE = 1.0
# Published MobileNet v1.1 (used when comparing a v1.2 candidate).
V11_BASELINE_THRESHOLD = 0.5
V11_BASELINE_TEMPERATURE = 0.885
# Refuse writing into the shared outputs folder so a v1.2 run cannot overwrite v1.1 reports.
PROTECTED_OUT = (HERE / "outputs").resolve()


def require_frozen(calib: Path, thr: Path) -> tuple[dict, dict]:
    if not calib.exists() or not thr.exists():
        raise SystemExit(
            f"Need {calib} and {thr} before the final test (calibrate.py writes them). "
            "Decisions must be frozen on the validation split first."
        )
    return json.loads(calib.read_text()), json.loads(thr.read_text())


def load_final_rows(test_dir: Path, real_world: Path) -> list[dict]:
    """Rows of final_test.csv (prepare_plantdoc.py already deduped real_world_set against test/)."""
    final_csv = test_dir / "final_test.csv"
    if not final_csv.exists():
        raise SystemExit(f"{final_csv} not found. Run prepare_plantdoc.py first.")
    rows, seen = [], set()
    for r in csv.DictReader(open(final_csv, newline="", encoding="utf-8")):
        root = test_dir if r["source"] == "plantdoc" else real_world
        path = root / r["file"]
        if not path.exists():
            raise SystemExit(f"final-test image missing: {path}")
        if r.get("source_path") and r["source_path"] in seen:
            continue
        seen.add(r.get("source_path") or str(path))
        rows.append({"file": r["file"], "path": path, "label": r["label"],
                     "stratum": r.get("stratum") or ("plantdoc_test" if r["source"] == "plantdoc" else "real_world_set"),
                     "source_path": r.get("source_path", "")})
    return rows


def run_model(rows: list[dict], model: str, revision: str | None, threshold: float,
              temperature: float, device: str | None, strategy: str = "B", crop_mask: bool = False) -> list[dict]:
    from app_predictor import WirePredictor, diagnose_bytes
    from predict import get_predictor

    get_predictor.cache_clear()
    pred = WirePredictor(model, None if Path(model).exists() else revision, device)
    pred.model.temperature = float(temperature)
    pred.model.crop_mask = crop_mask
    out = []
    for i, r in enumerate(rows, 1):
        res = diagnose_bytes(r["path"].read_bytes(), strategy=strategy, threshold=threshold, predictor=pred)
        out.append(res)
        if i % 25 == 0 or i == len(rows):
            print(f"[final] {model}: {i}/{len(rows)}")
    return out


def summarize(results: list[dict], rows: list[dict], threshold: float, label_names: list[str] | None = None) -> dict:
    n = len(rows)
    statuses, preds, confs, correct, reached, top3 = [], [], [], [], [], []
    for res, r in zip(results, rows):
        if res["status"] == "retake_photo":
            statuses.append("retake_photo")
            preds.append("")
            confs.append(0.0)
            correct.append(False)
            reached.append(False)
            top3.append(False)
            continue
        statuses.append(res["status"])
        preds.append(res.get("disease") or "")
        confs.append(float(res.get("confidence") or 0.0))
        ok = norm(res.get("disease")) == norm(r["label"])
        correct.append(ok)
        reached.append(True)
        tk = [norm(t.get("disease")) for t in (res.get("top_k") or [])]
        top3.append(norm(r["label"]) in tk)
    ok = [(s == "ok") for s in statuses]
    passes = [o and c for o, c in zip(ok, correct)]
    reached_correct = [c for c, rh in zip(correct, reached) if rh]
    conf_r = [cf for cf, rh in zip(confs, reached) if rh]
    corr_r = [c for c, rh in zip(correct, reached) if rh]
    pred_reached = [p or "none" for p, rh in zip(preds, reached) if rh]
    true_reached = [r["label"] for r, rh in zip(rows, reached) if rh]
    ok_n = sum(ok)
    ok_right = sum(passes)
    lo, hi = wilson(sum(passes), n)
    return {
        "n": n,
        "passes": sum(passes),
        "pass_rate": sum(passes) / n if n else 0.0,
        "pass_ci": (lo, hi),
        "top1": sum(reached_correct),
        "top1_n": sum(reached),
        "top3": sum(top3),
        "macro_f1": macro_f1_present(true_reached, pred_reached),
        "ok_n": ok_n,
        "ok_wrong": ok_n - ok_right,
        "wrong_confident_rate": (ok_n - ok_right) / ok_n if ok_n else None,
        "ece": ece(conf_r, corr_r) if conf_r else 0.0,
        "gate_n": sum(s == "retake_photo" for s in statuses),
        "not_sure_n": sum(s == "not_sure" for s in statuses),
        "pass_vector": passes,
        "pred": preds,
        "status": statuses,
        "threshold": threshold,
    }


def stratum_table(m: dict, rows: list[dict]) -> dict[str, tuple[int, int]]:
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p, r in zip(m["pass_vector"], rows):
        out[r["stratum"]][0] += int(p)
        out[r["stratum"]][1] += 1
    return {k: (a, b) for k, (a, b) in out.items()}


def stratum_payload(m: dict, rows: list[dict], name: str = "real_world_set") -> dict:
    """Passes, n and Wilson interval for one stratum — what the Hub card before/after row needs."""
    k, n = stratum_table(m, rows).get(name, (0, 0))
    lo, hi = wilson(k, n)
    return {
        "passes": k,
        "n": n,
        "rate": (k / n) if n else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "label": frac_ci(k, n),
    }


def write_final_summary(
    path: Path,
    m_base: dict,
    m_cand: dict,
    rows: list[dict],
    cand_thr: float,
    temperature: float,
    *,
    baseline_name: str = "v1.0",
    candidate_name: str = "v1.1",
    baseline_threshold: float = DEFAULT_BASELINE_THRESHOLD,
    baseline_temperature: float = DEFAULT_BASELINE_TEMPERATURE,
) -> dict:
    summary = {
        "date": datetime.date.today().isoformat(),
        "baseline": baseline_name,
        "candidate": candidate_name,
        baseline_name: {
            "threshold": float(baseline_threshold),
            "temperature": float(baseline_temperature),
            "real_world_set": stratum_payload(m_base, rows),
        },
        candidate_name: {
            "threshold": float(cand_thr),
            "temperature": float(temperature),
            "real_world_set": stratum_payload(m_cand, rows),
        },
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def per_class_md(results: list[dict], rows: list[dict]) -> list[str]:
    by = defaultdict(lambda: [0, 0, 0])  # n, top1, pass
    for res, r in zip(results, rows):
        by[r["label"]][0] += 1
        if res["status"] != "retake_photo" and norm(res.get("disease")) == norm(r["label"]):
            by[r["label"]][1] += 1
            if res["status"] == "ok":
                by[r["label"]][2] += 1
    lines = ["| Class | n | Top-1 | Pass |", "|---|---|---|---|"]
    for lab in sorted(by):
        n, t, p = by[lab]
        lines.append(f"| `{lab}` | {n} | {t}/{n} | {p}/{n} |")
    return lines


def metrics_block(name: str, m: dict, rows: list[dict]) -> list[str]:
    lines = [
        f"### {name}", "",
        f"Threshold {m['threshold']}. " + WIDE_CI_NOTE, "",
        "| Metric | Value |", "|---|---|",
        f"| Pass rate (correct and ok, of all images) | {frac_ci(m['passes'], m['n'])} |",
        f"| Top-1 on images that reached the model | {frac(m['top1'], m['top1_n'])} |",
        f"| Top-3 on images that reached the model | {frac(m['top3'], m['top1_n'])} |",
        f"| Macro F1 (classes present) | {100 * m['macro_f1']:.1f}% |",
        f"| Wrong but confident (of ok answers) | {frac(m['ok_wrong'], m['ok_n'])} |",
        f"| Expected calibration error | {100 * m['ece']:.1f}% |",
        f"| Not sure | {frac(m['not_sure_n'], m['n'])} |",
        f"| Rejected by the gate | {frac(m['gate_n'], m['n'])} |",
    ]
    for stratum, (k, n) in stratum_table(m, rows).items():
        note = " (pristine held-out)" if stratum == "plantdoc_test" else ""
        lines.append(f"| Pass rate, {stratum}{note} | {frac_ci(k, n)} |")
    return lines + [""]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v11", default=None, help="Local run dir of the candidate (fine-tuned) weights.")
    ap.add_argument("--revision_name", default="v1.1", help="Tag the candidate would be published as; names the guard record and report file.")
    ap.add_argument("--guard", default=None,
                    help="hub:<repo_id> (default: hub:$HF_MODEL_ID) or drive:<dir>. Never local outputs/.")
    ap.add_argument("--v10", default="neghani/plantvillage-mobilenetv2",
                    help="Baseline model id or local dir (legacy name; kept for callers).")
    ap.add_argument("--v10_revision", default=None,
                    help="Deprecated alias for --baseline_revision.")
    ap.add_argument("--baseline_revision", default=None,
                    help="Hub revision of the baseline (default: v1.0; use v1.1 when scoring a v1.2 candidate).")
    ap.add_argument("--baseline_threshold", type=float, default=None,
                    help="Confidence threshold for the baseline chain (default: 0.85 for v1.0, 0.5 for v1.1).")
    ap.add_argument("--baseline_temperature", type=float, default=None,
                    help="Softmax temperature for the baseline (default: 1.0 for v1.0, 0.885 for v1.1).")
    ap.add_argument("--test_dir", default=str(HERE / "data" / "plantdoc_test"))
    ap.add_argument("--real_world", default=str(HERE / "data" / "real_world_set"))
    ap.add_argument("--out", default=str(HERE / "outputs"),
                    help="Report directory. Must NOT be model/outputs for a non-v1.1 revision (protects v1.1 artifacts).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--crop_mask", action="store_true", help="Also report crop-masked metrics (diagnostic).")
    ap.add_argument("--smoke", action="store_true", help="synthetic data under outputs/smoke; no guard")
    return ap.parse_args(argv)


def resolve_baseline(a: argparse.Namespace) -> tuple[str, float, float]:
    """Return (revision, threshold, temperature) for the baseline chain."""
    rev = a.baseline_revision or a.v10_revision or DEFAULT_BASELINE_REVISION
    if a.baseline_threshold is not None:
        thr = float(a.baseline_threshold)
    elif rev == "v1.1":
        thr = V11_BASELINE_THRESHOLD
    else:
        thr = DEFAULT_BASELINE_THRESHOLD
    if a.baseline_temperature is not None:
        temp = float(a.baseline_temperature)
    elif rev == "v1.1":
        temp = V11_BASELINE_TEMPERATURE
    else:
        temp = DEFAULT_BASELINE_TEMPERATURE
    return rev, thr, temp


def assert_out_safe(out: Path, revision_name: str, smoke: bool) -> None:
    """Refuse writing a non-v1.1 report into the shared model/outputs folder."""
    if smoke:
        return
    if revision_name == "v1.1":
        return
    if out.resolve() == PROTECTED_OUT:
        raise SystemExit(
            f"--out {out} is the shared model/outputs folder. "
            f"A {revision_name} final test must write elsewhere "
            f"(e.g. .../v1.2/final) so v1.1 reports are not overwritten."
        )


def resolve_guard(a: argparse.Namespace):
    """None in smoke mode (and --guard is refused there); otherwise a Hub or Drive guard."""
    if a.smoke:
        if a.guard:
            raise SystemExit("--smoke never reads or writes the final-test guard; drop --guard.")
        return None
    from common import DEFAULT_HF_MODEL_ID, get_hf_token
    from guard import make_guard

    spec = a.guard or f"hub:{os.environ.get('HF_MODEL_ID', DEFAULT_HF_MODEL_ID)}"
    return make_guard(spec, token=get_hf_token())


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    if a.smoke:
        from smoke import ensure

        p = ensure("final")
        a.v11, a.v10 = str(p.run), str(p.base)
        a.baseline_revision = None
        a.v10_revision = None
        a.test_dir, a.real_world, a.out, a.device = str(p.pd_test), str(p.data / "real_world_set"), str(p.artifacts), "cpu"
    if not a.v11:
        raise SystemExit("--v11 is required (or --smoke)")
    base_rev, base_thr, base_temp = resolve_baseline(a)
    # Local baseline dir (smoke) has no Hub revision.
    baseline_hub_rev = None if Path(a.v10).exists() else base_rev
    cand_name = a.revision_name
    base_name = base_rev or "baseline"
    guard = resolve_guard(a)

    cand = Path(a.v11)
    calib, thr = require_frozen(cand / "calibration.json", cand / "threshold.json")
    key = model_key(str(cand), None)
    out = Path(a.out)
    assert_out_safe(out, cand_name, a.smoke)
    out.mkdir(parents=True, exist_ok=True)
    if guard is not None:
        from guard import check

        if check(guard, cand_name, key, a.force) == "skip":
            return

    rows = load_final_rows(Path(a.test_dir), Path(a.real_world))
    if not rows:
        raise SystemExit("No final-test images found. Run prepare_plantdoc.py first.")
    print(f"[final] {len(rows)} images "
          f"({sum(1 for r in rows if r['stratum']=='plantdoc_test')} plantdoc_test, "
          f"{sum(1 for r in rows if r['stratum']=='real_world_set')} real_world_set)")
    print(f"[final] candidate={cand_name} vs baseline={base_name} "
          f"(thr={base_thr}, T={base_temp})")

    cand_thr = thr.get("threshold")
    if cand_thr is None:
        print(f"[final] threshold is provisional; scoring {cand_name} at the baseline threshold "
              f"{base_thr} so the comparison is defined, without changing the config.")
        cand_thr = base_thr
    base_res = run_model(rows, a.v10, baseline_hub_rev, float(base_thr), float(base_temp), a.device)
    cand_res = run_model(rows, str(cand), None, float(cand_thr), float(calib["temperature"]), a.device)
    m_base = summarize(base_res, rows, float(base_thr))
    m_cand = summarize(cand_res, rows, float(cand_thr))
    gain, lo, hi = paired_bootstrap(m_cand["pass_vector"], m_base["pass_vector"])

    report_stem = f"final_report_{cand_name.replace('.', '_')}"
    reuse_note = ""
    if cand_name == "v1.2":
        reuse_note = (
            " These images were already used once for the v1.1 report; recipe choices for "
            f"{cand_name} used PlantDoc validation only, not this split."
        )
    lines = [
        f"# Final test report ({cand_name} vs {base_name})", "",
        f"Run {datetime.datetime.now().isoformat(timespec='seconds')}. "
        f"Images: {len(rows)}. This split was not used for any choice.{reuse_note}",
        "",
        f"{cand_name} temperature T={calib['temperature']:.3f}; threshold status **{thr['status']}** "
        f"({thr['note']})",
        "",
        *metrics_block(
            f"{base_name} (threshold {base_thr:.2f}, T={base_temp:g})", m_base, rows
        ),
        *metrics_block(
            f"{cand_name} (threshold {cand_thr:.2f}, T={calib['temperature']:.3f})", m_cand, rows
        ),
        f"## Paired difference ({cand_name} minus {base_name} pass rate)", "",
        f"Observed {100 * gain:+.1f} points; 95% bootstrap interval "
        f"{100 * lo:+.1f} to {100 * hi:+.1f} (1000 resamples, seed 0).",
        "",
        f"## Per-class ({cand_name})", "",
        *per_class_md(cand_res, rows),
        "",
    ]
    if m_cand["pass_rate"] < 0.5:
        lines += [
            f"**The final pass rate is {frac(m_cand['passes'], m_cand['n'])}, below 50%.** "
            "Reported as-is. The threshold and the data are not changed after seeing this number.",
            "",
        ]
    if a.crop_mask:
        masked = run_model(rows, str(cand), None, float(cand_thr), float(calib["temperature"]), a.device,
                           crop_mask=True)
        mm = summarize(masked, rows, float(cand_thr))
        lines += metrics_block(f"{cand_name} with crop masking (diagnostic)", mm, rows)

    lines += [
        "The 50-image real_world_set is all from Cropped-PlantDoc `train/` and half of it chose",
        "the v1.0 threshold, so it is not pristine. `plantdoc_test` is the honest held-out stratum.",
        "All images are web-sourced; gains overstate phone-photo performance.",
        "",
        "Per-image results: `final_test_per_image.csv`.",
        "",
    ]
    summary_kwargs = dict(
        baseline_name=base_name,
        candidate_name=cand_name,
        baseline_threshold=float(base_thr),
        baseline_temperature=float(base_temp),
    )
    (out / f"{report_stem}.md").write_text("\n".join(lines))
    summary = write_final_summary(
        out / "final_summary.json", m_base, m_cand, rows, float(cand_thr),
        float(calib["temperature"]), **summary_kwargs,
    )
    write_final_summary(
        cand / "final_summary.json", m_base, m_cand, rows, float(cand_thr),
        float(calib["temperature"]), **summary_kwargs,
    )
    with open(out / "final_test_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "stratum", "true", "base_pred", "base_status",
                                          "base_conf", "cand_pred", "cand_status", "cand_conf"])
        w.writeheader()
        for r, a_base, a_cand in zip(rows, base_res, cand_res):
            w.writerow({
                "file": r["file"], "stratum": r["stratum"], "true": r["label"],
                "base_pred": a_base.get("disease") or "", "base_status": a_base["status"],
                "base_conf": a_base.get("confidence") if a_base.get("confidence") is not None else "",
                "cand_pred": a_cand.get("disease") or "", "cand_status": a_cand["status"],
                "cand_conf": a_cand.get("confidence") if a_cand.get("confidence") is not None else "",
            })
    if guard is not None:
        guard.write(cand_name, {
            "revision": cand_name, "model_key": key, "v11": str(cand), "n": len(rows),
            "passes": m_cand["passes"], "pass_rate": m_cand["pass_rate"],
            "date": datetime.date.today().isoformat(),
            "threshold": cand_thr, "threshold_status": thr["status"],
            "temperature": calib["temperature"],
            "baseline": base_name, "baseline_threshold": base_thr,
            "baseline_temperature": base_temp,
            "code_commit": os.environ.get("CODE_COMMIT") or None,
        })
        print(f"[final] guard written: {guard.describe(cand_name)}")
    print("\n".join(lines[:30]))
    rw = summary[cand_name]["real_world_set"]
    print(f"Wrote {out}/{report_stem}.md, final_summary.json "
          f"(real_world_set {rw['label']})")


if __name__ == "__main__":
    main()
