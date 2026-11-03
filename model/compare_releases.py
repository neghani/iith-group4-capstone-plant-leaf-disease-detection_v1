"""Side-by-side field eval: ResNet candidate vs ResNet v1.1 vs MobileNet v1.1.

Uses the live-app chain (strategy B) on plantdoc_test + real_world_set when available,
otherwise on real_world_set alone. Also scores the strawberry OOD photo when present.

Examples:
    python -m model.compare_releases --candidate model/outputs/stage2-resnet18/lr1e-4
    python -m model.compare_releases --candidate neghani/plantvillage-resnet18 --candidate_revision v1.2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from final_test import load_final_rows, run_model, summarize  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", required=True, help="Local dir or Hub id for the new model.")
    p.add_argument("--candidate_revision", default=None)
    p.add_argument("--candidate_threshold", type=float, default=None)
    p.add_argument("--candidate_temperature", type=float, default=None)
    p.add_argument("--resnet_v11", default="neghani/plantvillage-resnet18")
    p.add_argument("--resnet_v11_revision", default="v1.1")
    p.add_argument("--mobilenet_v11", default="neghani/plantvillage-mobilenetv2")
    p.add_argument("--mobilenet_v11_revision", default="v1.1")
    p.add_argument("--test_dir", default=str(HERE / "data" / "plantdoc_test"))
    p.add_argument("--real_world", default=str(HERE / "data" / "real_world_set"))
    p.add_argument("--strawberry", default=str(HERE / "outputs" / "ood" / "negatives" / "strawberry.jpg"))
    p.add_argument("--out", default=str(HERE / "outputs" / "compare_v1_2.json"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_calib(model: str, revision: str | None) -> tuple[float, float]:
    """Return (threshold, temperature). Local dirs win; Hub falls back to release pin defaults."""
    local = Path(model)
    if local.is_dir():
        thr_path = local / "threshold.json"
        cal_path = local / "calibration.json"
        thr = json.loads(thr_path.read_text())["threshold"] if thr_path.exists() else 0.55
        if thr is None:
            thr = 0.55
        temp = json.loads(cal_path.read_text())["temperature"] if cal_path.exists() else 1.0
        return float(thr), float(temp)
    # Hub: prefer published calibration.json next to the tag.
    try:
        from huggingface_hub import hf_hub_download

        cal = json.loads(Path(hf_hub_download(model, "calibration.json", revision=revision)).read_text())
        temp = float(cal.get("temperature") or 1.0)
    except Exception:
        temp = 0.9357514486544162 if "resnet" in model else 0.885
    thr = 0.55 if "resnet" in model else 0.5
    try:
        pin = json.loads((HERE.parent / "shared" / "release.json").read_text())
        if pin.get("hf_model_id") == model and pin.get("hf_model_revision") == revision:
            thr = float(pin.get("confidence_threshold") or thr)
            temp = float(pin.get("calibration_temperature") or temp)
        cand = pin.get("candidate") or {}
        if cand.get("hf_model_id") == model and cand.get("hf_model_revision") == revision:
            thr = float(cand.get("confidence_threshold") or thr)
            temp = float(cand.get("calibration_temperature") or temp)
    except Exception:
        pass
    return float(thr), float(temp)


def score_one(name: str, model: str, revision: str | None, rows: list[dict],
              threshold: float | None, temperature: float | None, device: str | None) -> dict:
    thr, temp = load_calib(model, revision)
    if threshold is not None:
        thr = float(threshold)
    if temperature is not None:
        temp = float(temperature)
    print(f"[compare] {name}: {model}@{revision or 'local'} thr={thr} T={temp}")
    results = run_model(rows, model, None if Path(model).exists() else revision, thr, temp, device)
    metrics = summarize(results, rows, thr)
    return {
        "name": name,
        "model": model,
        "revision": revision,
        "threshold": thr,
        "temperature": temp,
        "passes": metrics["passes"],
        "n": metrics["n"],
        "pass_rate": metrics["pass_rate"],
        "top1": metrics["top1"],
        "top1_n": metrics["top1_n"],
        "not_sure_n": metrics["not_sure_n"],
        "gate_n": metrics["gate_n"],
        "ece": metrics["ece"],
    }


def strawberry_check(path: Path, model: str, revision: str | None, threshold: float, temperature: float,
                     device: str | None) -> dict | None:
    if not path.exists():
        return None
    from app_predictor import WirePredictor, diagnose_bytes
    from predict import get_predictor

    get_predictor.cache_clear()
    pred = WirePredictor(model, None if Path(model).exists() else revision, device)
    pred.model.temperature = float(temperature)
    res = diagnose_bytes(path.read_bytes(), strategy="B", threshold=threshold, predictor=pred)
    return {
        "status": res.get("status"),
        "disease": res.get("disease"),
        "confidence": res.get("confidence"),
        "ok_expected": res.get("status") in ("not_sure", "retake_photo"),
    }


def main() -> None:
    a = parse_args()
    test_dir, real_world = Path(a.test_dir), Path(a.real_world)
    if (test_dir / "final_test.csv").exists() and real_world.exists():
        rows = load_final_rows(test_dir, real_world)
    elif (real_world / "labels.csv").exists():
        rows = []
        for r in csv.DictReader(open(real_world / "labels.csv", newline="", encoding="utf-8")):
            path = real_world / r["file"]
            if path.exists():
                rows.append({"file": r["file"], "path": path, "label": r["label"],
                             "stratum": "real_world_set", "source_path": str(path)})
    else:
        raise SystemExit(f"Need {test_dir}/final_test.csv + {real_world} or a labelled real_world_set.")

    models = [
        ("candidate", a.candidate, a.candidate_revision, a.candidate_threshold, a.candidate_temperature),
        ("resnet_v1.1", a.resnet_v11, a.resnet_v11_revision, None, None),
        ("mobilenet_v1.1", a.mobilenet_v11, a.mobilenet_v11_revision, None, None),
    ]
    scored = [score_one(n, m, rev, rows, thr, temp, a.device) for n, m, rev, thr, temp in models]

    berry = {}
    berry_path = Path(a.strawberry)
    for row in scored:
        berry[row["name"]] = strawberry_check(
            berry_path, row["model"], row["revision"], row["threshold"], row["temperature"], a.device
        )

    out = {
        "n_images": len(rows),
        "models": scored,
        "strawberry": berry,
        "note": "Same images for every candidate. The 284-image set was already used for v1.1 decisions.",
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
