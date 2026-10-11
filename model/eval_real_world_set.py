"""Evaluate the model on the labelled real-world set and write report + model-card section.

Run the SAME chain the live app uses (shared preprocessing + classifier) by pointing
--predictor at a function `fn(PIL.Image) -> dict`. The dict should look like the
output of model/predict.py::diagnose:
    {"status": "ok" | "not_sure" | "retake_photo",
     "disease": "<label>", "confidence": 0.93,
     "top_k": [{"disease": "<label>", "confidence": 0.93}, ...]}
Key names are matched loosely (disease/label/class). Labels are compared after
lower-casing and stripping punctuation, so "Tomato___Early_blight" == "Tomato Early blight".

Examples
  python eval_real_world_set.py --set_dir real_world_set --split tune  --predictor predict:diagnose
  python eval_real_world_set.py --set_dir real_world_set --split final --predictor predict:diagnose --threshold 0.5
  python eval_real_world_set.py --set_dir real_world_set --selftest        # no model needed
  python model/eval_real_world_set.py --set_dir model/data/real_world_set --split final \
      --predictor app_predictor:diagnose --threshold 0.5 --push           # live-app chain, update Hub card
  python model/eval_real_world_set.py ... --model model/outputs/plantvillage-mobilenetv2 \
      --push --only_if_changed                                            # Colab: publish only on change
Report the final split ONCE, after the threshold and gate are fixed on the tune split.
--push uploads reports/real_world_set/<split>/ and the set metadata (labels, sources, README,
build script; not the images) to the Hub model repo. Only --split final rewrites the model card.
Each pushed split carries run.json with a fingerprint of the set (labels.csv + every image),
the model weights and the threshold; --only_if_changed skips the run when all three match the Hub.
"""
import argparse, csv, datetime, hashlib, importlib, json, math, os, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(x):
    return f"{100 * x:.1f}%"


def _label_of(item):
    if isinstance(item, dict):
        for k in ("disease", "label", "class", "name"):
            if k in item:
                return item[k]
        return None
    if isinstance(item, (list, tuple)) and item:
        return item[0]
    return item if isinstance(item, str) else None


def top1(res):
    return _label_of(res)


def topk(res):
    for k in ("top_k", "topk", "top3", "predictions"):
        if k in res and res[k]:
            return [_label_of(i) for i in res[k]]
    return [top1(res)]


def _stub_predictor(labels, seed=0):
    rng = random.Random(seed)

    def fn(img):
        r = rng.random()
        if r < 0.10:
            return {"status": "retake_photo", "message": "blurry"}
        true_guess = rng.choice(labels)
        conf = rng.uniform(0.3, 0.99)
        return {"status": "ok" if conf >= 0.5 else "not_sure", "disease": true_guess,
                "confidence": conf, "top_k": [{"disease": true_guess}, {"disease": rng.choice(labels)}]}
    return fn


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def set_fingerprint(root, rows):
    """Changes when labels.csv or any listed image changes (content, not mtime)."""
    h = hashlib.sha256(_sha256(root / "labels.csv").encode())
    for r in sorted(rows, key=lambda r: r["file"]):
        h.update(f"{r['file']}\0{_sha256(root / r['file'])}\n".encode())
    return h.hexdigest()


def model_fingerprint(model, revision, token):
    """sha256 of model.safetensors, identical for a local run dir and the same weights on the Hub."""
    local = Path(model) / "model.safetensors"
    if local.exists():
        return _sha256(local)
    from huggingface_hub import HfApi
    info = HfApi(token=token).get_paths_info(model, ["model.safetensors"], revision=revision)
    return info[0].lfs.sha256 if info and info[0].lfs else f"{model}@{revision}"


def hub_run_info(repo, split, token):
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo, f"reports/real_world_set/{split}/run.json", token=token, force_download=True)
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def hub_card_has(repo, token, needles):
    """False when the card was overwritten since (evaluate.py / publish.py upload a fresh template)."""
    from huggingface_hub import hf_hub_download
    try:
        text = Path(hf_hub_download(repo, "README.md", token=token, force_download=True)).read_text()
    except Exception:
        return False
    return all(n in text for n in needles)


def source_summary(root):
    path = root / "sources.csv"
    if not path.exists():
        return "a labelled real-world set"
    counts = Counter(r.get("source") or "unknown" for r in csv.DictReader(open(path)))
    return ", ".join(f"{name} ({c})" if len(counts) > 1 else name for name, c in counts.most_common())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_dir", required=True)
    ap.add_argument("--split", default="all", choices=["all", "tune", "final"])
    ap.add_argument("--predictor", default="predict:diagnose")
    ap.add_argument("--threshold", type=float, default=None, help="only for the report text")
    ap.add_argument("--strategy", default=None, choices=list("ABCDE"),
                    help="crop strategy for app_predictor (shared/vision/strategies.py)")
    ap.add_argument("--tta", action="store_true", help="app_predictor: horizontal-flip TTA")
    ap.add_argument("--target", type=float, default=0.5, help="coordinator target for end-to-end pass rate")
    ap.add_argument("--out", default="outputs/real_world")
    ap.add_argument("--card_template", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--push", action="store_true", help="upload results to the Hub model repo")
    ap.add_argument("--only_if_changed", action="store_true",
                    help="with --push: skip when set, weights and threshold match the Hub's run.json")
    ap.add_argument("--hub_model_id", default=os.environ.get("HF_MODEL_ID", "neghani/plantvillage-mobilenetv2"))
    ap.add_argument("--model", default=None, help="local run dir or Hub id for the predictor (default: HF_MODEL_ID)")
    ap.add_argument("--revision", default=None, help="Hub revision for --model (default: HF_MODEL_REVISION)")
    a = ap.parse_args()

    # Picked up by app_predictor, so the threshold in the report is the one actually applied.
    if a.model:
        os.environ["EVAL_MODEL"] = a.model
    if a.revision:
        os.environ["EVAL_REVISION"] = a.revision
    if a.threshold is not None:
        os.environ["CONFIDENCE_THRESHOLD"] = str(a.threshold)
    if a.strategy:
        os.environ["CROP_STRATEGY"] = a.strategy
    if a.tta:
        os.environ["CROP_TTA"] = "1"
    strategy_label = (a.strategy or os.environ.get("CROP_STRATEGY", "B")) + (" + flip TTA" if a.tta else "")

    root = Path(a.set_dir)
    rows = list(csv.DictReader(open(root / "labels.csv")))
    if a.split != "all":
        rows = [r for r in rows if r["split"] == a.split]
    missing = [r["file"] for r in rows if not (root / r["file"]).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} image(s) listed in {root}/labels.csv are missing "
            f"(images/ is git-ignored). e.g. {missing[0]}\n"
            "On Colab, copy the zip from Drive:\n"
            "  python notebooks/colab_sync.py --real-world-set-from "
            "/content/drive/MyDrive/plant-disease/real_world_set.zip\n"
            "Or rebuild the same 50 images (seed 42) from Cropped-PlantDoc:\n"
            "  python model/build_real_world_set.py model/data/plantdoc_src model/data/real_world_set"
        )
    labels_all = sorted({r["label"] for r in rows})

    run = None
    if a.push and not a.selftest:
        from common import get_hf_token
        token = get_hf_token()
        model = a.model or os.environ.get("HF_MODEL_ID", a.hub_model_id)
        revision = a.revision or os.environ.get("HF_MODEL_REVISION")
        run = {
            "split": a.split,
            "set_fingerprint": set_fingerprint(root, rows),
            "model": model,
            "revision": None if Path(model).exists() else revision,
            "model_fingerprint": model_fingerprint(model, revision, token),
            "threshold": a.threshold,
            "predictor": a.predictor,
            "strategy": strategy_label,
        }
        keys = ("set_fingerprint", "model_fingerprint", "threshold", "predictor", "strategy")
        prev = hub_run_info(a.hub_model_id, a.split, token)
        card_ok = a.split != "final" or hub_card_has(
            a.hub_model_id, token, [run["set_fingerprint"][:12], run["model_fingerprint"][:12]])
        if a.only_if_changed and prev and card_ok and all(prev.get(k) == run[k] for k in keys):
            print(f"[{a.split}] unchanged (set {run['set_fingerprint'][:12]}, weights "
                  f"{run['model_fingerprint'][:12]}, threshold {a.threshold}); nothing to publish.")
            if a.split == "final" and Path(model).is_dir():
                from huggingface_hub import hf_hub_download
                card = hf_hub_download(a.hub_model_id, "README.md", token=token, force_download=True)
                (Path(model) / "README.md").write_text(Path(card).read_text())  # keep publish.py in sync
            return
        changed = [k for k in keys if not prev or prev.get(k) != run[k]] + ([] if card_ok else ["model card"])
        print(f"[{a.split}] publishing; changed vs Hub: {', '.join(changed) or 'forced'}")

    if a.selftest:
        fn = _stub_predictor(labels_all)
    else:
        mod, func = a.predictor.split(":")
        fn = getattr(importlib.import_module(mod), func)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in rows:
        if getattr(fn, "accepts_bytes", False):
            res = fn((root / r["file"]).read_bytes()) or {}
        else:
            res = fn(ImageOps.exif_transpose(Image.open(root / r["file"])).convert("RGB")) or {}
        status = res.get("status", "ok")
        pred = top1(res) if status != "retake_photo" else None
        tk = topk(res) if status != "retake_photo" else []
        t = norm(r["label"])
        recs.append({
            "file": r["file"], "split": r["split"], "true": r["label"], "pred": pred or "",
            "status": status, "confidence": res.get("confidence", ""),
            "reached_model": status != "retake_photo",
            "top1_correct": bool(pred) and norm(pred) == t,
            "top3_correct": any(norm(x) == t for x in tk if x),
        })
        recs[-1]["pass"] = recs[-1]["top1_correct"] and status == "ok"

    n = len(recs)
    gate = sum(r["status"] == "retake_photo" for r in recs)
    ns = sum(r["status"] == "not_sure" for r in recs)
    reached = [r for r in recs if r["reached_model"]]
    k1 = sum(r["top1_correct"] for r in reached)
    k3 = sum(r["top3_correct"] for r in reached)
    kp = sum(r["pass"] for r in recs)
    ok_n = sum(r["status"] == "ok" for r in recs)
    ok_correct = sum(r["pass"] for r in recs)
    lo, hi = wilson(kp, n)
    lo1, hi1 = wilson(k1, len(reached))

    per = defaultdict(lambda: [0, 0])
    conf = Counter()
    for r in reached:
        per[r["true"]][0] += 1
        per[r["true"]][1] += r["top1_correct"]
        if not r["top1_correct"]:
            conf[(r["true"], r["pred"])] += 1

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)

    meets = "meets" if kp / n >= a.target else "is below"
    lines = [
        f"# Real-world evaluation ({a.split} split)", "",
        f"- Images: {n} across {len(labels_all)} classes",
        f"- Rejected by quality gate (retake photo): {gate}",
        f"- Reached the model: {len(reached)}",
        f"- Model top-1 on images that reached it: {k1}/{len(reached)} = {pct(k1 / max(1, len(reached)))} "
        f"(95% CI {pct(lo1)} to {pct(hi1)})",
        f"- Model top-3 on images that reached it: {k3}/{len(reached)} = {pct(k3 / max(1, len(reached)))}",
        f"- Answered confidently (status ok): {ok_n}; of those correct: {ok_correct}"
        f" = {pct(ok_correct / max(1, ok_n))}",
        f"- 'Not sure' answers: {ns}",
        f"- **End-to-end pass rate (correct and confident, out of all images): {kp}/{n} = {pct(kp / n)} "
        f"(95% CI {pct(lo)} to {pct(hi)})**",
        f"- Target {pct(a.target)}: result {meets} the target (point estimate).", "",
        "## Per class (images that reached the model)", "", "| Class | Images | Correct |", "|---|---|---|",
    ]
    lines += [f"| {c} | {v[0]} | {v[1]} |" for c, v in sorted(per.items())]
    mistakes = ["| True | Predicted | Count |", "|---|---|---|"]
    mistakes += [f"| `{t}` | `{p}` | {c} |" for (t, p), c in conf.most_common(8)]
    lines += ["", "## Most common mistakes", ""] + mistakes
    (out / "report.md").write_text("\n".join(lines) + "\n")

    tpl_path = Path(a.card_template) if a.card_template else Path(__file__).with_name("model_card_real_world_section.md")
    if tpl_path.exists():
        txt = tpl_path.read_text()
        rep = {
            "{{n}}": str(n), "{{n_classes}}": str(len(labels_all)), "{{split}}": a.split,
            "{{gate_rejected}}": str(gate), "{{not_sure}}": str(ns),
            "{{top1_model}}": pct(k1 / max(1, len(reached))), "{{top1_ci}}": f"{pct(lo1)} to {pct(hi1)}",
            "{{top3_model}}": pct(k3 / max(1, len(reached))),
            "{{e2e_pass}}": pct(kp / n), "{{e2e_ci}}": f"{pct(lo)} to {pct(hi)}",
            "{{target}}": pct(a.target), "{{target_status}}": meets,
            "{{threshold}}": str(a.threshold) if a.threshold is not None else "see API config",
            "{{strategy}}": strategy_label,
            "{{date}}": datetime.date.today().isoformat(),
            "{{mistakes_table}}": "\n".join(mistakes),
            "{{source_summary}}": source_summary(root),
            "{{set_notes}}": (root / "card_notes.md").read_text().strip() if (root / "card_notes.md").exists() else "",
            "{{fingerprint}}": (f"Set fingerprint `{run['set_fingerprint'][:12]}` · model weights "
                                f"`{run['model_fingerprint'][:12]}`") if run else "",
        }
        for k, v in rep.items():
            txt = txt.replace(k, v)
        (out / "model_card_section.md").write_text(txt)

    print("\n".join(lines[:12]))
    print(f"\nWrote {out}/results.csv, report.md" + (", model_card_section.md" if tpl_path.exists() else ""))

    if run:
        run.update(date=datetime.date.today().isoformat(), n=n, end_to_end_pass_rate=round(kp / n, 4),
                   top1_reached=round(k1 / max(1, len(reached)), 4))
        (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        push(a, root, out, n, kp, k1, len(reached))


def push(a, root, out, n, kp, k1, n_reached):
    from huggingface_hub import CommitOperationAdd, EvalResult, HfApi, ModelCard
    from common import get_hf_token

    token = get_hf_token()
    api = HfApi(token=token)
    repo = a.hub_model_id
    ops = [CommitOperationAdd(f"reports/real_world_set/{a.split}/{name}", str(out / name))
           for name in ("results.csv", "report.md", "run.json")]
    ops += [CommitOperationAdd(f"real_world_set/{name}", str(root / name))
            for name in ("labels.csv", "sources.csv", "README.md", "card_notes.md") if (root / name).exists()]
    build = Path(__file__).with_name("build_real_world_set.py")
    if build.exists():
        ops.append(CommitOperationAdd(f"real_world_set/{build.name}", str(build)))
    api.create_commit(repo_id=repo, operations=ops,
                      commit_message=f"Real-world set: {a.split} split results ({n} images)")

    section = out / "model_card_section.md"
    if a.split != "final" or not section.exists():
        print(f"Pushed {a.split} results to https://huggingface.co/{repo} (card unchanged)")
        return

    card = ModelCard.load(repo, token=token)
    start, end = "<!-- REAL_WORLD_START -->", "<!-- REAL_WORLD_END -->"
    block = section.read_text().strip()
    if start in card.text:
        card.text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, card.text, flags=re.S)
    else:
        card.text = card.text.rstrip() + "\n\n" + block + "\n"

    dataset_type = "real-world-set"
    dataset_name = f"Real-world set: {source_summary(root)} ({a.split} split, {n} images)"
    kept = [r for r in (card.data.eval_results or []) if r.dataset_type == "plantvillage"]
    model_name = card.data.model_name or repo.split("/")[-1]
    card.data.eval_results = kept + [
        EvalResult(task_type="image-classification", task_name="Image Classification",
                   dataset_type=dataset_type, dataset_name=dataset_name,
                   metric_type="accuracy", metric_name="Top-1 accuracy (reached model)",
                   metric_value=round(k1 / max(1, n_reached), 4)),
        EvalResult(task_type="image-classification", task_name="Image Classification",
                   dataset_type=dataset_type, dataset_name=dataset_name,
                   metric_type="accuracy", metric_name="End-to-end pass rate",
                   metric_value=round(kp / n, 4)),
    ]
    card.data.model_name = model_name
    card.push_to_hub(repo, token=token, commit_message=f"Model card: real-world results ({n} images)")
    if a.model and Path(a.model).is_dir():
        card.save(Path(a.model) / "README.md")  # publish.py uploads this file later
        print(f"Model card also written to {Path(a.model) / 'README.md'}")
    print(f"Pushed results and model card to https://huggingface.co/{repo}")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
