"""Phase 1: publish the trained model to Hugging Face and tag the release for Phase 5.

Token comes from HF_TOKEN (.env locally, Colab secret in Colab, or `hf auth login`).

Examples:
    python model/publish.py --check                      # verify token, owner and repo access only
    python model/publish.py --dry_run                    # fill the card, list what would be uploaded
    python model/publish.py                              # upload model/outputs/plantvillage-mobilenetv2 + tag v1.0
    python model/publish.py --tag v1.1 --private
    python model/publish.py --reports_only               # re-push README.md + reports/ (no weights)

Fills README.md from trainer_state.json, reports/metrics.json, final_summary.json,
calibration.json, latency.json and strategy_comparison.md before uploading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from card import assemble_card  # noqa: E402
from common import DEFAULT_HF_MODEL_REPO, get_hf_token, hf_api  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "outputs" / "masterdataset-resnet18"
# Floor from the corrected serving-path test score (n=14454). Update when the pin moves.
MASTER_TEST_ACCURACY = 0.9614639546146395
MAX_ACCURACY_DROP = 0.02
DIAGNOSTICS = HERE / "outputs" / "diagnostics_v1_0.json"

# Trainer scratch files that should never land on the Hub.
IGNORE = ["checkpoint-*", "checkpoint-*/**", "runs/**", "optimizer.pt", "scheduler.pt", "rng_state*.pth", "*.log", ".DS_Store"]
WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "pytorch_model*"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="Local model directory produced by train.py.")
    p.add_argument("--hub_model_id", default=DEFAULT_HF_MODEL_REPO)
    p.add_argument("--tag", default="v1.0", help="Release tag Phase 5 pins (HF_MODEL_REVISION). Use '' to skip tagging.")
    p.add_argument("--tag_message", default="Phase 1 release: ResNet-18 on PlantVillage")
    p.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
    p.add_argument("--reports_only", action="store_true", help="Upload README.md and reports/ only (skip weights).")
    p.add_argument("--commit_message", default="Phase 1: ResNet-18 on PlantVillage")
    p.add_argument("--check", action="store_true", help="Verify token / owner / repo access and exit.")
    p.add_argument("--dry_run", action="store_true", help="Show the files that would be uploaded and exit.")
    p.add_argument("--skip_gate", action="store_true", help="Do not enforce the v1.1 publish gate (not for the real release).")
    p.add_argument("--skip_card_check", action="store_true",
                   help="Do not refuse v1.1 when a model-card section is missing.")
    p.add_argument("--diagnostics", default=str(DIAGNOSTICS), help="v1.0 validation diagnostics (diagnostics.py).")
    return p.parse_args()


def masterdataset_publish_gate(model_dir: Path) -> None:
    """Require a one-shot test score, PlantDoc OOD with zero pHash hits, and the accuracy floor."""
    problems = []
    test_path = model_dir / "test_results.json"
    info_path = model_dir / "run_info.json"
    ood_path = model_dir / "plantdoc_ood.json"
    n_test = None
    accuracy = None
    if not test_path.is_file():
        problems.append(f"{test_path.name} missing (score the held-out test split once)")
    else:
        test = json.loads(test_path.read_text())
        accuracy = test.get("test_accuracy")
        if accuracy is None:
            problems.append("test_results.json has no test_accuracy")
        elif float(accuracy) < MASTER_TEST_ACCURACY - MAX_ACCURACY_DROP:
            problems.append(
                f"test accuracy {float(accuracy):.4f} dropped more than {MAX_ACCURACY_DROP:.0%} "
                f"from MasterDataset baseline {MASTER_TEST_ACCURACY:.4f}"
            )
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        n_test = (info.get("split_counts") or {}).get("test")
        if info.get("dataset") not in (None, "masterdataset"):
            problems.append(f"run_info dataset is {info.get('dataset')!r}, expected masterdataset")
    else:
        problems.append("run_info.json missing")
    if not n_test:
        problems.append("run_info.split_counts.test missing (sample count must be reported with the accuracy)")
    if not ood_path.is_file():
        problems.append(f"{ood_path.name} missing (PlantDoc overlap eval, out of training)")
    else:
        ood = json.loads(ood_path.read_text())
        if "n_images" not in ood or "n_overlap_classes" not in ood:
            problems.append("plantdoc_ood.json must include n_images and n_overlap_classes")
        elif ood.get("trained_on_plantdoc"):
            problems.append("plantdoc_ood.json says the model was trained on PlantDoc")
        elif not ood.get("ok", False):
            problems.append("plantdoc_ood.json ok is false (pHash hits remain in the evaluated OOD set)")
    if problems:
        raise SystemExit("MasterDataset publish gate failed:\n  - " + "\n  - ".join(problems))
    print(
        f"MasterDataset publish gate OK: test accuracy {float(accuracy):.4f} on {n_test} images "
        f"(floor {MASTER_TEST_ACCURACY - MAX_ACCURACY_DROP:.4f}); PlantDoc overlap ok."
    )


def publish_gate(model_dir: Path, tag: str, diagnostics: Path = DIAGNOSTICS) -> None:
    """MasterDataset tags use masterdataset_publish_gate. PlantVillage publishing is retired."""
    del diagnostics  # PlantVillage diagnostics path is unused after the baseline swap.
    info_path = model_dir / "run_info.json"
    dataset = None
    if info_path.is_file():
        try:
            dataset = json.loads(info_path.read_text()).get("dataset")
        except json.JSONDecodeError:
            dataset = None
    if dataset == "masterdataset":
        masterdataset_publish_gate(model_dir)
        return
    raise SystemExit(
        "PlantVillage publishing is retired. The live training set is the MasterDataset; "
        "rollback still reads the existing Hub tag without re-publishing."
    )


def verify_access(api, repo_id: str) -> str:
    """Return the account name; raise with an actionable message if the token cannot push to repo_id."""
    try:
        who = api.whoami()
    except Exception as exc:
        raise SystemExit(f"Token rejected by Hugging Face ({exc}). Create a new WRITE token at https://huggingface.co/settings/tokens") from exc

    name = who["name"]
    owner = repo_id.split("/")[0]
    orgs = [o["name"] for o in who.get("orgs", [])]
    if owner != name and owner not in orgs:
        raise SystemExit(
            f"Logged in as '{name}' but HF_MODEL_ID is '{repo_id}'. Log in as '{owner}' or set HF_MODEL_ID={name}/{repo_id.split('/')[-1]}."
        )

    auth = who.get("auth", {}).get("accessToken", {})
    role = auth.get("role")
    if role in ("read",):
        raise SystemExit("This token is read-only. Create a WRITE token (or fine-grained with 'write' on this repo).")
    print(f"Token OK: user '{name}', role '{role or 'unknown'}', pushing to {repo_id}")
    return name


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        if not get_hf_token():
            raise SystemExit("No HF token found. Put HF_TOKEN=... in .env, add the Colab secret HF_TOKEN, or run `hf auth login`.")
        api = hf_api()
        verify_access(api, args.hub_model_id)
        if args.check:
            return

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        raise SystemExit(f"{model_dir} not found. Train first (model/train.py) or pass --model <dir>.")
    if not args.reports_only and not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("*.bin")):
        raise SystemExit(f"No weights in {model_dir}. Did train.py finish? (expected model.safetensors)")
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"{model_dir}/config.json is missing; this is not a saved model directory.")
    if not (model_dir / "reports" / "metrics.json").exists():
        print("Warning: reports/metrics.json missing. Run model/evaluate.py first so the Hub gets metrics and the model card.")
    card_status = assemble_card(model_dir)
    print(f"Model card sections filled: {', '.join(card_status['filled']) or '(none)'}")
    if card_status["missing"]:
        print(f"Model card sections missing: {', '.join(card_status['missing'])}")
    if not (model_dir / "README.md").exists():
        print("Warning: README.md (model card) missing. evaluate.py writes it.")
    if args.tag == "v1.1" and card_status["missing"] and not args.skip_card_check:
        msg = ("v1.1 card check failed; missing sections: "
               + ", ".join(card_status["missing"])
               + ". Run the matching eval scripts or pass --skip_card_check.")
        if not args.dry_run:
            raise SystemExit(msg)
        print(f"[dry run] the real publish would stop here: {msg}")
    if args.tag == "v1.1" and not args.skip_gate:
        try:
            publish_gate(model_dir, args.tag, Path(args.diagnostics))
        except SystemExit as exc:
            if not args.dry_run:
                raise
            print(f"[dry run] the real publish would stop here: {exc}")
    if args.tag == "v1.1" and not (model_dir / "calibration.json").exists():
        print("Warning: calibration.json missing; v1.1 should ship the fitted temperature.")

    ignore = IGNORE + (WEIGHT_PATTERNS if args.reports_only else [])
    allow = ["README.md", "reports/**"] if args.reports_only else None

    if args.dry_run:
        files = sorted(p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*") if p.is_file())
        import fnmatch

        def skipped(rel: str) -> bool:
            return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat) for pat in ignore)

        def wanted(rel: str) -> bool:
            return allow is None or any(fnmatch.fnmatch(rel, pat) for pat in allow)

        for rel in files:
            if wanted(rel) and not skipped(rel):
                print("  would upload", rel)
        print(f"Dry run: repo {args.hub_model_id}, tag {args.tag or '(none)'}")
        return

    api.create_repo(args.hub_model_id, repo_type="model", private=args.private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.hub_model_id,
        repo_type="model",
        allow_patterns=allow,
        ignore_patterns=ignore,
        commit_message=args.commit_message,
    )
    print("Uploaded:", getattr(commit, "commit_url", commit))

    if args.tag:
        api.create_tag(args.hub_model_id, tag=args.tag, tag_message=args.tag_message, exist_ok=True)
        print(f"Tagged {args.tag}")

    files = api.list_repo_files(args.hub_model_id, revision=args.tag or None)
    print("Files on the Hub:", ", ".join(sorted(files)))
    print(f"Done: https://huggingface.co/{args.hub_model_id}" + (f"/tree/{args.tag}" if args.tag else ""))
    print(f"Phase 5 pin: shared/release.json (currently HF_MODEL_REVISION). "
          f"If {args.tag or 'main'} should be the live API default, bump that file "
          f"and copy it to hf_space/release.json.")


if __name__ == "__main__":
    main()
