"""Phase 1: publish the trained model to Hugging Face and tag the release for Phase 5.

Token comes from HF_TOKEN (.env locally, Colab secret in Colab, or `hf auth login`).

Examples:
    python model/publish.py --check                      # verify token, owner and repo access only
    python model/publish.py --release v1.2 --model <stage2 dir> --dry_run   # the v1.2 release, no upload
    python model/publish.py --release v1.2 --model <stage2 dir>             # upload branch + tag v1.2
    python model/publish.py                              # upload model/outputs/plantvillage-resnet18 + tag v1.0
    python model/publish.py --tag v1.1 --private
    python model/publish.py --reports_only               # re-push README.md + reports/ (no weights)

--release sets the branch, tag and messages for a named release. Gated tags need the baseline
PlantDoc-val diagnostics; if they are missing, they are computed once (inference only) into
<model dir>/../artifacts, so no separate notebook step is needed.

Fills README.md from trainer_state.json, reports/metrics.json, final_summary.json,
calibration.json, latency.json and strategy_comparison.md before uploading.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from card import assemble_card  # noqa: E402
from common import DEFAULT_HF_MODEL_REPO, get_hf_token, hf_api  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "outputs" / "plantvillage-resnet18"
V10_PV_ACCURACY = 0.9912
MAX_PV_DROP = 0.02
DIAGNOSTICS = HERE / "outputs" / "diagnostics_v1_0.json"

# Named releases. Everything that used to be typed by hand in each notebook cell lives here.
RELEASES = {
    "v1.2": {
        "hub_model_id": "neghani/plantvillage-mobilenetv2",
        "tag": "v1.2",
        "branch": "v1.2",  # leaves main and the v1.1 tag untouched
        "tag_message": "MobileNetV2 v1.2: PlantVillage then PlantDoc mix",
        "commit_message": "MobileNetV2 v1.2: ImageNet -> PlantVillage -> PlantDoc mix",
    },
}
# Used when no --release is given (the original ResNet-18 Phase 1 defaults).
PLAIN_DEFAULTS = {
    "tag": "v1.0",
    "tag_message": "Phase 1 release: ResNet-18 on PlantVillage",
    "commit_message": "Phase 1: ResNet-18 on PlantVillage",
}

# Trainer scratch files that should never land on the Hub.
IGNORE = ["checkpoint-*", "checkpoint-*/**", "runs/**", "optimizer.pt", "scheduler.pt", "rng_state*.pth", "*.log", ".DS_Store"]
WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "pytorch_model*"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release", choices=sorted(RELEASES), default=None,
                   help="Named release preset: sets --branch, --tag, --tag_message and --commit_message.")
    p.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="Local model directory produced by train.py.")
    p.add_argument("--hub_model_id", default=None,
                   help="Hub repo. Default: the --release repo, else neghani/plantvillage-resnet18.")
    p.add_argument("--tag", default=None, help="Release tag Phase 5 pins (HF_MODEL_REVISION). Use '' to skip tagging.")
    p.add_argument("--tag_message", default=None)
    p.add_argument(
        "--branch",
        default=None,
        help="Hub branch to upload into (default: repo default / main). "
        "Use 'v1.2' (or --release v1.2) so main and the v1.1 tag stay untouched.",
    )
    p.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
    p.add_argument("--reports_only", action="store_true", help="Upload README.md and reports/ only (skip weights).")
    p.add_argument("--commit_message", default=None)
    p.add_argument("--check", action="store_true", help="Verify token / owner / repo access and exit.")
    p.add_argument("--dry_run", action="store_true", help="Show the files that would be uploaded and exit.")
    p.add_argument("--skip_gate", action="store_true", help="Do not enforce the stage-2 publish gate (not for the real release).")
    p.add_argument("--skip_card_check", action="store_true",
                   help="Debug only: publish even if a model-card section is missing. Release presets never set this.")
    p.add_argument(
        "--diagnostics",
        default=None,
        help="Baseline PlantDoc-val diagnostics JSON. Default: <model dir>/../artifacts/diagnostics_<baseline>.json, "
        "computed once if missing. v1.1 gate: v1.0 file. v1.2 gate: v1.1 file (same code).",
    )
    args = p.parse_args(argv)
    return resolve_release(args)


def resolve_release(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unset options from the --release preset, then from the plain Phase 1 defaults."""
    preset = RELEASES[args.release] if args.release else {}
    if args.hub_model_id is None:
        args.hub_model_id = preset.get("hub_model_id", DEFAULT_HF_MODEL_REPO)
    for key in ("tag", "branch", "tag_message", "commit_message"):
        if getattr(args, key) is None:
            setattr(args, key, preset.get(key, PLAIN_DEFAULTS.get(key)))
    if args.diagnostics is None:
        args.diagnostics = str(default_diagnostics_path(Path(args.model), args.tag))
    return args


def default_diagnostics_path(model_dir: Path, tag: str) -> Path:
    """Where the gate baseline lives for a gated tag: next to the model's run folder, in artifacts/."""
    baseline = GATED_TAGS.get(tag)
    if baseline is None:
        return DIAGNOSTICS
    return model_dir.parent / "artifacts" / f"diagnostics_{baseline.replace('.', '_')}.json"


def ensure_diagnostics(path: Path, baseline: str, hub_model_id: str) -> Path:
    """Return the baseline PlantDoc-val diagnostics, computing them once if missing.

    Inference only (never opens the final test). Builds the PlantDoc validation manifest first
    if that is missing too. Results are cached at `path`, so later runs skip this.
    """
    if path.exists():
        return path
    repo = HERE.parent
    manifest = HERE / "data" / "plantdoc_train" / "manifest.csv"
    print(f"{path} missing: computing {baseline} PlantDoc-val diagnostics once (inference only).")
    if not manifest.exists():
        subprocess.run(
            [sys.executable, "-m", "model.prepare_plantdoc",
             "--exclude_set", str(HERE / "data" / "real_world_set"),
             "--artifacts_dir", str(path.parent), "--skip_if_exists"],
            cwd=repo, check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "model.diagnostics",
         "--model", hub_model_id, "--revision", baseline,
         "--manifest", str(manifest), "--extra_dir", str(manifest.parent),
         "--out", str(path.parent)],
        cwd=repo, check=True,
    )
    if not path.exists():
        raise SystemExit(
            f"Diagnostics ran but {path} was not written. diagnostics.py names its output "
            f"diagnostics_{baseline.replace('.', '_')}.json; pass --diagnostics with that path."
        )
    return path


# Tags that must beat a baseline PlantDoc-val F1 and keep PlantVillage accuracy within MAX_PV_DROP.
GATED_TAGS = {
    "v1.1": "v1.0",
    "v1.2": "v1.1",
}


def publish_gate(model_dir: Path, tag: str, diagnostics: Path = DIAGNOSTICS) -> None:
    """Stage-2 tags publish only if PlantDoc val macro-F1 beats the baseline and PV accuracy drops by at most 2 points."""
    baseline = GATED_TAGS.get(tag)
    if baseline is None:
        return
    if not diagnostics.exists():
        raise SystemExit(
            f"{tag} gate: {diagnostics} missing. Run diagnostics.py on the PlantDoc validation split "
            f"for the {baseline} weights first."
        )
    diag = json.loads(diagnostics.read_text())
    base_f1 = float(diag["plantdoc_macro_f1"])
    eval_path = model_dir / "eval_results.json"
    if not eval_path.exists():
        eval_path = model_dir / "all_results.json"
    if not eval_path.exists():
        raise SystemExit(f"{tag} gate: {model_dir}/eval_results.json missing.")
    ev = json.loads(eval_path.read_text())
    cand_f1 = ev.get("eval_plantdoc_macro_f1")
    if cand_f1 is None:
        raise SystemExit(f"{tag} gate: eval_plantdoc_macro_f1 not in eval results.")
    metrics_path = model_dir / "reports" / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"{tag} gate: {metrics_path} missing. Run evaluate.py on the PlantVillage test split.")
    pv_acc = float(json.loads(metrics_path.read_text())["accuracy"])
    problems = []
    if cand_f1 <= base_f1:
        problems.append(
            f"PlantDoc val macro F1 {cand_f1:.4f} is not higher than {baseline} {base_f1:.4f}"
        )
    if pv_acc < V10_PV_ACCURACY - MAX_PV_DROP:
        problems.append(
            f"PlantVillage test accuracy {pv_acc:.4f} dropped more than {MAX_PV_DROP:.0%} from {V10_PV_ACCURACY:.4f}"
        )
    if problems:
        raise SystemExit(f"{tag} publish gate failed:\n  - " + "\n  - ".join(problems))
    print(
        f"Publish gate OK: PlantDoc val F1 {cand_f1:.4f} > {baseline} {base_f1:.4f}; "
        f"PlantVillage acc {pv_acc:.4f} (floor {V10_PV_ACCURACY - MAX_PV_DROP:.4f})"
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


def hub_tag_exists(api, repo_id: str, tag: str) -> bool:
    """True when the Hub model repo already has this git tag."""
    refs = api.list_repo_refs(repo_id, repo_type="model")
    return any(t.name == tag for t in refs.tags)


def refuse_if_tag_exists(api, repo_id: str, tag: str) -> None:
    """v1.2 must be created once. Never overwrite an existing v1.2 tag."""
    if tag != "v1.2":
        return
    if hub_tag_exists(api, repo_id, tag):
        raise SystemExit(
            f"Refuse publish: tag {tag!r} already exists on {repo_id}. "
            f"This run will not overwrite it. Inspect "
            f"https://huggingface.co/{repo_id}/tree/{tag} and choose a new tag if you need another release."
        )


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
    gated = args.tag in GATED_TAGS
    if gated and card_status["missing"] and not args.skip_card_check:
        msg = (f"{args.tag} card check failed; missing sections: "
               + ", ".join(card_status["missing"])
               + ". Run the matching eval scripts or pass --skip_card_check.")
        if not args.dry_run:
            raise SystemExit(msg)
        print(f"[dry run] the real publish would stop here: {msg}")
    if gated and not args.skip_gate:
        diag = ensure_diagnostics(Path(args.diagnostics), GATED_TAGS[args.tag], args.hub_model_id)
        try:
            publish_gate(model_dir, args.tag, diag)
        except SystemExit as exc:
            if not args.dry_run:
                raise
            print(f"[dry run] the real publish would stop here: {exc}")
    if gated and not (model_dir / "calibration.json").exists():
        print(f"Warning: calibration.json missing; {args.tag} should ship the fitted temperature.")
    if args.tag == "v1.2" and not args.branch:
        print("Warning: publishing v1.2 without --branch will rewrite main. Pass --branch v1.2 to leave main/v1.1 alone.")

    ignore = IGNORE + (WEIGHT_PATTERNS if args.reports_only else [])
    allow = ["README.md", "reports/**"] if args.reports_only else None
    revision = args.branch or None

    if args.dry_run:
        if args.tag == "v1.2":
            # Dry-run still needs a token to check whether the tag already exists.
            if get_hf_token():
                dry_api = hf_api()
                if hub_tag_exists(dry_api, args.hub_model_id, args.tag):
                    print(
                        f"[dry run] the real publish would stop here: tag {args.tag!r} "
                        f"already exists on {args.hub_model_id}"
                    )
                else:
                    print(f"[dry run] tag {args.tag!r} is free on {args.hub_model_id}")
            else:
                print("[dry run] skipped tag-exists check (no HF token)")
        files = sorted(p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*") if p.is_file())
        import fnmatch

        def skipped(rel: str) -> bool:
            return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat) for pat in ignore)

        def wanted(rel: str) -> bool:
            return allow is None or any(fnmatch.fnmatch(rel, pat) for pat in allow)

        for rel in files:
            if wanted(rel) and not skipped(rel):
                print("  would upload", rel)
        print(
            f"Dry run: repo {args.hub_model_id}, branch {args.branch or '(default)'}, "
            f"tag {args.tag or '(none)'}"
        )
        return

    refuse_if_tag_exists(api, args.hub_model_id, args.tag)

    api.create_repo(args.hub_model_id, repo_type="model", private=args.private, exist_ok=True)
    if args.branch:
        api.create_branch(args.hub_model_id, branch=args.branch, repo_type="model", exist_ok=True)
        print(f"Branch ready: {args.branch}")
    commit = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.hub_model_id,
        repo_type="model",
        revision=revision,
        allow_patterns=allow,
        ignore_patterns=ignore,
        commit_message=args.commit_message,
    )
    print("Uploaded:", getattr(commit, "commit_url", commit))

    if args.tag:
        # exist_ok=False for v1.2 so a race cannot silently retarget the tag.
        api.create_tag(
            args.hub_model_id,
            tag=args.tag,
            tag_message=args.tag_message,
            revision=revision,
            exist_ok=False if args.tag == "v1.2" else True,
        )
        print(f"Tagged {args.tag}" + (f" at branch {args.branch}" if args.branch else ""))

    files = api.list_repo_files(args.hub_model_id, revision=args.tag or revision)
    print("Files on the Hub:", ", ".join(sorted(files)))
    print(f"Done: https://huggingface.co/{args.hub_model_id}" + (f"/tree/{args.tag}" if args.tag else ""))
    print(f"Phase 5 pin: shared/release.json (currently HF_MODEL_REVISION). "
          f"If {args.tag or 'main'} should be the live API default, bump that file "
          f"and copy it to hf_space/release.json.")


if __name__ == "__main__":
    main()
