"""Build the shared-label field set for the three-model bake-off.

Keeps only PlantDoc / real_world rows whose disease exists in both PlantVillage
(38) and MasterDataset (86). PlantDoc test/ is held out for PlantVillage stage-2;
MasterDataset never trains on PlantDoc. Any path listed in the MasterDataset
plantdoc_ood exclusion list is dropped.

    python model/build_shared_set.py
    python model/build_shared_set.py --out model/data/shared_set
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from plantdoc_map import MAP, MASTER_ALIASES, SKIP_CLASS  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_OUT = HERE / "data" / "shared_set"
DEFAULT_PLANTDOC = HERE / "data" / "plantdoc_src"
DEFAULT_REAL_WORLD = HERE / "data" / "real_world_set"
DEFAULT_OOD = HERE / "outputs" / "masterdataset-resnet18" / "plantdoc_ood.json"


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def shared_pairs() -> list[tuple[str, str, str]]:
    """(plantdoc_class, plantvillage_label, master_label) present in MASTER_ALIASES."""
    rows = []
    for plantdoc_name, pv in MAP.items():
        if plantdoc_name == SKIP_CLASS:
            continue
        master = MASTER_ALIASES.get(pv)
        if master:
            rows.append((plantdoc_name, pv, master))
    return rows


def excluded_basenames(ood_path: Path) -> set[str]:
    if not ood_path.is_file():
        return set()
    payload = json.loads(ood_path.read_text())
    names = set()
    for item in payload.get("excluded_plantdoc") or []:
        names.add(Path(item).name)
    for hit in payload.get("phash_hits_before_exclusion") or []:
        names.add(Path(hit["plantdoc"]).name)
    return names


def collect_plantdoc_test(
    plantdoc_root: Path,
    pairs: list[tuple[str, str, str]],
    ban: set[str],
) -> list[dict]:
    by_pd = {p: (pv, m) for p, pv, m in pairs}
    rows = []
    test = plantdoc_root / "test"
    for plantdoc_name, (pv, master) in sorted(by_pd.items()):
        folder = test / plantdoc_name
        if not folder.is_dir():
            continue
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name not in ban
        )
        # Stable half-split when a class has 2+ images; single image goes to final.
        for i, src in enumerate(files):
            split = "final" if (len(files) == 1 or i % 2 == 1) else "tune"
            rel = f"images/plantdoc/{_safe_name(plantdoc_name)}/{src.name}"
            rows.append({
                "file": rel,
                "label": pv,
                "label_pv": pv,
                "label_master": master,
                "plantdoc_class": plantdoc_name,
                "source": "plantdoc_test",
                "split": split,
                "source_path": str(src.resolve()),
                "_src": src,
            })
    return rows


def collect_real_world(
    rw_root: Path,
    pairs: list[tuple[str, str, str]],
) -> list[dict]:
    pv_to_master = {pv: m for _, pv, m in pairs}
    labels_path = rw_root / "labels.csv"
    if not labels_path.is_file():
        return []
    rows = []
    for r in csv.DictReader(open(labels_path, newline="", encoding="utf-8")):
        pv = r["label"]
        if pv not in pv_to_master:
            continue
        src = rw_root / r["file"]
        if not src.is_file():
            raise SystemExit(f"real_world image missing: {src}")
        name = Path(r["file"]).name
        rel = f"images/real_world/{_safe_name(pv)}/{name}"
        rows.append({
            "file": rel,
            "label": pv,
            "label_pv": pv,
            "label_master": pv_to_master[pv],
            "plantdoc_class": r.get("plantdoc_class", ""),
            "source": "real_world_set",
            "split": r["split"],
            "source_path": str(src.resolve()),
            "_src": src,
        })
    return rows


def link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)


def build(out: Path, plantdoc: Path, real_world: Path, ood: Path) -> dict:
    pairs = shared_pairs()
    ban = excluded_basenames(ood)
    rows = collect_plantdoc_test(plantdoc, pairs, ban) + collect_real_world(real_world, pairs)
    if not rows:
        raise SystemExit("No shared-label rows found. Is PlantDoc test/ present?")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for r in rows:
        link_or_copy(r["_src"], out / r["file"])

    fieldnames = [
        "file", "label", "label_pv", "label_master", "plantdoc_class",
        "source", "split", "source_path",
    ]
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    n_tune = sum(1 for r in rows if r["split"] == "tune")
    n_final = sum(1 for r in rows if r["split"] == "final")
    manifest = {
        "n": len(rows),
        "n_tune": n_tune,
        "n_final": n_final,
        "n_shared_classes": len(pairs),
        "shared_classes": [
            {"plantdoc": p, "plantvillage": pv, "master": m} for p, pv, m in pairs
        ],
        "sources": {
            "plantdoc_test": sum(1 for r in rows if r["source"] == "plantdoc_test"),
            "real_world_set": sum(1 for r in rows if r["source"] == "real_world_set"),
        },
        "excluded_basenames": sorted(ban),
        "note": (
            "Shared-label field set for bake-off. label/label_pv are PlantVillage ids; "
            "label_master is the MasterDataset id. Score each model against its own column."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Shared-label field set\n\n"
        f"{len(rows)} images, {len(pairs)} classes present in PlantVillage and MasterDataset.\n"
        f"tune={n_tune}, final={n_final}. Built by `model/build_shared_set.py`.\n"
        "Decision metric: final-split top-1 / end-to-end pass on this set only.\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--plantdoc", default=str(DEFAULT_PLANTDOC))
    p.add_argument("--real_world", default=str(DEFAULT_REAL_WORLD))
    p.add_argument("--ood", default=str(DEFAULT_OOD))
    args = p.parse_args(argv)
    manifest = build(Path(args.out), Path(args.plantdoc), Path(args.real_world), Path(args.ood))
    print(
        f"Wrote {manifest['n']} images "
        f"(tune={manifest['n_tune']}, final={manifest['n_final']}, "
        f"classes={manifest['n_shared_classes']}) -> {args.out}",
        flush=True,
    )
    if manifest["n_final"] < 100:
        print(
            f"[warn] final n={manifest['n_final']} < 100; Wilson CIs will be wide "
            "(stats.WIDE_N). Prefer n_final >= 100 before pinning.",
            flush=True,
        )


if __name__ == "__main__":
    main()
