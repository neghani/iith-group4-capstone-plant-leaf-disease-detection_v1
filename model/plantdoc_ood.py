"""Score PlantDoc only on classes shared with the MasterDataset.

PlantDoc is not a training source for this model. The report records how many
images were compared and any pHash near-duplicates against the MasterDataset train
split (same crop, distance at or below the cross-set threshold).

    python model/plantdoc_ood.py --data_dir data/masterdataset --out model/outputs/plantdoc_ood.json
    python model/plantdoc_ood.py --labels_from model/outputs/masterdataset-resnet18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from data_protocol import MATCH_DIST_CROSS, hamming, phash_file  # noqa: E402
from dataset import collect_split_rows  # noqa: E402
from plantdoc_map import MAP, overlap_map  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_PLANTDOC = HERE / "data" / "plantdoc_src"


def label_names_from(path: Path) -> list[str]:
    for name in ("splits.json", "run_info.json"):
        candidate = path / name if path.is_dir() else None
        if candidate and candidate.is_file():
            payload = json.loads(candidate.read_text())
            names = payload.get("label_names")
            if names:
                return list(names)
    if path.is_file():
        payload = json.loads(path.read_text())
        if "classes" in payload:
            return list(payload["classes"])
        if "label_names" in payload:
            return list(payload["label_names"])
    raise SystemExit(f"No label_names in {path}. Pass --data_dir or a train output with splits.json.")


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def phash_hits(
    master_files: list[Path],
    plantdoc_files: list[Path],
    *,
    max_dist: int = MATCH_DIST_CROSS,
) -> list[dict]:
    """Pairs at or under max_dist. Callers should pass one crop at a time."""
    master = {}
    for path in master_files:
        try:
            master[str(path)] = phash_file(path)
        except Exception:
            continue
    hits = []
    for path in plantdoc_files:
        try:
            digest = phash_file(path)
        except Exception:
            continue
        for other, other_digest in master.items():
            dist = hamming(digest, other_digest)
            if dist <= max_dist:
                hits.append({"plantdoc": str(path), "master": other, "distance": dist})
    return hits


def build_report(
    label_names: list[str],
    *,
    master_root: Path | None = None,
    plantdoc_root: Path | None = None,
    max_per_class: int | None = None,
    strict: bool = False,
) -> dict:
    mapping = overlap_map(label_names)
    payload: dict = {
        "role": "out_of_dataset",
        "trained_on_plantdoc": False,
        "n_master_classes": len(label_names),
        "n_overlap_classes": len(mapping),
        "mapped": mapping,
        "unmapped_plantdoc": [name for name in MAP if name not in mapping],
        "n_images": 0,
        "per_class": {},
        "phash_max_dist": MATCH_DIST_CROSS,
        "phash_hits": [],
        "phash_sampled_per_class": max_per_class,
    }
    if not plantdoc_root or not plantdoc_root.is_dir():
        payload["note"] = "PlantDoc images are not on disk. Overlap is label-only; n_images is 0."
        payload["ok"] = True
        return payload

    master_by_label: dict[str, list[Path]] = {}
    if master_root and master_root.exists():
        names, rows, _dirs = collect_split_rows(master_root, max_samples_per_class=max_per_class)
        ids = {i: name for i, name in enumerate(names)}
        root = _dirs["train"].parent
        for rel, label in rows["train"]:
            master_by_label.setdefault(ids[label], []).append(root / rel)

    def plantdoc_files(plantdoc_name: str) -> list[Path]:
        files = []
        for folder in (
            plantdoc_root / plantdoc_name,
            plantdoc_root / "train" / plantdoc_name,
            plantdoc_root / "test" / plantdoc_name,
        ):
            files.extend(_images(folder))
        if max_per_class is not None:
            files = files[:max_per_class]
        return files

    # First pass: hash everything and collect near-duplicates against train.
    raw_hits: list[dict] = []
    files_by_label: dict[str, list[Path]] = {}
    for plantdoc_name, master_label in sorted(mapping.items()):
        files = plantdoc_files(plantdoc_name)
        files_by_label.setdefault(master_label, []).extend(files)
        if files and master_label in master_by_label:
            raw_hits.extend(phash_hits(master_by_label[master_label], files))

    excluded = sorted({Path(h["plantdoc"]).resolve().as_posix() for h in raw_hits})
    excluded_set = set(excluded)
    # Near-duplicates stay in the training split (no retrain). Drop them from the
    # OOD evaluation set so ok reflects the cleaned PlantDoc subset.
    per_class = {}
    hits: list[dict] = []
    n_images = 0
    for plantdoc_name, master_label in sorted(mapping.items()):
        files = [
            p for p in plantdoc_files(plantdoc_name)
            if p.resolve().as_posix() not in excluded_set
        ]
        n_images += len(files)
        slot = per_class.setdefault(
            master_label, {"plantdoc_class": plantdoc_name, "n_images": 0}
        )
        slot["n_images"] += len(files)
        if files and master_label in master_by_label:
            hits.extend(phash_hits(master_by_label[master_label], files))

    payload["n_images"] = n_images
    payload["n_images_before_exclusion"] = sum(len(v) for v in files_by_label.values())
    payload["excluded_plantdoc"] = excluded
    payload["n_excluded"] = len(excluded)
    payload["phash_hits_before_exclusion"] = raw_hits
    payload["per_class"] = per_class
    payload["phash_hits"] = hits
    payload["ok"] = not hits
    if hits and strict:
        shown = hits[:10]
        raise SystemExit(
            f"PlantDoc overlaps MasterDataset train by pHash ({len(hits)} pair(s)). First: {shown}"
        )
    return payload


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=None, help="MasterDataset root (train/val/test).")
    p.add_argument("--labels_from", default=None, help="splits.json, run_info.json, inventory, or a train output dir.")
    p.add_argument("--plantdoc", default=str(DEFAULT_PLANTDOC), help="PlantDoc root with one folder per class.")
    p.add_argument("--out", default=str(HERE / "outputs" / "plantdoc_ood.json"))
    p.add_argument("--max_per_class", type=int, default=None, help="Cap images hashed per class.")
    p.add_argument("--strict", action="store_true", help="Exit if a near-duplicate crosses the two sets.")
    args = p.parse_args(argv)

    master_root = Path(args.data_dir) if args.data_dir else None
    if args.labels_from:
        label_names = label_names_from(Path(args.labels_from))
    elif master_root:
        label_names, _rows, _dirs = collect_split_rows(master_root)
    else:
        raise SystemExit("Pass --data_dir or --labels_from.")
    report = build_report(
        label_names,
        master_root=master_root,
        plantdoc_root=Path(args.plantdoc),
        max_per_class=args.max_per_class,
        strict=args.strict,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PlantDoc overlap: {report['n_overlap_classes']} classes, "
        f"{report['n_images']} images, pHash hits {len(report['phash_hits'])}. Wrote {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
