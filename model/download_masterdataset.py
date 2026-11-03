"""Download the Plant Leaf Disease MasterDataset and write an inventory.

Uses the ``kaggle`` package when KAGGLE_USERNAME and KAGGLE_KEY are set. Otherwise
falls back to ``kagglehub``, which can fetch this public dataset without a token.

    python model/download_masterdataset.py
    python model/download_masterdataset.py --inventory-only --path data/masterdataset

The inventory is written to ``data/inventory.json``. Training stops later if the
train, val, and test class sets differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from common import MASTER_KAGGLE_DATASET, crop_from_class, load_dotenv  # noqa: E402
from dataset import class_files, resolve_split_root  # noqa: E402

DEFAULT_DEST = REPO / "data" / "masterdataset"
INVENTORY_PATH = REPO / "data" / "inventory.json"
SIZE_SAMPLE = 8


def _kaggle_download(dest: Path) -> Path:
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(MASTER_KAGGLE_DATASET, path=str(dest), unzip=True, quiet=False)
    return dest


def _kagglehub_download() -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(MASTER_KAGGLE_DATASET))


def _link(root: Path, dest: Path) -> Path:
    """Point data/masterdataset at the extracted tree without copying gigabytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.resolve() == root.resolve():
            return dest
        return dest
    try:
        dest.symlink_to(root, target_is_directory=True)
        return dest
    except OSError:
        return root


def download(dest: Path = DEFAULT_DEST) -> Path:
    load_dotenv()
    try:
        root = _kaggle_download(dest)
        source = "kaggle"
    except Exception as exc:
        print(f"kaggle package download unavailable ({exc}). Using kagglehub.", flush=True)
        root = _kagglehub_download()
        source = "kagglehub"
        root = _link(root, dest)
    print(f"Dataset ({source}) at {root}", flush=True)
    return root


def _sample_sizes(files_by_class: dict[str, list[Path]], n: int) -> dict:
    from PIL import Image

    counts: Counter[str] = Counter()
    checked = 0
    for paths in files_by_class.values():
        for path in paths[:n]:
            try:
                with Image.open(path) as im:
                    counts[f"{im.size[0]}x{im.size[1]}"] += 1
            except Exception:
                counts["unreadable"] += 1
            checked += 1
    return {"per_class": n, "checked": checked, "sizes": dict(counts)}


def build_inventory(root: Path, size_sample: int = SIZE_SAMPLE) -> dict:
    split_dirs = resolve_split_root(root)
    dataset_root = split_dirs["train"].parent
    listed = {key: class_files(path) for key, path in split_dirs.items()}
    names = {key: set(files) for key, files in listed.items()}
    reference = next(iter(names.values()))
    mismatches = {key: sorted(have.symmetric_difference(reference)) for key, have in names.items() if have != reference}
    empty = sorted({name for files in listed.values() for name, paths in files.items() if not paths})
    splits = {}
    for key, path in split_dirs.items():
        per_class = {name: len(paths) for name, paths in listed[key].items()}
        splits[key] = {
            "dir": path.name,
            "n_images": sum(per_class.values()),
            "n_classes": len(per_class),
            "per_class": per_class,
        }
    payload = {
        "dataset": MASTER_KAGGLE_DATASET,
        "root": str(dataset_root),
        "n_classes": len(reference),
        "n_crops": len({crop_from_class(name) for name in reference}),
        "crops": sorted({crop_from_class(name) for name in reference}),
        "classes": sorted(reference),
        "class_sets_identical": not mismatches and not empty,
        "mismatches": mismatches,
        "empty_classes": empty,
        "splits": splits,
        "n_images": sum(item["n_images"] for item in splits.values()),
        "size_sample": _sample_sizes(listed["train"], size_sample),
    }
    return payload


def write_inventory(payload: dict, path: Path = INVENTORY_PATH) -> Path:
    if not payload["class_sets_identical"]:
        detail = payload["mismatches"] or payload["empty_classes"]
        raise SystemExit(f"MasterDataset inventory failed: {detail}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Inventory: {payload['n_classes']} classes, {payload['n_images']} images, "
        f"splits { {k: v['n_images'] for k, v in payload['splits'].items()} }",
        flush=True,
    )
    print(f"Wrote {path}", flush=True)
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", default=str(DEFAULT_DEST), help="Directory for the unzipped dataset.")
    p.add_argument("--path", default=None, help="Existing dataset root. Skips the download.")
    p.add_argument("--inventory-only", action="store_true", help="Do not download; inventory --path or --dest.")
    p.add_argument("--inventory", default=str(INVENTORY_PATH))
    p.add_argument("--size-sample", type=int, default=SIZE_SAMPLE)
    args = p.parse_args(argv)
    if args.path:
        root = Path(args.path)
    elif args.inventory_only and Path(args.dest).exists():
        root = Path(args.dest)
    else:
        root = download(Path(args.dest))
    payload = build_inventory(root, size_sample=args.size_sample)
    write_inventory(payload, Path(args.inventory))


if __name__ == "__main__":
    main()
