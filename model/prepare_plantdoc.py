"""Prepare Cropped-PlantDoc for stage-2 fine-tuning (Singh et al. 2020, CC BY 4.0).

    python model/prepare_plantdoc.py                      # clones PlantDoc-Dataset if it is not there yet

Source is the pinned Cropped-PlantDoc repo (class folders under train/ and test/), commit
5467f6012d78d1c446145d5f582da6096f852ae8. Training images are rendered with crop strategy B
so they match what inference sends the model.

Exclusions from PlantDoc train (never enter training or validation):
  1. Every image whose original path appears in real_world_set/sources.csv.
  2. Any image that is a pHash near-duplicate (Hamming <= 6) of PlantDoc test/ or of the 50-image set.
  3. The class "Tomato two spotted spider mites leaf".

Remaining train images are clustered by pHash, then ~12% of clusters become validation.
Final test = PlantDoc test/ plus the 50-image set, deduped by original path.

Assertions run at the end and again at the start of every training run (data_protocol.py).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data_protocol import (  # noqa: E402
    MATCH_DIST,
    PINNED_COMMIT,
    REPO_URL,
    VAL_PCT,
    Exclusion,
    Item,
    assert_protocol,
    first_near_dup,
    group_by_hash,
    in_validation,
    load_labelled_images,
    norm_path,
    phash,
    read_original_paths,
    write_manifest,
)
from plantdoc_map import MAP, SKIP_CLASS  # noqa: E402
from shared.vision import config as C  # noqa: E402

LICENSE = "CC BY 4.0 (dataset repository)"
CROP_JPEG_QUALITY = C.INFERENCE_JPEG_QUALITY


def ensure_repo(src: Path, url: str = REPO_URL, clone: bool = True) -> tuple[Path, str]:
    """Return (src, commit). Shallow-clone if train/ is missing. Warn if commit != pin."""
    if not (src / "train").is_dir():
        if not clone:
            raise SystemExit(f"PlantDoc not found at {src} (expected train/); drop --no_clone to fetch it.")
        if src.exists() and any(src.iterdir()):
            raise SystemExit(f"{src} exists but has no train/; remove it or pass --src elsewhere.")
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"[clone] {url} -> {src}", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(src)], check=True)
        if not (src / "train").is_dir():
            raise SystemExit(f"Cloned {url} but train/ is missing; the repository layout changed.")
    commit = repo_commit(src)
    if commit != PINNED_COMMIT:
        print(f"[warn] PlantDoc commit {commit} differs from the pinned {PINNED_COMMIT}. "
              "Splits will not match a pinned run.", flush=True)
    else:
        print(f"[pin] PlantDoc commit {commit} matches the pin.", flush=True)
    return src, commit


def repo_commit(src: Path) -> str:
    """HEAD of the PlantDoc clone; '' if src is not itself a git checkout (e.g. a folder in this repo)."""
    top = subprocess.run(["git", "-C", str(src), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != src.resolve():
        return ""
    r = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
    return (r.stdout or "").strip()


def scan_split(src: Path, split: str) -> list[tuple[str, Path, str]]:
    """(relative path from repo root, abs path, PlantDoc class) for images under src/split/."""
    root = src / split
    if not root.is_dir():
        return []
    out = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                out.append((f"{split}/{cls_dir.name}/{p.name}", p, cls_dir.name))
    return out


def strategy_b_crop(img: Image.Image) -> Image.Image:
    """The 256x256 crop the live app sends for strategy B (best leaf by leaf_score)."""
    from shared.vision.candidates import find_candidates

    cands = find_candidates(img)
    return cands[0].crop


def short_id(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:10]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def prepare(src: Path, out_train: Path, out_test: Path, exclude_set: Path, val_pct: int = VAL_PCT,
            commit: str = "", artifacts_dir: Path | None = None) -> dict:
    train_files = scan_split(src, "train")
    test_files = scan_split(src, "test")
    if not train_files:
        raise SystemExit(f"No images under {src}/train")

    print(f"[scan] hashing {len(train_files)} train + {len(test_files)} test images", flush=True)
    train_hash: dict[str, int] = {}
    test_hash: dict[str, int] = {}
    train_meta: dict[str, tuple[Path, str]] = {}
    test_meta: dict[str, tuple[Path, str]] = {}
    for rel, path, cls in train_files:
        try:
            train_hash[rel] = phash(Image.open(path).convert("RGB"))
            train_meta[rel] = (path, cls)
        except Exception:
            print(f"[skip] unreadable {rel}")
    for rel, path, cls in test_files:
        try:
            test_hash[rel] = phash(Image.open(path).convert("RGB"))
            test_meta[rel] = (path, cls)
        except Exception:
            print(f"[skip] unreadable {rel}")

    blocked_paths = set(read_original_paths(exclude_set / "sources.csv").values())
    eval_hash: dict[str, int] = {}
    eval_label: dict[str, str] = {}
    for rel_file, path, label in load_labelled_images(exclude_set):
        key = f"real_world_set/{rel_file}"
        eval_hash[key] = phash(Image.open(path).convert("RGB"))
        eval_label[key] = label
    print(f"[exclude] {exclude_set.name}: {len(eval_hash)} images, {len(blocked_paths)} original_path values")

    exclusions: list[Exclusion] = []
    kept: dict[str, int] = {}
    for rel, h in train_hash.items():
        path, cls = train_meta[rel]
        if cls == SKIP_CLASS:
            exclusions.append(Exclusion(rel, "spider_mites_class", SKIP_CLASS))
            print(f"[exclude] {rel}: class {SKIP_CLASS}")
            continue
        if cls not in MAP:
            exclusions.append(Exclusion(rel, "class_not_in_plantvillage", cls))
            print(f"[exclude] {rel}: unmapped class {cls}")
            continue
        if norm_path(rel) in blocked_paths:
            exclusions.append(Exclusion(rel, "path_in_real_world_set", rel))
            print(f"[exclude] {rel}: original_path is in real_world_set/sources.csv")
            continue
        hit = first_near_dup(h, test_hash)
        if hit:
            exclusions.append(Exclusion(rel, "near_dup_of_plantdoc_test", f"{hit[0]} dist={hit[1]}"))
            print(f"[exclude] {rel}: pHash near-dup of test {hit[0]} (dist {hit[1]})")
            continue
        hit = first_near_dup(h, eval_hash)
        if hit:
            exclusions.append(Exclusion(rel, "near_dup_of_real_world_set", f"{hit[0]} dist={hit[1]}"))
            print(f"[exclude] {rel}: pHash near-dup of {hit[0]} (dist {hit[1]})")
            continue
        kept[rel] = h

    groups = group_by_hash(kept)
    for d in (out_train, out_test):
        shutil.rmtree(d / "images", ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_train / "originals", ignore_errors=True)
    originals: dict[str, str] = {}

    train_items: list[Item] = []
    val_items: list[Item] = []
    for rel, h in sorted(kept.items()):
        path, cls = train_meta[rel]
        label = MAP[cls]
        gid = groups[rel]
        split = "validation" if in_validation(gid, val_pct) else "train"
        dest = f"images/{label}/pd_{short_id(rel)}.jpg"
        (out_train / dest).parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as im:
            strategy_b_crop(im.convert("RGB")).save(out_train / dest, format="JPEG", quality=CROP_JPEG_QUALITY)
        if split == "validation":
            # The threshold sweep runs the live chain (gate + crop) on the uncropped photo.
            orig = f"originals/{label}/pd_{short_id(rel)}{path.suffix.lower()}"
            (out_train / orig).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out_train / orig)
            originals[dest] = orig
        item = Item(path=rel, split=split, label=label, phash=h, group=gid, source="plantdoc",
                    plantdoc_class=cls, file=dest)
        (val_items if split == "validation" else train_items).append(item)

    test_items: list[Item] = []
    used_paths = set()
    for rel, h in sorted(test_hash.items()):
        path, cls = test_meta[rel]
        if cls == SKIP_CLASS or cls not in MAP:
            continue
        label = MAP[cls]
        dest = f"images/{label}/pd_{short_id(rel)}.jpg"
        (out_test / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_test / dest)
        gid = f"test:{rel}"
        test_items.append(Item(path=rel, split="holdout", label=label, phash=h, group=gid,
                               source="plantdoc", plantdoc_class=cls, stratum="plantdoc_test", file=dest))
        used_paths.add(norm_path(rel))

    eval_paths = read_original_paths(exclude_set / "sources.csv")
    rw_items: list[Item] = []
    for rel_file, path, label in load_labelled_images(exclude_set):
        orig = eval_paths.get(rel_file, "")
        if orig and orig in used_paths:
            continue
        key = f"real_world_set/{rel_file}"
        rw_items.append(Item(path=orig or key, split="holdout", label=label, phash=eval_hash[key],
                             group=f"rw:{rel_file}", source="real_world_set",
                             stratum="real_world_set", file=rel_file))
        if orig:
            used_paths.add(orig)

    final_items = test_items + rw_items
    assert_protocol(train_items, val_items, final_items)

    write_csv(out_train / "manifest.csv",
              ["file", "original_file", "label", "plantdoc_class", "split", "source_path", "group", "phash"],
              [{"file": x.file, "original_file": originals.get(x.file, ""), "label": x.label,
                "plantdoc_class": x.plantdoc_class, "split": x.split,
                "source_path": x.path, "group": x.group, "phash": f"{x.phash:016x}"}
               for x in train_items + val_items])
    write_csv(out_test / "labels.csv",
              ["file", "label", "plantdoc_class", "split", "stratum", "source_path", "group", "notes"],
              [{"file": x.file, "label": x.label, "plantdoc_class": x.plantdoc_class, "split": "holdout",
                "stratum": x.stratum, "source_path": x.path, "group": x.group, "notes": ""}
               for x in test_items])
    write_csv(out_test / "sources.csv",
              ["file", "source", "original_path", "original_class", "license", "url"],
              [{"file": x.file, "source": "Cropped-PlantDoc TEST", "original_path": x.path,
                "original_class": x.plantdoc_class, "license": LICENSE, "url": REPO_URL}
               for x in test_items])
    write_csv(out_test / "final_test.csv",
              ["file", "label", "split", "stratum", "source", "source_path", "group"],
              [{"file": x.file, "label": x.label, "split": "holdout", "stratum": x.stratum,
                "source": x.source, "source_path": x.path, "group": x.group}
               for x in final_items])
    write_csv(out_train / "excluded.csv", ["path", "reason", "detail"],
              [{"path": e.path, "reason": e.reason, "detail": e.detail} for e in exclusions])

    # Persist items with hashes so train.py can re-assert without re-opening every JPEG.
    payload_items = {
        "train": [x.as_row() for x in train_items],
        "validation": [x.as_row() for x in val_items],
        "final_test": [x.as_row() for x in final_items],
    }
    (out_train / "items.json").write_text(json.dumps(payload_items) + "\n")

    manifest_path = (artifacts_dir or HERE / "outputs") / "data_manifest.json"
    summary = write_manifest(
        manifest_path,
        commit=commit or repo_commit(src),
        pinned=PINNED_COMMIT,
        exclusions=exclusions,
        groups={
            "plantdoc_train": train_items,
            "plantdoc_validation": val_items,
            "plantdoc_test": test_items,
            "real_world_set": rw_items,
            "final_test": final_items,
        },
        params={"phash_dist": MATCH_DIST, "val_pct": val_pct, "strategy": "B"},
        extra={"src": str(src), "exclude_set": str(exclude_set)},
    )
    (out_train / "prep_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_train / "prep_report.md").write_text(report_md(summary))
    return summary


def report_md(s: dict) -> str:
    c = s["counts"]
    lines = [
        "# PlantDoc preparation report (Cropped-PlantDoc)", "",
        f"Commit `{s['plantdoc_commit']}` "
        + ("matches the pin." if s["commit_matches_pin"] else f"(pinned `{s['plantdoc_commit_pinned']}`)."),
        "",
        "| Output | Images |", "|---|---|",
        f"| plantdoc train | {c.get('plantdoc_train', 0)} |",
        f"| plantdoc validation | {c.get('plantdoc_validation', 0)} |",
        f"| plantdoc test (held out) | {c.get('plantdoc_test', 0)} |",
        f"| real_world_set in final test | {c.get('real_world_set', 0)} |",
        f"| final test (deduped) | {c.get('final_test', 0)} |",
        "",
        "## Exclusions from train", "",
        "| Reason | Images |", "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in sorted(s.get("exclusions", {}).items())] or ["| (none) | 0 |"]
    lines += ["", "## Images per class", "",
              "| Class | train | validation | test | real_world_set |", "|---|---|---|---|---|"]
    per = s.get("per_class", {})
    for lab, row in per.items():
        lines.append(
            f"| {lab} | {row.get('plantdoc_train', 0)} | {row.get('plantdoc_validation', 0)} | "
            f"{row.get('plantdoc_test', 0)} | {row.get('real_world_set', 0)} |"
        )
    lines += [
        "",
        "The other 11 PlantVillage classes are absent from PlantDoc and keep lab-only training.",
        "Training and validation crops are rendered with strategy B (256×256).",
        "The 50-image real_world_set is all from Cropped-PlantDoc `train/`; it is excluded from",
        "training. The honest held-out stratum is `plantdoc_test`.",
        "",
    ]
    return "\n".join(lines)


def items_from_json(path: Path) -> dict[str, list[Item]]:
    raw = json.loads(path.read_text())
    out: dict[str, list[Item]] = {}
    for name, rows in raw.items():
        items = []
        for r in rows:
            h = int(r["phash"], 16) if isinstance(r["phash"], str) else int(r["phash"])
            items.append(Item(path=r["path"], split=r["split"], label=r["label"], phash=h,
                              group=r["group"], source=r["source"],
                              plantdoc_class=r.get("plantdoc_class", ""),
                              stratum=r.get("stratum", ""), file=r.get("file", "")))
        out[name] = items
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "data" / "plantdoc_src"), help="Cropped-PlantDoc checkout")
    ap.add_argument("--repo_url", default=REPO_URL)
    ap.add_argument("--no_clone", action="store_true")
    ap.add_argument("--out_train", default=str(HERE / "data" / "plantdoc_train"))
    ap.add_argument("--out_test", default=str(HERE / "data" / "plantdoc_test"))
    ap.add_argument("--exclude_set", default=str(HERE / "data" / "real_world_set"))
    ap.add_argument("--artifacts_dir", default=str(HERE / "outputs"), help="where data_manifest.json goes")
    ap.add_argument("--val_pct", type=int, default=VAL_PCT)
    ap.add_argument("--skip_if_exists", action="store_true",
                    help="Do nothing if items.json is already there (do not rebuild crops).")
    a = ap.parse_args(argv)

    if a.skip_if_exists and (Path(a.out_train) / "items.json").exists():
        print(f"Skipping prepare; {a.out_train}/items.json already exists. "
              "Drop --skip_if_exists to rebuild.")
        return

    src, commit = ensure_repo(Path(a.src), a.repo_url, clone=not a.no_clone)
    s = prepare(src, Path(a.out_train), Path(a.out_test), Path(a.exclude_set), a.val_pct, commit,
                artifacts_dir=Path(a.artifacts_dir))
    c = s["counts"]
    print(f"\ntrain {c.get('plantdoc_train', 0)}  validation {c.get('plantdoc_validation', 0)}  "
          f"plantdoc_test {c.get('plantdoc_test', 0)}  final_test {c.get('final_test', 0)}")
    print(f"excluded: {s.get('exclusions')}")
    print(f"Wrote {a.out_train}/manifest.csv, {a.out_test}/labels.csv, {a.artifacts_dir}/data_manifest.json")


if __name__ == "__main__":
    main()
