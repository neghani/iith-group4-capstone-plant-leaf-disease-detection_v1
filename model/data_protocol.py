"""Data protocol for stage-2: pHash, clustering, exclusions, leakage assertions, manifest.

Used by prepare_plantdoc.py (build the splits) and train.py (fail loudly before any training).
No torch. Tests run in the API venv.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct

PINNED_COMMIT = "5467f6012d78d1c446145d5f582da6096f852ae8"
REPO_URL = "https://github.com/pratikkayal/PlantDoc-Dataset"
MATCH_DIST = 6  # pHash Hamming distance at or below which two images are near-duplicates
VAL_PCT = 12
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------------------------
# Paths and hashing
# ---------------------------------------------------------------------------------------------

def norm_path(p: str) -> str:
    """'./train\\\\Apple Scab Leaf\\\\foo.JPG' -> 'train/Apple Scab Leaf/foo.JPG'."""
    return str(p).replace("\\", "/").lstrip("./").strip()


def phash(img: Image.Image, size: int = 32, hash_size: int = 8) -> int:
    """64-bit perceptual hash (DCT). Stable under resize and JPEG recompression."""
    g = img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(g, dtype=np.float64)
    d = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:hash_size, :hash_size].copy()
    low[0, 0] = 0.0  # drop DC
    bits = (low > np.median(low)).reshape(-1)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def phash_file(path: Path) -> int:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return phash(im)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def group_by_hash(hashes: dict[str, int], max_dist: int = MATCH_DIST) -> dict[str, str]:
    """key -> group id (the smallest key in its group). Near-duplicates share a group."""
    keys = sorted(hashes)
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, a in enumerate(keys):
        ha = hashes[a]
        for b in keys[i + 1:]:
            if hamming(ha, hashes[b]) <= max_dist:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return {k: find(k) for k in keys}


def in_validation(group_id: str, val_pct: int = VAL_PCT) -> bool:
    return int(hashlib.sha1(group_id.encode()).hexdigest()[:8], 16) % 100 < val_pct


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------

@dataclass
class Item:
    path: str
    split: str  # train | validation | holdout
    label: str
    phash: int
    group: str
    source: str  # plantdoc | real_world_set | plantvillage
    plantdoc_class: str = ""
    stratum: str = ""
    file: str = ""  # prepared relative file, if any

    def as_row(self) -> dict:
        d = asdict(self)
        d["phash"] = f"{self.phash:016x}"
        return d


@dataclass
class Exclusion:
    path: str
    reason: str
    detail: str = ""


# ---------------------------------------------------------------------------------------------
# Sources.csv / labels.csv
# ---------------------------------------------------------------------------------------------

def read_original_paths(sources_csv: Path) -> dict[str, str]:
    """file -> original_path (normed) from real_world_set/sources.csv."""
    out = {}
    if not sources_csv.exists():
        return out
    with open(sources_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("original_path"):
                out[r["file"]] = norm_path(r["original_path"])
    return out


def load_labelled_images(root: Path) -> list[tuple[str, Path, str]]:
    """(rel_file, abs_path, label) from labels.csv, skipping missing files."""
    labels = root / "labels.csv"
    if not labels.exists():
        return []
    rows = []
    with open(labels, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            path = root / r["file"]
            if path.exists():
                rows.append((r["file"], path, r.get("label", "")))
    return rows


# ---------------------------------------------------------------------------------------------
# Near-duplicate search
# ---------------------------------------------------------------------------------------------

def near_dups(query: dict[str, int], gallery: dict[str, int], max_dist: int = MATCH_DIST
              ) -> list[tuple[str, str, int]]:
    """(query_key, gallery_key, distance) for every pair at or under max_dist."""
    hits = []
    for qk, qh in query.items():
        for gk, gh in gallery.items():
            d = hamming(qh, gh)
            if d <= max_dist:
                hits.append((qk, gk, d))
    return hits


def first_near_dup(h: int, gallery: dict[str, int], max_dist: int = MATCH_DIST) -> tuple[str, int] | None:
    for gk, gh in gallery.items():
        d = hamming(h, gh)
        if d <= max_dist:
            return gk, d
    return None


# ---------------------------------------------------------------------------------------------
# Assertions (fail loudly)
# ---------------------------------------------------------------------------------------------

class ProtocolError(SystemExit):
    """Raised when a leakage rule is broken. SystemExit so training scripts stop."""


def _pairs_across(a: list[Item], b: list[Item], max_dist: int = MATCH_DIST) -> list[str]:
    problems = []
    paths_a = {norm_path(x.path) for x in a}
    for x in b:
        if norm_path(x.path) in paths_a:
            problems.append(f"shared path {x.path}")
    ha = {x.path: x.phash for x in a}
    hb = {x.path: x.phash for x in b}
    for qa, qb, d in near_dups(ha, hb, max_dist):
        problems.append(f"pHash dist {d}: {qa} ~ {qb}")
    return problems


def assert_protocol(train: list[Item], val: list[Item], test: list[Item], extra: list[Item] | None = None
                    ) -> None:
    """Fail if train/val leak into the final test, or if a near-duplicate straddles train and val."""
    extra = extra or []
    problems = []
    problems += [f"train/val vs test: {p}" for p in _pairs_across(train + val, test)]
    problems += [f"train vs val: {p}" for p in _pairs_across(train, val)]
    if extra:
        problems += [f"extra (PlantVillage) vs test: {p}" for p in _pairs_across(extra, test)]
    groups_tv = defaultdict(set)
    for x in train:
        groups_tv[x.group].add("train")
    for x in val:
        groups_tv[x.group].add("validation")
    for g, splits in groups_tv.items():
        if len(splits) > 1:
            problems.append(f"group {g} is in both {sorted(splits)}")
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise ProtocolError(f"Data-protocol assertion failed ({len(problems)} problem(s)):\n  {shown}{more}")


# ---------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------

def counts_by(items: list[Item], key: str) -> dict[str, int]:
    return dict(sorted(Counter(getattr(x, key) for x in items).items()))


def per_class_table(groups: dict[str, list[Item]]) -> dict[str, dict[str, int]]:
    labels = sorted({x.label for xs in groups.values() for x in xs})
    out = {}
    for lab in labels:
        out[lab] = {name: sum(1 for x in xs if x.label == lab) for name, xs in groups.items()}
    return out


def write_manifest(path: Path, *, commit: str, pinned: str, exclusions: list[Exclusion],
                   groups: dict[str, list[Item]], params: dict, extra: dict | None = None) -> dict:
    payload = {
        "plantdoc_commit": commit,
        "plantdoc_commit_pinned": pinned,
        "commit_matches_pin": commit == pinned,
        "params": params,
        "counts": {name: len(xs) for name, xs in groups.items()},
        "counts_by_source": {name: counts_by(xs, "source") for name, xs in groups.items()},
        "counts_by_stratum": {name: counts_by(xs, "stratum") for name, xs in groups.items() if any(x.stratum for x in xs)},
        "per_class": per_class_table(groups),
        "exclusions": dict(sorted(Counter(e.reason for e in exclusions).items())),
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def load_items_from_manifest(manifest_csv: Path, split: str | None = None) -> list[dict]:
    rows = list(csv.DictReader(open(manifest_csv, newline="", encoding="utf-8")))
    if split:
        rows = [r for r in rows if r["split"] == split]
    return rows


# ---------------------------------------------------------------------------------------------
# PlantVillage hash cache (so every training run can assert against the final test cheaply)
# ---------------------------------------------------------------------------------------------

def load_hash_cache(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: int(v, 16) if isinstance(v, str) else int(v) for k, v in raw.items()}


def save_hash_cache(path: Path, cache: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: f"{v:016x}" for k, v in cache.items()}) + "\n")


def phash_cached(path: Path, cache: dict[str, int]) -> int:
    key = f"{path.resolve()}:{path.stat().st_size}"
    if key in cache:
        return cache[key]
    h = phash_file(path)
    cache[key] = h
    return h
