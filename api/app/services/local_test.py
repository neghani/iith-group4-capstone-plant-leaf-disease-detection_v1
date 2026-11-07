"""Local smoke harness: sample labelled photos from model/data/real_world_set.

Same classify path as Web Test / Check a crop (quality gate + leaf crop + confidence band
+ optional GT crop hint). GT comes from labels.csv (PlantVillage label), not a web search.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Iterator

from app.config import get_settings
from app.services.smoke_common import classify_prepared_row, tally_verdicts
from app.vision.hf_client import hf_configured
from app.vision.predictor import display_name
from app.vision.quality import prepare_image

_log = logging.getLogger("leafcheck.local_test")


def _set_roots() -> list[Path]:
    here = Path(__file__).resolve()
    # repo: …/api/app/services → parents[3]=repo
    # docker: /app/app/services → parents[2]=/app (if set is mounted under /app/model/...)
    return [
        here.parents[3] / "model" / "data" / "real_world_set",
        Path("/app/model/data/real_world_set"),
        here.parents[2] / "model" / "data" / "real_world_set",
    ]


def resolve_set_dir() -> Path | None:
    for root in _set_roots():
        if (root / "labels.csv").is_file():
            return root
    return None


def _load_items(set_dir: Path) -> list[dict]:
    path = set_dir / "labels.csv"
    items: list[dict] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = (row.get("file") or "").strip().strip('"')
            label = (row.get("label") or "").strip().strip('"')
            if not rel or not label:
                continue
            abs_path = set_dir / rel
            if not abs_path.is_file():
                # Some checkouts only have labels.csv (images git-ignored).
                continue
            items.append({
                "path": abs_path,
                "rel": rel,
                "class_id": label,
                "split": (row.get("split") or "").strip(),
                "plantdoc_class": (row.get("plantdoc_class") or "").strip(),
            })
    return items


def list_local_set() -> dict:
    settings = get_settings()
    set_dir = resolve_set_dir()
    items = _load_items(set_dir) if set_dir else []
    splits: dict[str, int] = {}
    for item in items:
        splits[item["split"] or "unknown"] = splits.get(item["split"] or "unknown", 0) + 1
    return {
        "model_id": settings.hf_model_id,
        "model_revision": settings.hf_model_revision,
        "hf_configured": hf_configured(),
        "confidence_threshold": settings.confidence_threshold,
        "set_dir": str(set_dir) if set_dir else None,
        "available": bool(items),
        "n_images": len(items),
        "n_labels_csv": _count_csv_rows(set_dir) if set_dir else 0,
        "splits": splits,
        "note": (
            "Samples random labelled photos from model/data/real_world_set (PlantDoc-mapped). "
            "Same gate + leaf crop + confidence band as Web Test / Check a crop. "
            "Images are often git-ignored — keep the folder locally or mount it in Docker."
        ),
    }


def _count_csv_rows(set_dir: Path | None) -> int:
    if set_dir is None:
        return 0
    path = set_dir / "labels.csv"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def run_local_test(
    n: int = 10,
    *,
    use_gt_crop_hint: bool = True,
    split: str | None = None,
) -> Iterator[dict]:
    """Yield NDJSON events for n random real_world_set images."""
    if not hf_configured():
        yield {"status": "error", "detail": "HF_ENDPOINT_URL is not configured."}
        return

    set_dir = resolve_set_dir()
    if set_dir is None:
        yield {
            "status": "error",
            "detail": (
                "real_world_set not found (expected model/data/real_world_set/labels.csv). "
                "From repo root, ensure the set is present; in Docker mount it at "
                "/app/model/data/real_world_set."
            ),
        }
        return

    items = _load_items(set_dir)
    if split:
        items = [i for i in items if i["split"] == split]
    if not items:
        csv_n = _count_csv_rows(set_dir)
        detail = (
            f"No image files under {set_dir} "
            f"(labels.csv has {csv_n} rows but images/ may be missing — they are git-ignored)."
        )
        yield {"status": "error", "detail": detail}
        return

    n = max(1, min(int(n), 15, len(items)))
    sample = random.sample(items, n)
    settings = get_settings()
    pin_id, pin_rev = settings.hf_model_id, settings.hf_model_revision
    threshold = float(settings.confidence_threshold)

    yield {
        "status": "searching",
        "mode": "local",
        "source": "real_world_set",
        "set_dir": str(set_dir),
        "gt_display": "real_world_set (random)",
        "schedule": [display_name(i["class_id"]) for i in sample],
        "n": n,
        "model_id": pin_id,
        "model_revision": pin_rev,
        "confidence_threshold": threshold,
        "use_gt_crop_hint": use_gt_crop_hint,
        "split": split,
    }
    yield {"status": "found", "candidates": n}

    rows: list[dict] = []
    for index, item in enumerate(sample):
        try:
            jpeg = item["path"].read_bytes()
        except OSError as exc:
            _log.warning("read failed %s: %s", item["path"], exc)
            continue
        prepared = prepare_image(jpeg)
        # Local set is labelled — still run the gate, but do not skip the row if it fails;
        # show abstain/retake so demos see the gate behaviour.
        row = classify_prepared_row(
            index,
            item["class_id"],
            str(item["rel"]),
            jpeg,
            prepared,
            use_gt_crop_hint=use_gt_crop_hint,
        )
        row["split"] = item["split"]
        row["plantdoc_class"] = item["plantdoc_class"]
        rows.append(row)
        yield row

    if not rows:
        yield {"status": "error", "detail": "Could not read any real_world_set images."}
        return

    tallied = tally_verdicts(rows)
    yield {
        "status": "done",
        "mode": "local",
        "source": "real_world_set",
        **tallied,
        "confidence_threshold": threshold,
        "use_gt_crop_hint": use_gt_crop_hint,
        "class_id": None,
        "gt_display": "real_world_set (random)",
        "model_id": pin_id,
        "model_revision": pin_rev,
    }
