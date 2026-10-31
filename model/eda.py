"""Read-only PlantVillage sanity check and EDA. Does not rewrite splits or drop files.

    python model/eda.py --data_dir "$DATA_DIR"

Hard-fails (exit 1) on a missing color folder, a class set that does not match
api/tests/model_labels.json, an empty class, or an unreadable image. Warnings (imbalance,
small images, non-RGB, extra files, a capped pHash sample) are recorded and do not block training.

Writes model/outputs/plantvillage_eda.json, plantvillage_eda.md,
plantvillage_class_counts.png, and plantvillage_aug_preview.png.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from common import IMAGE_EXTS, resolve_data_dir, scan_imagefolder  # noqa: E402
from progress import fmt_secs  # noqa: E402
from data_protocol import MATCH_DIST, hamming, phash_file  # noqa: E402

DEFAULT_LABELS = REPO_ROOT / "api" / "tests" / "model_labels.json"
DEFAULT_OUT = HERE / "outputs"
MIN_SIZE = 224
PHASH_PER_CLASS = 30
IMBALANCE_WARN = 2.0
MAX_PAIR_ROWS = 50
MAX_EXAMPLE = 8
PREVIEW_CLASSES = 4
PREVIEW_VIEWS = 4
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_expected_labels(path: Path) -> list[str]:
    labels = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(labels, dict):
        labels = list(labels.values())
    return list(labels)


def inspect_image(path: Path) -> tuple[dict | None, str | None]:
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as e:
        return None, f"unreadable: {e}"
    try:
        with Image.open(path) as im:
            im.load()
            width, height = im.size
            mode = im.mode
            im.convert("RGB").getpixel((0, 0))
            return {"width": int(width), "height": int(height), "mode": mode}, None
    except Exception as e:
        return None, f"unreadable: {e}"


def _listdir_files(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    return [p for p in class_dir.iterdir() if p.is_file() and not p.name.startswith(".")]


def _sample_near_dups(
    data_dir: Path, by_class: dict[str, list[Path]], n: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    pairs: list[dict] = []
    for label, files in sorted(by_class.items()):
        sample = files if len(files) <= n else rng.sample(files, n)
        hashes: dict[str, int] = {}
        for path in sample:
            try:
                hashes[str(path.relative_to(data_dir))] = phash_file(path)
            except Exception:
                continue
        keys = list(hashes)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                dist = hamming(hashes[a], hashes[b])
                if dist <= MATCH_DIST:
                    pairs.append({"a": a, "b": b, "dist": dist, "label": label})
    return pairs


def audit(
    data_dir: str | Path,
    *,
    crop_filter: str | None = None,
    labels_path: Path | None = None,
    phash_per_class: int = PHASH_PER_CLASS,
    min_size: int = MIN_SIZE,
    seed: int = 42,
) -> dict:
    errors: list[str] = []
    warnings: list[dict] = []

    try:
        root = resolve_data_dir(data_dir)
    except FileNotFoundError as e:
        return {
            "data_dir": str(data_dir),
            "ok": False,
            "errors": [str(e)],
            "warnings": [],
            "per_class": [],
            "n_classes": 0,
            "n_images": 0,
            "near_duplicates": {"sampled_per_class": phash_per_class, "n_pairs": 0, "pairs": []},
        }

    try:
        label_names, items = scan_imagefolder(root, crop_filter=crop_filter, seed=seed)
    except ValueError as e:
        return {
            "data_dir": str(root),
            "ok": False,
            "errors": [str(e)],
            "warnings": [],
            "per_class": [],
            "n_classes": 0,
            "n_images": 0,
            "near_duplicates": {"sampled_per_class": phash_per_class, "n_pairs": 0, "pairs": []},
        }

    expected_path = Path(labels_path) if labels_path else DEFAULT_LABELS
    if not crop_filter:
        expected = load_expected_labels(expected_path)
        if label_names != expected:
            missing = [n for n in expected if n not in label_names]
            extra = [n for n in label_names if n not in expected]
            parts = []
            if missing:
                parts.append(f"missing {missing}")
            if extra:
                parts.append(f"extra {extra}")
            if not missing and not extra:
                parts.append(f"order differs from {expected_path.name}")
            errors.append("class set does not match " + expected_path.name + ": " + "; ".join(parts))

    by_class: dict[str, list[Path]] = {name: [] for name in label_names}
    for rel, idx in items:
        by_class[label_names[idx]].append(root / rel)

    per_class = []
    small: list[str] = []
    non_rgb: list[str] = []
    extra_files: list[str] = []
    unreadable: list[str] = []
    n_images = 0
    n_classes = len(label_names)

    total_files = sum(len(by_class[n]) for n in label_names)
    print(f"[eda] scanning {total_files} images in {len(label_names)} classes under {root}", flush=True)
    t_start = time.time()
    done_all = 0
    last_beat = t_start
    for i, name in enumerate(label_names, start=1):
        files = by_class[name]
        class_dir = root / name
        listed = _listdir_files(class_dir)
        image_listed = [p for p in listed if p.suffix.lower() in IMAGE_EXTS]
        extras = [p.name for p in listed if p.suffix.lower() not in IMAGE_EXTS]
        if extras:
            extra_files.append(f"{name}: {', '.join(extras[:MAX_EXAMPLE])}")

        if not files:
            errors.append(f"empty class: {name}")

        modes: dict[str, int] = defaultdict(int)
        widths: list[int] = []
        heights: list[int] = []
        opened = 0
        for path in files:
            done_all += 1
            now = time.time()
            if now - last_beat >= 10:
                last_beat = now
                el = now - t_start
                rate = done_all / el if el else 0.0
                eta = (total_files - done_all) / rate if rate else 0.0
                print(f"[eda] {done_all}/{total_files} images ({100 * done_all // total_files}%)  "
                      f"class {i}/{n_classes} {name}  {rate:.0f} img/s  elapsed {fmt_secs(el)}  "
                      f"eta {fmt_secs(eta)}", flush=True)
            info, err = inspect_image(path)
            if err:
                unreadable.append(f"{path.relative_to(root)} ({err})")
                continue
            assert info is not None
            n_images += 1
            opened += 1
            modes[info["mode"]] += 1
            widths.append(info["width"])
            heights.append(info["height"])
            if min(info["width"], info["height"]) < min_size:
                small.append(str(path.relative_to(root)))
            if info["mode"] != "RGB":
                non_rgb.append(f"{path.relative_to(root)} ({info['mode']})")

        per_class.append(
            {
                "name": name,
                "n": len(files),
                "n_listdir": len(listed),
                "n_image_ext": len(image_listed),
                "modes": dict(modes),
                "min_width": min(widths) if widths else None,
                "min_height": min(heights) if heights else None,
            }
        )
        elapsed = time.time() - t_start
        eta = elapsed / i * (n_classes - i)
        print(f"[eda] class {i}/{n_classes} ({100 * i // n_classes}%) {name}: {opened} images  "
              f"elapsed {fmt_secs(elapsed)}  eta {fmt_secs(eta)}", flush=True)

    if unreadable:
        errors.append(f"{len(unreadable)} unreadable image(s): " + "; ".join(unreadable[:MAX_EXAMPLE]))

    counts = [row["n"] for row in per_class if row["n"] > 0]
    imbalance = None
    min_class = max_class = None
    if counts:
        lo, hi = min(counts), max(counts)
        imbalance = round(hi / lo, 2) if lo else None
        min_class = min((row for row in per_class if row["n"] > 0), key=lambda r: r["n"])["name"]
        max_class = max((row for row in per_class if row["n"] > 0), key=lambda r: r["n"])["name"]
        if imbalance is not None and imbalance >= IMBALANCE_WARN:
            warnings.append(
                {
                    "code": "imbalance",
                    "message": (
                        f"class imbalance {imbalance}:1 "
                        f"(min {min_class}={lo}, max {max_class}={hi})"
                    ),
                    "ratio": imbalance,
                    "min_class": min_class,
                    "max_class": max_class,
                }
            )

    if small:
        warnings.append(
            {
                "code": "small_images",
                "message": f"{len(small)} image(s) smaller than {min_size} px",
                "examples": small[:MAX_EXAMPLE],
                "n": len(small),
            }
        )
    if non_rgb:
        warnings.append(
            {
                "code": "non_rgb",
                "message": f"{len(non_rgb)} non-RGB image(s); train.py converts to RGB",
                "examples": non_rgb[:MAX_EXAMPLE],
                "n": len(non_rgb),
            }
        )
    if extra_files:
        warnings.append(
            {
                "code": "extra_files",
                "message": "listdir count includes non-image files that scan_imagefolder skips",
                "examples": extra_files[:MAX_EXAMPLE],
                "n": len(extra_files),
            }
        )

    print(f"[eda] image scan done in {fmt_secs(time.time() - t_start)}; near-duplicate sample next", flush=True)
    pairs = _sample_near_dups(root, by_class, phash_per_class, seed) if phash_per_class else []
    if pairs:
        warnings.append(
            {
                "code": "near_duplicates",
                "message": (
                    f"{len(pairs)} near-duplicate pair(s) in a sample of up to "
                    f"{phash_per_class} images/class (pHash Hamming ≤ {MATCH_DIST}). "
                    "Splits are not rewritten."
                ),
                "n": len(pairs),
            }
        )

    payload = {
        "data_dir": str(root),
        "ok": not errors,
        "n_classes": len(label_names),
        "n_images": n_images,
        "label_names": label_names,
        "imbalance_ratio": imbalance,
        "min_class": min_class,
        "max_class": max_class,
        "errors": errors,
        "warnings": warnings,
        "per_class": per_class,
        "near_duplicates": {
            "sampled_per_class": phash_per_class,
            "n_pairs": len(pairs),
            "pairs": pairs[:MAX_PAIR_ROWS],
        },
    }
    return payload


def report_md(payload: dict) -> str:
    lines = [
        "# PlantVillage EDA",
        "",
        f"- Data dir: `{payload.get('data_dir', '')}`",
        f"- Classes: {payload.get('n_classes', 0)}",
        f"- Readable images: {payload.get('n_images', 0)}",
        f"- Status: {'OK' if payload.get('ok') else 'FAILED'}",
    ]
    if payload.get("imbalance_ratio") is not None:
        lines.append(
            f"- Imbalance (max/min): {payload['imbalance_ratio']}:1 "
            f"({payload.get('min_class')} … {payload.get('max_class')})"
        )
    lines += ["", "## Per-class counts", "", "| Class | Images | Share |", "|---|---:|---:|"]
    total = sum(row["n"] for row in payload.get("per_class", [])) or 1
    for row in payload.get("per_class", []):
        share = 100.0 * row["n"] / total
        lines.append(f"| `{row['name']}` | {row['n']} | {share:.2f}% |")
    errors = payload.get("errors") or []
    warnings = payload.get("warnings") or []
    if errors:
        lines += ["", "## Errors (training must not start)", ""]
        lines.extend(f"- {e}" for e in errors)
    if warnings:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {w['message']}" for w in warnings)
    n_pairs = (payload.get("near_duplicates") or {}).get("n_pairs", 0)
    if n_pairs:
        lines += ["", f"Near-duplicate sample pairs: {n_pairs} (splits unchanged)."]
    return "\n".join(lines) + "\n"


def _denorm(tensor, mean=MEAN, std=STD) -> Image.Image:
    import torch

    m = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    s = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    x = (tensor * s + m).clamp(0, 1)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


class _StubProcessor:
    size = {"shortest_edge": 256}
    crop_size = {"height": 224, "width": 224}
    image_mean = MEAN
    image_std = STD


def write_preview(
    data_dir: Path,
    label_names: list[str],
    by_rel: dict[str, list[str]],
    out_path: Path,
    n_classes: int = PREVIEW_CLASSES,
    n_views: int = PREVIEW_VIEWS,
    seed: int = 42,
) -> Path | None:
    from aug import build_transforms

    chosen = []
    for name in label_names:
        rels = by_rel.get(name) or []
        if rels:
            chosen.append((name, rels[0]))
        if len(chosen) >= n_classes:
            break
    if not chosen:
        return None

    import torch

    torch.manual_seed(seed)
    tf = build_transforms(_StubProcessor(), 224, is_training=True, rrc_scale=(0.7, 1.0))
    cell = 224
    cols = 1 + n_views
    rows = len(chosen)
    canvas = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for r, (_name, rel) in enumerate(chosen):
        with Image.open(data_dir / rel) as im:
            src = im.convert("RGB")
        orig = src.resize((cell, cell), Image.Resampling.BILINEAR)
        canvas.paste(orig, (0, r * cell))
        for c in range(n_views):
            canvas.paste(_denorm(tf(src)), ((c + 1) * cell, r * cell))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def write_class_counts_chart(payload: dict, out_path: Path) -> Path | None:
    """Horizontal bar of per-class image counts. Does not change splits."""
    rows = payload.get("per_class") or []
    if not rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"WARN: class-count chart skipped: {e}", flush=True)
        return None
    names = [str(r["name"]) for r in rows]
    counts = [int(r["n"]) for r in rows]
    fig_h = max(4.0, 0.28 * len(names))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(names, counts, color="#3d7a46")
    ax.set_xlabel("Images")
    ax.set_title("PlantVillage class counts")
    ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_outputs(
    payload: dict,
    out_dir: Path,
    preview_path: Path | None = None,
    chart_path: Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    if preview_path:
        payload["augmentation_preview"] = str(preview_path)
    if chart_path:
        payload["class_counts_chart"] = str(chart_path)
    (out_dir / "plantvillage_eda.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "plantvillage_eda.md").write_text(report_md(payload), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="model/data/plantvillage")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--crop_filter", default=None)
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--phash_per_class", type=int, default=PHASH_PER_CLASS)
    p.add_argument("--min_size", type=int, default=MIN_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_preview", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = audit(
        args.data_dir,
        crop_filter=args.crop_filter,
        labels_path=Path(args.labels),
        phash_per_class=args.phash_per_class,
        min_size=args.min_size,
        seed=args.seed,
    )
    out_dir = Path(args.out)
    preview = None
    if not args.skip_preview and payload.get("label_names") and not payload["errors"]:
        root = Path(payload["data_dir"])
        by_rel: dict[str, list[str]] = defaultdict(list)
        names = payload["label_names"]
        _, items = scan_imagefolder(root, crop_filter=args.crop_filter, seed=args.seed)
        for rel, idx in items:
            by_rel[names[idx]].append(rel)
        try:
            preview = write_preview(root, names, by_rel, out_dir / "plantvillage_aug_preview.png", seed=args.seed)
        except Exception as e:
            payload["warnings"] = list(payload.get("warnings") or []) + [
                {"code": "preview_failed", "message": f"augmentation preview skipped: {e}"}
            ]
    chart = write_class_counts_chart(payload, out_dir / "plantvillage_class_counts.png")
    write_outputs(payload, out_dir, preview, chart)
    for err in payload.get("errors") or []:
        print("ERROR:", err, flush=True)
    for warn in payload.get("warnings") or []:
        print("WARN:", warn["message"], flush=True)
    print("Wrote", out_dir / "plantvillage_eda.json", flush=True)
    if chart:
        print("Class counts", chart, flush=True)
    if preview:
        print("Preview", preview, flush=True)
    if payload["errors"]:
        raise SystemExit(1)
    print("PlantVillage sanity check passed.", flush=True)


if __name__ == "__main__":
    main()
