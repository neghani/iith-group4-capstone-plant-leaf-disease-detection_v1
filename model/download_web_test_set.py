"""Download ~16 web leaf photos for UI / smoke testing the live PlantVillage pin.

Uses DuckDuckGo image search (no Google API key). Files are named with a safe slug
of the PlantVillage class id; ``manifest.json`` keeps the exact Hub label.

    python model/download_web_test_set.py
    python model/download_web_test_set.py --out data/web_test_set --limit 16

Targets match ``neghani/plantvillage-resnet18`` / MobileNet (38 classes). For local
demo use only. Respect site terms; do not redistribute the images.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_OUT = REPO / "data" / "web_test_set"

# (exact Hub label, search query) — prefer clear single-leaf photos.
TARGETS: list[tuple[str, str]] = [
    ("Apple___Cedar_apple_rust", "apple cedar apple rust leaf orange spots close up"),
    ("Apple___Apple_scab", "apple scab leaf disease spots close up"),
    ("Apple___healthy", "healthy apple tree leaf close up green"),
    ("Corn_(maize)___Common_rust_", "corn maize common rust leaf pustules orange"),
    ("Corn_(maize)___Northern_Leaf_Blight", "corn northern leaf blight long lesions"),
    ("Corn_(maize)___healthy", "healthy corn maize leaf close up green"),
    ("Grape___Black_rot", "grape black rot leaf lesions close up"),
    ("Grape___healthy", "healthy grape vine leaf close up"),
    ("Potato___Early_blight", "potato early blight leaf concentric rings"),
    ("Potato___Late_blight", "potato late blight leaf dark lesions"),
    ("Strawberry___Leaf_scorch", "strawberry leaf scorch disease purple spots"),
    ("Strawberry___healthy", "healthy strawberry leaf close up"),
    ("Tomato___Early_blight", "tomato early blight leaf concentric target spots"),
    ("Tomato___Late_blight", "tomato late blight leaf water soaked lesions"),
    ("Tomato___Bacterial_spot", "tomato bacterial spot leaf small dark spots"),
    ("Tomato___healthy", "healthy tomato leaf close up green"),
    ("Pepper,_bell___Bacterial_spot", "bell pepper bacterial spot leaf angular lesions"),
    ("Pepper,_bell___healthy", "healthy bell pepper leaf close up"),
]


def slug(label: str) -> str:
    """Filesystem-safe stem that still reads as the disease."""
    s = label.replace("___", "__").replace(",", "").replace("(", "").replace(")", "")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def download_one(class_id: str, query: str, out_dir: Path, timeout: float = 25.0) -> Path | None:
    from ddgs import DDGS

    out_dir.mkdir(parents=True, exist_ok=True)
    results = list(DDGS().images(query, max_results=10))
    headers = {"User-Agent": "CropCareAI-test-set/1.0 (capstone local testing)"}
    dest = out_dir / f"{slug(class_id)}.jpg"
    for item in results:
        url = item.get("image") or item.get("url")
        if not url:
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
            resp.raise_for_status()
            raw = resp.content
            if len(raw) < 8_000:
                continue
            img = Image.open(BytesIO(raw)).convert("RGB")
            w, h = img.size
            if min(w, h) < 180:
                continue
            # Skip very square art-like / poster images when possible (weak heuristic).
            if abs(w - h) < 20 and max(w, h) > 1000:
                continue
            img.thumbnail((1600, 1600))
            img.save(dest, format="JPEG", quality=90)
            return dest
        except Exception:
            continue
    return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--limit", type=int, default=len(TARGETS))
    p.add_argument("--clean", action="store_true", help="Remove existing jpg/json in --out first")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for path in out.iterdir():
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".json"}:
                path.unlink()
                print(f"[clean] {path.name}", flush=True)

    manifest = []
    for class_id, query in TARGETS[: args.limit]:
        print(f"[download] {class_id} …", flush=True)
        path = download_one(class_id, query, out)
        if path is None:
            print(f"  FAILED {class_id}", flush=True)
            continue
        print(f"  OK {path.name}", flush=True)
        manifest.append({
            "class_id": class_id,
            "file": path.name,
            "slug": slug(class_id),
            "query": query,
            "model": "neghani/plantvillage-resnet18",
            "revision": "v1.1",
        })
    payload = {
        "n": len(manifest),
        "live_pin": "neghani/plantvillage-resnet18@v1.1",
        "label_space": "plantvillage_38",
        "items": manifest,
        "note": "Web smoke set aligned to live PlantVillage pin. Not a decision / bake-off set.",
    }
    (out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(manifest)} images to {out}", flush=True)
    if len(manifest) < min(args.limit, 12):
        raise SystemExit(f"Only {len(manifest)} downloads succeeded; need at least 12.")


if __name__ == "__main__":
    main()
