"""Download ~20 web leaf photos for UI / smoke testing the MasterDataset pin.

Uses DuckDuckGo image search (no Google API key). Files are named after the
model class id, e.g. sugarcane_rust.jpg, under data/web_test_set/.

    python model/download_web_test_set.py
    python model/download_web_test_set.py --out data/web_test_set --limit 20

For local demo use only. Respect site terms; do not redistribute the images.
"""

from __future__ import annotations

import argparse
import mimetypes
import sys
from pathlib import Path

import requests
from PIL import Image
from io import BytesIO

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_OUT = REPO / "data" / "web_test_set"

# One query per MasterDataset class. Prefer clear single-leaf photos.
TARGETS: list[tuple[str, str]] = [
    ("healthy_sugarcane", "healthy sugarcane leaf close up"),
    ("sugarcane_rust", "sugarcane leaf rust disease symptoms"),
    ("sugarcane_bacterial_blight", "sugarcane bacterial blight leaf"),
    ("healthy_corn", "healthy corn maize leaf close up"),
    ("corn_northern_leaf_blight", "corn northern leaf blight symptoms leaf"),
    ("corn_common_rust", "corn common rust leaf pustules"),
    ("healthy_tomato", "healthy tomato leaf close up"),
    ("tomato_early_blight", "tomato early blight leaf concentric rings"),
    ("tomato_late_blight", "tomato late blight leaf symptoms"),
    ("potato_early_blight", "potato early blight leaf"),
    ("potato_late_blight", "potato late blight leaf"),
    ("healthy_rice", "healthy rice plant leaf close up"),
    ("diseased_rice", "rice leaf blast disease symptoms"),
    ("wheat_yellow_rust", "wheat yellow rust leaf stripes"),
    ("wheat_brown_rust", "wheat brown rust leaf"),
    ("apple_cedar_apple_rust", "apple cedar apple rust leaf spots"),
    ("healthy_apple", "healthy apple leaf close up"),
    ("grape_black_rot", "grape black rot leaf lesions"),
    ("strawberry_leaf_scorch", "strawberry leaf scorch disease"),
    ("cotton_bacterial_blight", "cotton bacterial blight leaf angular spots"),
]


def _ext_from(url: str, content_type: str | None) -> str:
    guess = mimetypes.guess_extension((content_type or "").split(";")[0].strip()) or ""
    if guess in {".jpe", ".jpeg"}:
        return ".jpg"
    if guess in {".jpg", ".png", ".webp"}:
        return guess
    path = url.split("?", 1)[0].lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def download_one(class_id: str, query: str, out_dir: Path, timeout: float = 25.0) -> Path | None:
    from ddgs import DDGS

    out_dir.mkdir(parents=True, exist_ok=True)
    results = list(DDGS().images(query, max_results=8))
    headers = {"User-Agent": "CropCareAI-test-set/1.0 (capstone local testing)"}
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
            # Cap huge photos for the UI upload path.
            img.thumbnail((1600, 1600))
            dest = out_dir / f"{class_id}.jpg"
            img.save(dest, format="JPEG", quality=90)
            return dest
        except Exception:
            continue
    return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for class_id, query in TARGETS[: args.limit]:
        print(f"[download] {class_id} …", flush=True)
        path = download_one(class_id, query, out)
        if path is None:
            print(f"  FAILED {class_id}", flush=True)
            continue
        print(f"  OK {path}", flush=True)
        manifest.append({"class_id": class_id, "file": path.name, "query": query})
    (out / "manifest.json").write_text(
        __import__("json").dumps({"n": len(manifest), "items": manifest}, indent=2) + "\n"
    )
    print(f"Wrote {len(manifest)} images to {out}", flush=True)
    if len(manifest) < args.limit:
        raise SystemExit(f"Only {len(manifest)}/{args.limit} downloads succeeded.")


if __name__ == "__main__":
    main()
