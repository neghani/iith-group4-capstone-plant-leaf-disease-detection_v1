"""Download ~50 web leaf photos for UI / smoke testing the live PlantVillage pin.

Uses DuckDuckGo image search (no Google API key). Files are named with a safe slug
of the PlantVillage class id; ``manifest.json`` keeps the exact Hub label.

    python model/download_web_test_set.py --clean
    python model/download_web_test_set.py --out data/web_test_set --limit 50 --clean

Targets match MobileNet / ResNet PlantVillage (38 classes). Prefer clear single-leaf
photos. For local review only — delete non-leaf / noisy shots by hand. Respect site
terms; do not redistribute the images.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_OUT = REPO / "data" / "web_test_set"

# Live pin (shared/release.json). Override with --model / --revision if needed.
DEFAULT_MODEL = "neghani/plantvillage-mobilenetv2"
DEFAULT_REVISION = "v1.2"

# (exact Hub label, search query) — all 38 classes, then extras for common demo crops.
# Extras use the same label so review files stay class-named (*_2, *_3).
TARGETS: list[tuple[str, str]] = [
    # Apple (4)
    ("Apple___Apple_scab", "apple scab leaf disease olive spots close up"),
    ("Apple___Black_rot", "apple black rot leaf frog eye spots close up"),
    ("Apple___Cedar_apple_rust", "apple cedar apple rust leaf orange spots close up"),
    ("Apple___healthy", "healthy apple tree leaf close up green single leaf"),
    # Blueberry / Cherry
    ("Blueberry___healthy", "healthy blueberry leaf close up green"),
    ("Cherry_(including_sour)___Powdery_mildew", "cherry powdery mildew leaf white coating close up"),
    ("Cherry_(including_sour)___healthy", "healthy cherry leaf close up green"),
    # Corn (4)
    ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "corn gray leaf spot rectangular lesions close up"),
    ("Corn_(maize)___Common_rust_", "corn maize common rust leaf pustules orange close up"),
    ("Corn_(maize)___Northern_Leaf_Blight", "corn northern leaf blight long gray lesions"),
    ("Corn_(maize)___healthy", "healthy corn maize leaf close up green"),
    # Grape (4)
    ("Grape___Black_rot", "grape black rot leaf lesions brown close up"),
    ("Grape___Esca_(Black_Measles)", "grape esca black measles leaf stripes close up"),
    ("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "grape isariopsis leaf blight dark spots close up"),
    ("Grape___healthy", "healthy grape vine leaf close up green"),
    # Orange / Peach
    ("Orange___Haunglongbing_(Citrus_greening)", "citrus greening HLB blotchy leaf yellow mottling close up"),
    ("Peach___Bacterial_spot", "peach bacterial spot leaf lesions close up"),
    ("Peach___healthy", "healthy peach leaf close up green"),
    # Pepper
    ("Pepper,_bell___Bacterial_spot", "bell pepper bacterial spot leaf angular lesions close up"),
    ("Pepper,_bell___healthy", "healthy bell pepper leaf close up green"),
    # Potato (3)
    ("Potato___Early_blight", "potato early blight leaf concentric rings close up"),
    ("Potato___Late_blight", "potato late blight leaf dark water soaked lesions"),
    ("Potato___healthy", "healthy potato plant leaf close up green"),
    # Raspberry / Soybean / Squash / Strawberry
    ("Raspberry___healthy", "healthy raspberry leaf close up green"),
    ("Soybean___healthy", "healthy soybean leaf close up green trifoliate"),
    ("Squash___Powdery_mildew", "squash powdery mildew leaf white spots close up"),
    ("Strawberry___Leaf_scorch", "strawberry leaf scorch purple spots disease close up"),
    ("Strawberry___healthy", "healthy strawberry leaf close up green"),
    # Tomato (10)
    ("Tomato___Bacterial_spot", "tomato bacterial spot leaf small dark spots close up"),
    ("Tomato___Early_blight", "tomato early blight leaf concentric target spots close up"),
    ("Tomato___Late_blight", "tomato late blight leaf water soaked lesions close up"),
    ("Tomato___Leaf_Mold", "tomato leaf mold olive green patches underside close up"),
    ("Tomato___Septoria_leaf_spot", "tomato septoria leaf spot small circular lesions close up"),
    ("Tomato___Spider_mites Two-spotted_spider_mite", "tomato spider mite damage leaf stippling yellow close up"),
    ("Tomato___Target_Spot", "tomato target spot leaf lesions concentric close up"),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "tomato yellow leaf curl virus curled yellow leaf close up"),
    ("Tomato___Tomato_mosaic_virus", "tomato mosaic virus mottled leaf close up"),
    ("Tomato___healthy", "healthy tomato leaf close up green"),
    # Extras (~12) — second angles for high-likelihood demo crops
    ("Tomato___Early_blight", "tomato early blight diseased leaf field photo close up"),
    ("Tomato___Late_blight", "tomato late blight infected leaf garden close up"),
    ("Tomato___Bacterial_spot", "tomato bacterial leaf spot symptoms close up photo"),
    ("Tomato___healthy", "green healthy tomato plant leaf macro photo"),
    ("Potato___Early_blight", "potato early blight foliar disease leaf close up photo"),
    ("Potato___Late_blight", "potato late blight infected foliage leaf close up"),
    ("Pepper,_bell___Bacterial_spot", "pepper bacterial spot diseased leaf close up photo"),
    ("Pepper,_bell___healthy", "green healthy pepper plant leaf close up photo"),
    ("Corn_(maize)___Common_rust_", "maize common rust uredinia leaf close up photo"),
    ("Corn_(maize)___Northern_Leaf_Blight", "northern corn leaf blight lesions photo close up"),
    ("Apple___Apple_scab", "apple scab infected leaf orchard close up photo"),
    ("Grape___Black_rot", "grape black rot diseased leaf vineyard close up"),
]


def slug(label: str) -> str:
    """Filesystem-safe stem that still reads as the disease."""
    s = label.replace("___", "_").replace(",", "").replace("(", "").replace(")", "")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def unique_dest(out_dir: Path, class_id: str, used: dict[str, int]) -> Path:
    base = slug(class_id)
    n = used.get(base, 0) + 1
    used[base] = n
    name = f"{base}.jpg" if n == 1 else f"{base}_{n}.jpg"
    return out_dir / name


def download_one(
    class_id: str,
    query: str,
    dest: Path,
    *,
    skip_urls: set[str],
    timeout: float = 25.0,
) -> str | None:
    """Save one JPEG to dest. Returns the image URL used, or None."""
    from ddgs import DDGS

    results = list(DDGS().images(query, max_results=18))
    headers = {"User-Agent": "CropCareAI-test-set/1.0 (capstone local testing)"}
    for item in results:
        url = item.get("image") or item.get("url")
        if not url or url in skip_urls:
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
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest, format="JPEG", quality=90)
            skip_urls.add(url)
            return url
        except Exception:
            continue
    return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--limit", type=int, default=50, help="How many images to pull (default 50)")
    p.add_argument("--clean", action="store_true", help="Remove existing jpg/json in --out first")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for path in out.iterdir():
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".json"}:
                path.unlink()
                print(f"[clean] {path.name}", flush=True)

    used: dict[str, int] = {}
    skip_urls: set[str] = set()
    manifest = []
    targets = TARGETS[: args.limit]
    for class_id, query in targets:
        dest = unique_dest(out, class_id, used)
        print(f"[download] {dest.name} ← {class_id}", flush=True)
        url = download_one(class_id, query, dest, skip_urls=skip_urls)
        if url is None:
            # Roll back the counter so the next attempt for this class keeps numbering tidy.
            base = slug(class_id)
            used[base] = max(0, used.get(base, 1) - 1)
            if dest.exists():
                dest.unlink()
            print(f"  FAILED {class_id}", flush=True)
            continue
        print(f"  OK {dest.name}", flush=True)
        manifest.append({
            "class_id": class_id,
            "file": dest.name,
            "slug": dest.stem,
            "query": query,
            "model": args.model,
            "revision": args.revision,
            "source_url": url,
        })
    payload = {
        "n": len(manifest),
        "live_pin": f"{args.model}@{args.revision}",
        "label_space": "plantvillage_38",
        "items": manifest,
        "note": (
            "Web smoke set for manual review. Delete non-leaf / noisy files, then keep "
            "the rest for UI demos. Not a decision / bake-off set."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(manifest)} images to {out}", flush=True)
    if len(manifest) < min(args.limit, 30):
        raise SystemExit(f"Only {len(manifest)} downloads succeeded; need at least 30.")


if __name__ == "__main__":
    main()
