"""Build a labelled real-world test set (50 images) from Cropped-PlantDoc.

Usage: python build_set.py <plantdoc_repo_dir> <out_dir>
Deterministic (seed 42). Reproducible by anyone who clones PlantDoc.
"""
import csv, random, shutil, sys
from pathlib import Path
from PIL import Image

SEED, N_TOTAL, MAX_SIDE = 42, 50, 1600
REPO_URL = "https://github.com/pratikkayal/PlantDoc-Dataset"

from plantdoc_map import MAP  # PlantDoc class -> PlantVillage class (shared with prepare_plantdoc.py)

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
rng = random.Random(SEED)

pool = {}
for split in ("train", "test"):
    for cls in MAP:
        d = repo / split / cls
        for p in sorted(d.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        if min(im.size) < 100:
                            continue
                except Exception:
                    continue
                pool.setdefault(cls, []).append(p)

classes = sorted(pool)
assert len(classes) == len(MAP), set(MAP) - set(classes)
chosen = {c: [rng.choice(pool[c])] for c in classes}          # 1 per class
extra = rng.sample(classes, N_TOTAL - len(classes))           # +1 for 23 random classes
for c in extra:
    rest = [p for p in pool[c] if p not in chosen[c]]
    chosen[c].append(rng.choice(rest))

# tuning / final-test split: classes with 2 images -> one each; singles alternate
assign, singles = {}, []
for c in classes:
    imgs = chosen[c]
    if len(imgs) == 2:
        order = ["tune", "final"]; rng.shuffle(order)
        assign[imgs[0]], assign[imgs[1]] = order
    else:
        singles.append(imgs[0])
for i, p in enumerate(singles):
    assign[p] = "tune" if i % 2 == 0 else "final"

if out.exists():
    shutil.rmtree(out)
(out / "images").mkdir(parents=True)
labels, sources, n = [], [], 0
for c in classes:
    for p in chosen[c]:
        n += 1
        pv = MAP[c]
        slug = "".join(ch if ch.isalnum() else "_" for ch in c).strip("_")
        fname = f"rw_{n:03d}_{slug}{p.suffix.lower()}"
        dest_dir = out / "images" / pv
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / fname
        with Image.open(p) as im:
            w, h = im.size
            if max(w, h) > MAX_SIDE:
                im = im.convert("RGB"); im.thumbnail((MAX_SIDE, MAX_SIDE))
                dest = dest.with_suffix(".jpg"); im.save(dest, quality=92)
            else:
                shutil.copyfile(p, dest)
        rel = dest.relative_to(out).as_posix()
        labels.append([rel, pv, c, assign[p]])
        sources.append([rel, "Cropped-PlantDoc", p.relative_to(repo).as_posix(), c,
                        "CC BY 4.0 (dataset repository)", REPO_URL])

with open(out / "labels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "label", "plantdoc_class", "split"]); w.writerows(labels)
with open(out / "sources.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "source", "original_path", "original_class", "license", "url"]); w.writerows(sources)
print(len(labels), "images,", len(classes), "classes,",
      sum(r[3] == "tune" for r in labels), "tune /", sum(r[3] == "final" for r in labels), "final")
