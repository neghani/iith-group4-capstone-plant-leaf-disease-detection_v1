"""Pre-split ImageFolder loaders for the Plant Leaf Disease MasterDataset.

The Kaggle archive is already stratified into train / val / test with the same class
folders in each split. This module does not reshuffle those images.

Eval matches a 224 shortest-edge resize plus a 224 centre crop and ImageNet
normalization, which is what the saved image processor is aligned to in train.py.
"""

from __future__ import annotations

import random
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
    ToTensor,
)

from common import IMAGE_EXTS, crop_from_class, gpu_train_flags

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

_SPLIT_ALIASES = {
    "train": ("train",),
    "validation": ("validation", "val", "valid"),
    "test": ("test",),
}
_SPLIT_DIR_NAMES = {alias for aliases in _SPLIT_ALIASES.values() for alias in aliases}


def build_transforms(is_training: bool) -> Compose:
    """Train: 224 resize, flip, rotation. Eval: 224 resize and centre crop. Both use ImageNet norm."""
    normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if is_training:
        return Compose(
            [
                Resize((IMAGE_SIZE, IMAGE_SIZE)),
                RandomHorizontalFlip(),
                RandomRotation(degrees=180),
                ToTensor(),
                normalize,
            ]
        )
    return Compose(
        [
            Resize(IMAGE_SIZE),
            CenterCrop(IMAGE_SIZE),
            ToTensor(),
            normalize,
        ]
    )


def align_processor(image_processor) -> None:
    """Make the checkpoint processor match build_transforms(is_training=False).

    ConvNeXT processors default to crop_pct=0.875 (resize short side to 256) and
    bicubic resampling. Training uses a 224 short-side resize and bilinear, so those
    must be overwritten or serving sees different pixels than eval.
    """
    from PIL import Image as PILImage

    if hasattr(image_processor, "size"):
        image_processor.size = {"shortest_edge": IMAGE_SIZE}
    if hasattr(image_processor, "crop_size"):
        image_processor.crop_size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    if hasattr(image_processor, "do_resize"):
        image_processor.do_resize = True
    if hasattr(image_processor, "do_center_crop"):
        image_processor.do_center_crop = True
    # crop_pct < 1.0 makes ConvNeXT resize to shortest_edge/crop_pct before the crop.
    if hasattr(image_processor, "crop_pct"):
        image_processor.crop_pct = 1.0
    if hasattr(image_processor, "resample"):
        image_processor.resample = int(PILImage.Resampling.BILINEAR)


def _match_splits(root: Path) -> dict[str, Path] | None:
    if not root.is_dir():
        return None
    children = {p.name.lower(): p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")}
    found: dict[str, Path] = {}
    for key, aliases in _SPLIT_ALIASES.items():
        for alias in aliases:
            if alias in children:
                found[key] = children[alias]
                break
    if set(found) != {"train", "validation", "test"}:
        return None
    return found


def resolve_split_root(path: str | Path) -> dict[str, Path]:
    """Return train/validation/test directories. Accepts the dataset root, a split folder, or one wrapper level."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    if root.is_dir() and root.name.lower() in _SPLIT_DIR_NAMES:
        root = root.parent
    candidates = [root]
    if root.is_dir():
        for child in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            candidates.append(child)
            for grand in sorted(p for p in child.iterdir() if p.is_dir() and not p.name.startswith(".")):
                candidates.append(grand)
    for candidate in candidates:
        found = _match_splits(candidate)
        if found is None:
            continue
        if _class_names(found["train"]):
            return found
    raise FileNotFoundError(
        f"No train/val/test class folders under {root}. "
        "Expected train/, val/ (or valid/), and test/, each with one folder per class."
    )


def _class_names(split_dir: Path) -> list[str]:
    names = []
    for child in sorted(split_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if any(f.suffix.lower() in IMAGE_EXTS for f in child.iterdir() if f.is_file()):
            names.append(child.name)
    return names


def class_files(split_dir: Path) -> dict[str, list[Path]]:
    """Class folder name -> image paths. Empty folders are included so parity checks can fail on them."""
    out: dict[str, list[Path]] = {}
    for child in sorted(split_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        out[child.name] = sorted(f for f in child.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    return out


def assert_class_parity(splits: dict[str, dict[str, list[Path]]]) -> list[str]:
    """Return the shared class names. Raise SystemExit when a split differs or a class is empty."""
    names = {key: set(files) for key, files in splits.items()}
    reference_key, reference = next(iter(names.items()))
    problems = []
    for key, have in names.items():
        if have == reference:
            continue
        missing = sorted(reference - have)
        extra = sorted(have - reference)
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if extra:
            detail.append(f"extra {extra}")
        problems.append(f"{key} vs {reference_key}: {'; '.join(detail)}")
    empty = sorted({name for files in splits.values() for name, paths in files.items() if not paths})
    if empty:
        problems.append(f"empty classes: {empty}")
    if problems:
        raise SystemExit("MasterDataset class sets are not identical:\n  " + "\n  ".join(problems))
    return sorted(reference)


def _filter_classes(classes: list[str], crop_filter: str | None) -> list[str]:
    if not crop_filter:
        return classes
    wanted = {c.strip().lower() for c in crop_filter.split(",") if c.strip()}
    kept = [
        name
        for name in classes
        if crop_from_class(name).lower() in wanted or name.split("___")[0].lower() in wanted
    ]
    if not kept:
        raise SystemExit(f"No MasterDataset classes matched crop_filter={crop_filter!r}.")
    return kept


def _cap(files: list[Path], limit: int | None, rng: random.Random) -> list[Path]:
    if limit is None or len(files) <= limit:
        return files
    chosen = files[:]
    rng.shuffle(chosen)
    return sorted(chosen[:limit])


def collect_split_rows(
    root: str | Path,
    *,
    crop_filter: str | None = None,
    max_samples_per_class: int | None = None,
    seed: int = 42,
) -> tuple[list[str], dict[str, list[tuple[str, int]]], dict[str, Path]]:
    """Label names plus relative (path, label_id) rows for train/validation/test. Paths are relative to the dataset root."""
    split_dirs = resolve_split_root(root)
    dataset_root = split_dirs["train"].parent
    listed = {key: class_files(path) for key, path in split_dirs.items()}
    classes = _filter_classes(assert_class_parity(listed), crop_filter)
    rng = random.Random(seed)
    label_names = list(classes)
    ids = {name: i for i, name in enumerate(label_names)}
    rows: dict[str, list[tuple[str, int]]] = {}
    for key, files_by_class in listed.items():
        split_rows: list[tuple[str, int]] = []
        for name in label_names:
            for path in _cap(files_by_class[name], max_samples_per_class, rng):
                split_rows.append((str(path.relative_to(dataset_root)), ids[name]))
        if not split_rows:
            raise SystemExit(f"MasterDataset split {key!r} has no images after filtering.")
        rows[key] = split_rows
    return label_names, rows, split_dirs


def build_imagefolders(root: str | Path) -> dict[str, ImageFolder]:
    """ImageFolder per split. Train is augmented; validation and test are not."""
    split_dirs = resolve_split_root(root)
    listed = {key: class_files(path) for key, path in split_dirs.items()}
    assert_class_parity(listed)
    folders = {
        "train": ImageFolder(split_dirs["train"], transform=build_transforms(True)),
        "validation": ImageFolder(split_dirs["validation"], transform=build_transforms(False)),
        "test": ImageFolder(split_dirs["test"], transform=build_transforms(False)),
    }
    classes = folders["train"].classes
    for key, folder in folders.items():
        if folder.classes != classes:
            raise SystemExit(f"{key} class order {folder.classes} != train {classes}")
    return folders


def dataloader_settings() -> tuple[int, bool]:
    """Worker count and pin_memory from the same flags the Colab Trainer uses."""
    flags = gpu_train_flags()
    workers = int(flags[flags.index("--dataloader_num_workers") + 1])
    pin = flags[flags.index("--dataloader_pin_memory") + 1].lower() == "true"
    return workers, pin


def build_dataloaders(
    root: str | Path,
    batch_size: int = 64,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> dict[str, DataLoader]:
    """Train/validation/test loaders. num_workers defaults to gpu_train_flags()."""
    folders = build_imagefolders(root)
    default_workers, default_pin = dataloader_settings()
    workers = default_workers if num_workers is None else num_workers
    pin = default_pin if pin_memory is None else pin_memory
    loaders = {}
    for key, folder in folders.items():
        loaders[key] = DataLoader(
            folder,
            batch_size=batch_size,
            shuffle=key == "train",
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=workers > 0,
        )
    return loaders


def build_hf_datasets(root: str | Path, crop_filter: str | None, max_samples_per_class: int | None, seed: int):
    """Hugging Face datasets keyed train/validation/test, for transformers.Trainer."""
    from datasets import Dataset, Features, Image as HFImage, Value

    label_names, rows, split_dirs = collect_split_rows(
        root,
        crop_filter=crop_filter,
        max_samples_per_class=max_samples_per_class,
        seed=seed,
    )
    dataset_root = split_dirs["train"].parent
    features = Features(
        {
            "image": HFImage(),
            "label": Value("int64"),
            "path": Value("string"),
            "source": Value("string"),
        }
    )
    datasets = {}
    for key, split_rows in rows.items():
        datasets[key] = Dataset.from_dict(
            {
                "image": [str(dataset_root / rel) for rel, _ in split_rows],
                "label": [label for _, label in split_rows],
                "path": [rel for rel, _ in split_rows],
                "source": ["masterdataset"] * len(split_rows),
            },
            features=features,
        )
    return label_names, rows, datasets, dataset_root
