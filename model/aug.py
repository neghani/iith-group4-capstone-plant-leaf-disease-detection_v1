"""Training and eval image transforms. Torchvision only so EDA/tests can import without Hugging Face."""

from __future__ import annotations

from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    GaussianBlur,
    Normalize,
    RandomApply,
    RandomErasing,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

DEFAULT_TRAIN_OPS = (
    "RandomResizedCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "ColorJitter",
    "RandomApply",
    "ToTensor",
    "Normalize",
)


def _size_value(size, *keys: str) -> int | None:
    if size is None:
        return None
    if isinstance(size, (int, float)):
        return int(size)
    for key in keys:
        val = size.get(key) if isinstance(size, dict) else getattr(size, key, None)
        if val:
            return int(val)
    return None


def _processor_sizes(image_processor) -> tuple[int, int]:
    crop = _size_value(getattr(image_processor, "crop_size", None), "height", "shortest_edge") or _size_value(
        getattr(image_processor, "size", None), "height", "shortest_edge"
    ) or 224
    resize = _size_value(getattr(image_processor, "size", None), "shortest_edge", "height") or crop
    return int(resize), int(crop)


def _norm_aug_policy(policy: str | None) -> str:
    value = (policy or "default").strip().lower()
    if value not in ("default", "robust"):
        raise SystemExit(f"Unknown aug_policy={policy!r}; expected 'default' or 'robust'.")
    return value


def transform_op_names(transform: Compose) -> list[str]:
    return [type(step).__name__ for step in transform.transforms]


def augmentation_record(policy: str = "default") -> dict:
    policy = _norm_aug_policy(policy)
    ops = list(DEFAULT_TRAIN_OPS)
    if policy == "robust":
        ops.append("RandomErasing")
    return {
        "policy": policy,
        "pv_rrc_scale": [0.7, 1.0],
        "pd_rrc_scale": [0.5, 1.0],
        "ops": ops,
    }


def build_transforms(image_processor, crop_size: int | None, is_training: bool,
                     rrc_scale: tuple[float, float] = (0.7, 1.0),
                     aug_policy: str = "default") -> Compose:
    resize_size, proc_crop = _processor_sizes(image_processor)
    crop = crop_size or proc_crop
    mean = getattr(image_processor, "image_mean", None) or [0.485, 0.456, 0.406]
    std = getattr(image_processor, "image_std", None) or [0.229, 0.224, 0.225]
    normalize = Normalize(mean=mean, std=std)
    if is_training:
        ops = [
            RandomResizedCrop(crop, scale=rrc_scale, ratio=(0.8, 1.25)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(degrees=180),
            ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.03),
            RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.25),
            ToTensor(),
            normalize,
        ]
        if _norm_aug_policy(aug_policy) == "robust":
            ops.append(RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0))
        return Compose(ops)
    return Compose([Resize(resize_size), CenterCrop(crop), ToTensor(), normalize])
