"""Phase 1: fine-tune MobileNetV2 (or any AutoModelForImageClassification) on PlantVillage.

Adapted from the Hugging Face image-classification Trainer template. Differences:
- reads a local imagefolder (`--data_dir`, the PlantVillage `color` folder) instead of a Hub dataset
- deterministic stratified train/validation/test split saved to `splits.json`
- augmentations for tilted/inverted/blurry/low-light phone photos
- metrics: accuracy, macro F1, top-3 accuracy
- `--crop_filter Tomato` and `--max_samples_per_class` for fast first experiments

Example (Colab T4, full dataset):
    python model/train.py --data_dir "$DATA_DIR" --output_dir model/outputs/plantvillage-mobilenetv2 \
        --num_train_epochs 5 --per_device_train_batch_size 64 --learning_rate 5e-4 --fp16 \
        --push_to_hub --hub_model_id neghani/plantvillage-mobilenetv2
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, Features, Image as HFImage, Value
from sklearn.metrics import accuracy_score, f1_score
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    GaussianBlur,
    Normalize,
    RandomApply,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_HF_MODEL_ID,
    resolve_data_dir,
    save_splits,
    scan_imagefolder,
    stratified_split,
)

logger = logging.getLogger("phase1.train")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=DEFAULT_BASE_MODEL,
        metadata={"help": "Base checkpoint. MobileNetV2 by default; try microsoft/resnet-50 as the fallback."},
    )
    ignore_mismatched_sizes: bool = field(default=True, metadata={"help": "Replace the ImageNet head with a PlantVillage head."})
    token: str | None = field(default=None, metadata={"help": "HF token for private base models (normally not needed)."})


@dataclass
class DataArguments:
    data_dir: str = field(
        default="model/data/plantvillage",
        metadata={"help": "PlantVillage 'color' folder (or the kagglehub download root; it is auto-resolved)."},
    )
    crop_filter: str | None = field(
        default=None,
        metadata={"help": "Comma-separated crop names to keep, e.g. 'Tomato' or 'Tomato,Potato'. Default: all 38 classes."},
    )
    max_samples_per_class: int | None = field(default=None, metadata={"help": "Cap images per class (smoke tests)."})
    val_fraction: float = field(default=0.10, metadata={"help": "Validation share per class."})
    test_fraction: float = field(default=0.10, metadata={"help": "Held-out test share per class (used by evaluate.py)."})
    image_size: int | None = field(default=None, metadata={"help": "Training crop size; defaults to the image processor's crop size (224)."})
    max_eval_samples: int | None = field(default=None, metadata={"help": "Truncate validation set during training (smoke tests)."})


def _size_value(size, *keys: str) -> int | None:
    """Read a size from an int, a dict, or a transformers v5 SizeDict."""
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
    """Return (resize_shortest_edge, crop_size) consistent with the processor's eval preprocessing."""
    crop = _size_value(getattr(image_processor, "crop_size", None), "height", "shortest_edge") or _size_value(
        getattr(image_processor, "size", None), "height", "shortest_edge"
    ) or 224
    resize = _size_value(getattr(image_processor, "size", None), "shortest_edge", "height") or crop
    return int(resize), int(crop)


def build_transforms(image_processor, crop_size: int | None, is_training: bool) -> Compose:
    resize_size, proc_crop = _processor_sizes(image_processor)
    crop = crop_size or proc_crop
    mean = getattr(image_processor, "image_mean", None) or [0.485, 0.456, 0.406]
    std = getattr(image_processor, "image_std", None) or [0.229, 0.224, 0.225]
    normalize = Normalize(mean=mean, std=std)

    if is_training:
        # Phone photos are tilted, inverted, partially framed, blurry or badly lit.
        return Compose(
            [
                RandomResizedCrop(crop, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomRotation(degrees=180),
                ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.03),
                RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.25),
                ToTensor(),
                normalize,
            ]
        )
    # Mirrors AutoImageProcessor: resize shortest edge, center crop, normalize.
    return Compose([Resize(resize_size), CenterCrop(crop), ToTensor(), normalize])


def build_datasets(data_dir: Path, data_args: DataArguments, seed: int):
    label_names, items = scan_imagefolder(
        data_dir, crop_filter=data_args.crop_filter, max_samples_per_class=data_args.max_samples_per_class, seed=seed
    )
    splits = stratified_split(items, data_args.val_fraction, data_args.test_fraction, seed=seed)
    features = Features({"image": HFImage(), "label": Value("int64"), "path": Value("string")})

    def to_dataset(rows: list[tuple[str, int]]) -> Dataset:
        return Dataset.from_dict(
            {
                "image": [str(data_dir / p) for p, _ in rows],
                "label": [l for _, l in rows],
                "path": [p for p, _ in rows],
            },
            features=features,
        )

    datasets = {k: to_dataset(v) for k, v in splits.items() if v}
    return label_names, splits, datasets


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    k = min(3, logits.shape[-1])
    topk = np.argsort(-logits, axis=-1)[:, :k]
    top3 = float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "top3_accuracy": top3,
    }


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # set_transform() produces pixel_values on the fly; Trainer must not drop the raw columns.
    training_args.remove_unused_columns = False
    if training_args.do_train and training_args.resume_from_checkpoint is None:
        # fresh run: clear stale artifacts (reports, old README) from a previous run in the same dir
        stale = Path(training_args.output_dir)
        if stale.exists():
            for name in ("README.md", "splits.json", "run_info.json", "all_results.json", "train_results.json", "eval_results.json", "test_results.json"):
                (stale / name).unlink(missing_ok=True)
    if training_args.push_to_hub and not training_args.hub_model_id:
        training_args.hub_model_id = DEFAULT_HF_MODEL_ID
    if training_args.metric_for_best_model is None and training_args.load_best_model_at_end:
        training_args.metric_for_best_model = "macro_f1"
        training_args.greater_is_better = True

    hf_token = os.environ.get("HF_TOKEN")
    if training_args.push_to_hub:
        if hf_token and not training_args.hub_token:
            training_args.hub_token = hf_token
        if not training_args.hub_token:
            logger.warning("push_to_hub is set but no HF token found (HF_TOKEN env or `hf auth login`).")

    data_dir = resolve_data_dir(data_args.data_dir)
    logger.info("Data dir: %s", data_dir)

    label_names, splits, datasets = build_datasets(data_dir, data_args, training_args.seed)
    num_labels = len(label_names)
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in id2label.items()}
    logger.info("Classes (%d): %s", num_labels, ", ".join(label_names))
    logger.info("Split sizes: %s", {k: len(v) for k, v in splits.items()})

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_splits(
        output_dir / "splits.json",
        data_dir,
        label_names,
        splits,
        extra={
            "seed": training_args.seed,
            "crop_filter": data_args.crop_filter,
            "max_samples_per_class": data_args.max_samples_per_class,
            "base_model": model_args.model_name_or_path,
        },
    )

    pretrained_kwargs = {"token": model_args.token} if model_args.token else {}
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, num_labels=num_labels, id2label=id2label, label2id=label2id, **pretrained_kwargs
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        **pretrained_kwargs,
    )
    image_processor = AutoImageProcessor.from_pretrained(model_args.model_name_or_path, **pretrained_kwargs)

    train_tf = build_transforms(image_processor, data_args.image_size, is_training=True)
    eval_tf = build_transforms(image_processor, data_args.image_size, is_training=False)

    def preprocess_train(examples):
        return {"pixel_values": [train_tf(img.convert("RGB")) for img in examples["image"]], "labels": examples["label"]}

    def preprocess_eval(examples):
        return {"pixel_values": [eval_tf(img.convert("RGB")) for img in examples["image"]], "labels": examples["label"]}

    datasets["train"].set_transform(preprocess_train)
    for split_name in ("validation", "test"):
        if split_name in datasets:
            datasets[split_name].set_transform(preprocess_eval)

    eval_dataset = datasets.get("validation")
    if eval_dataset is not None and data_args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(data_args.max_eval_samples, len(eval_dataset))))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"] if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    started = time.time()
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        if eval_dataset is not None:
            metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        if "test" in datasets:
            metrics = trainer.evaluate(eval_dataset=datasets["test"], metric_key_prefix="test")
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

    run_info = {
        "base_model": model_args.model_name_or_path,
        "num_labels": num_labels,
        "label_names": label_names,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "crop_filter": data_args.crop_filter,
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "weight_decay": training_args.weight_decay,
        "warmup_steps": training_args.warmup_steps,
        "lr_scheduler_type": str(training_args.lr_scheduler_type),
        "fp16": training_args.fp16,
        "seed": training_args.seed,
        "wall_time_sec": round(time.time() - started, 1),
        "device": str(training_args.device),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    card_kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": "PlantVillage",
        "tasks": "image-classification",
        "tags": ["image-classification", "plant-disease", "plantvillage", "mobilenetv2", "agriculture"],
    }
    if training_args.push_to_hub:
        logger.info("Pushing to https://huggingface.co/%s", training_args.hub_model_id)
        trainer.push_to_hub(**card_kwargs)
    else:
        trainer.create_model_card(**card_kwargs)
    logger.info("Done. Artifacts in %s", output_dir)


if __name__ == "__main__":
    main()
