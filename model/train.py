"""Fine-tune MobileNetV2 on PlantVillage, optionally continuing from v1.0 on PlantDoc.

Stage 1 (from ImageNet):
    python model/train.py --data_dir "$DATA_DIR" --output_dir model/outputs/plantvillage-mobilenetv2 ...

Stage 2 (from published v1.0, mixed PlantVillage + Cropped-PlantDoc):
    python model/train.py --init_from neghani/plantvillage-mobilenetv2 --model_revision v1.0 \
        --data_dir "$DATA_DIR" --extra_data plantdoc --plantdoc_fraction 0.5 --epoch_samples 4000 \
        --learning_rate 1e-4 --num_train_epochs 8 --label_smoothing_factor 0.1 \
        --output_dir /content/drive/MyDrive/plant-disease/stage2/lr1e-4 \
        --resume_from_checkpoint auto

The checkpoint is chosen by PlantDoc validation macro F1. PlantVillage accuracy is logged
every epoch and never used to select. Real training runs on Colab (the notebook); locally use

    python -m model.train --smoke

which runs the whole chain on synthetic data on CPU (see model/smoke.py).
"""

from __future__ import annotations

import csv
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
from torch.utils.data import Sampler
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
    get_hf_token,
    resolve_data_dir,
    save_splits,
    scan_imagefolder,
    stratified_split,
)
from data_protocol import (  # noqa: E402
    Item,
    MATCH_DIST,
    PINNED_COMMIT,
    assert_protocol,
    load_hash_cache,
    phash_cached,
    save_hash_cache,
    write_manifest,
)
from aug import augmentation_record, build_transforms, _norm_aug_policy  # noqa: E402
from sampling import plan_global  # noqa: E402

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("phase1.train")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=DEFAULT_BASE_MODEL,
        metadata={"help": "Base checkpoint. Overridden by --init_from when that is set."},
    )
    init_from: str | None = field(
        default=None,
        metadata={"help": "Starting weights (e.g. neghani/plantvillage-mobilenetv2). Overrides model_name_or_path."},
    )
    model_revision: str | None = field(default=None, metadata={"help": "Hub revision/tag of the starting weights (e.g. v1.0)."})
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
    extra_data: str | None = field(
        default=None,
        metadata={"help": "'plantdoc' (model/data/plantdoc_train) or a folder written by prepare_plantdoc.py."},
    )
    plantdoc_fraction: float = field(
        default=0.5,
        metadata={"help": "Share of each epoch drawn from PlantDoc (the rest is PlantVillage)."},
    )
    epoch_samples: int = field(
        default=4000,
        metadata={"help": "Fixed number of mixed draws per epoch."},
    )
    artifacts_dir: str = field(
        default=str(Path(__file__).resolve().parent / "outputs"),
        metadata={"help": "Where data_manifest.json and the PlantVillage pHash cache are written."},
    )
    aug_policy: str = field(
        default="default",
        metadata={
            "help": "Training augmentation: 'default' (current stack) or 'robust' (adds RandomErasing). "
            "Changing this on a resumed run is a new experiment, not a resume."
        },
    )


def code_commit() -> dict:
    """Repo commit this run used: CODE_COMMIT (set by the notebook) or `git rev-parse HEAD`."""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    sha = os.environ.get("CODE_COMMIT", "")
    if not sha:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        sha = (r.stdout or "").strip()
    r = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                       capture_output=True, text=True)
    return {"code_commit": sha or None, "code_dirty": bool((r.stdout or "").strip()) if r.returncode == 0 else None}


def extra_dir_of(value: str | None) -> Path | None:
    if not value:
        return None
    if value in ("plantdoc", "plantdoc_train"):
        return HERE / "data" / "plantdoc_train"
    return Path(value)


def latest_checkpoint(output_dir: Path) -> str | None:
    ckpts = []
    for p in output_dir.glob("checkpoint-*"):
        try:
            ckpts.append((int(p.name.split("-")[1]), p))
        except (IndexError, ValueError):
            continue
    return str(sorted(ckpts)[-1][1]) if ckpts else None


def has_weights(output_dir: Path) -> bool:
    return any(output_dir.glob("*.safetensors")) or any(output_dir.glob("pytorch_model*.bin"))


def training_finished(output_dir: Path, num_epochs: float, max_steps: int | None) -> bool:
    """True when this output_dir already holds a completed run (re-running the cell must not start over)."""
    if not has_weights(output_dir):
        return False
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        return True  # saved model, no trainer state: treat as done
    state = json.loads(state_path.read_text())
    step = int(state.get("global_step") or 0)
    if max_steps and max_steps > 0:
        return step >= int(max_steps)
    epoch = float(state.get("epoch") or 0)
    return epoch + 1e-6 >= float(num_epochs)


def load_extra(extra_dir: Path, label_names: list[str]) -> tuple[list[tuple[str, int]], list[tuple[str, int]], dict):
    manifest = extra_dir / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found; run `python model/prepare_plantdoc.py` first.")
    ids = {n: i for i, n in enumerate(label_names)}
    train, val, skipped = [], [], {}
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"] not in ids:
                skipped[r["label"]] = skipped.get(r["label"], 0) + 1
                continue
            path = extra_dir / r["file"]
            if not path.exists():
                raise SystemExit(f"{path} listed in {manifest} is missing; rerun prepare_plantdoc.py.")
            (val if r["split"] == "validation" else train).append((str(path), ids[r["label"]]))
    return train, val, {"train": len(train), "validation": len(val), "skipped_labels": skipped, "dir": str(extra_dir)}


def items_from_prepared(extra_dir: Path) -> dict[str, list[Item]]:
    from prepare_plantdoc import items_from_json

    path = extra_dir / "items.json"
    if not path.exists():
        raise SystemExit(f"{path} missing; rerun prepare_plantdoc.py.")
    return items_from_json(path)


def assert_training_protocol(data_dir: Path, pv_train: list[tuple[str, int]], extra_dir: Path, cache_path: Path) -> None:
    prepared = items_from_prepared(extra_dir)
    cache = load_hash_cache(cache_path)
    extra_items = []
    for rel, _ in pv_train:
        path = data_dir / rel
        extra_items.append(Item(path=str(path), split="train", label=Path(rel).parts[0],
                                phash=phash_cached(path, cache),
                                group=f"pv:{rel}", source="plantvillage"))
    save_hash_cache(cache_path, cache)
    assert_protocol(prepared["train"], prepared["validation"], prepared["final_test"], extra_items)
    logger.info("Data-protocol assertions passed (train/val vs final test, train vs val, PlantVillage vs final test).")


class MixedIndexSampler(Sampler[int]):
    """Thin torch wrapper around sampling.plan_global. set_epoch keeps resume deterministic."""

    def __init__(self, pv_indices: list[int], pd_indices: list[int], pd_labels: list[int],
                 fraction: float, epoch_samples: int, seed: int):
        self.pv_indices = pv_indices
        self.pd_indices = pd_indices
        self.pd_labels = pd_labels
        self.fraction = fraction
        self.epoch_samples = epoch_samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(plan_global(self.pv_indices, self.pd_indices, self.pd_labels,
                                self.fraction, self.epoch_samples, self.seed, self.epoch))

    def __len__(self) -> int:
        return self.epoch_samples


class MixedTrainer(Trainer):
    def __init__(self, *args, mixed_sampler: MixedIndexSampler | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mixed_sampler = mixed_sampler

    def _get_train_sampler(self, *args, **kwargs):
        if self._mixed_sampler is not None:
            if hasattr(self.state, "epoch") and self.state.epoch is not None:
                self._mixed_sampler.set_epoch(int(self.state.epoch))
            return self._mixed_sampler
        return super()._get_train_sampler(*args, **kwargs)


def build_datasets(data_dir: Path, data_args: DataArguments, seed: int):
    label_names, items = scan_imagefolder(
        data_dir, crop_filter=data_args.crop_filter, max_samples_per_class=data_args.max_samples_per_class, seed=seed
    )
    splits = stratified_split(items, data_args.val_fraction, data_args.test_fraction, seed=seed)
    features = Features({
        "image": HFImage(), "label": Value("int64"), "path": Value("string"), "source": Value("string"),
    })

    def to_dataset(rows: list[tuple[str, int]], source: str, root: Path | None = None) -> Dataset:
        base = root or data_dir
        return Dataset.from_dict(
            {
                "image": [str(base / p) if not Path(p).is_absolute() else p for p, _ in rows],
                "label": [l for _, l in rows],
                "path": [p for p, _ in rows],
                "source": [source] * len(rows),
            },
            features=features,
        )

    datasets = {k: to_dataset(v, "plantvillage") for k, v in splits.items() if v}
    extra_info = None
    extra_dir = extra_dir_of(data_args.extra_data)
    sampler_spec = None
    if extra_dir:
        extra_train, extra_val, extra_info = load_extra(extra_dir, label_names)
        if not extra_train:
            raise SystemExit("No extra training rows left after mapping to the model's labels.")
        pv_n = len(datasets["train"])
        pd_train = to_dataset(extra_train, "plantdoc", root=Path("."))
        # Concatenate so the sampler can index into one dataset: [PV | PD].
        from datasets import concatenate_datasets

        datasets["train"] = concatenate_datasets([datasets["train"], pd_train])
        if extra_val:
            datasets["extra_validation"] = to_dataset(extra_val, "plantdoc", root=Path("."))
        extra_info["fraction"] = data_args.plantdoc_fraction
        extra_info["epoch_samples"] = data_args.epoch_samples
        sampler_spec = {
            "pv_indices": list(range(pv_n)),
            "pd_indices": list(range(pv_n, pv_n + len(extra_train))),
            "pd_labels": [lab for _, lab in extra_train],
        }
    return label_names, splits, datasets, extra_info, extra_dir, sampler_spec


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


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--smoke" in argv:
        from smoke import run_all

        run_all()
        return
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=argv)
    data_args.aug_policy = _norm_aug_policy(data_args.aug_policy)
    artifacts = Path(data_args.artifacts_dir)
    if os.environ.get("MODEL_SMOKE"):
        from smoke import assert_smoke_paths

        assert_smoke_paths(artifacts, Path(training_args.output_dir))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    if model_args.init_from:
        model_args.model_name_or_path = model_args.init_from

    output_dir = Path(training_args.output_dir)
    overwrite = bool(getattr(training_args, "overwrite_output_dir", False))
    resume = training_args.resume_from_checkpoint
    found = latest_checkpoint(output_dir)
    if isinstance(resume, str) and resume.lower() == "auto":
        training_args.resume_from_checkpoint = found
        logger.info("resume_from_checkpoint=auto -> %s", found or "(none, starting fresh)")
    elif training_args.do_train and found and not resume and not overwrite:
        # Re-running the Colab train cell after a disconnect must not start from scratch.
        training_args.resume_from_checkpoint = found
        logger.warning("Found %s; resuming instead of training from scratch. "
                       "Pass --overwrite_output_dir to start over.", found)

    if training_args.do_train and not overwrite and training_finished(
        output_dir, training_args.num_train_epochs, getattr(training_args, "max_steps", None)
    ):
        logger.info("Training already finished in %s (weights present). Skipping --do_train. "
                    "Pass --overwrite_output_dir to retrain.", output_dir)
        training_args.do_train = False
    if not training_args.do_train and has_weights(output_dir):
        logger.info("Loading finished weights from %s (not the base checkpoint).", output_dir)
        model_args.model_name_or_path = str(output_dir)
        model_args.init_from = None
        model_args.model_revision = None

    training_args.remove_unused_columns = False
    resuming = bool(training_args.resume_from_checkpoint)
    if training_args.do_train and not resuming:
        if output_dir.exists():
            for name in ("README.md", "splits.json", "run_info.json", "all_results.json",
                         "train_results.json", "eval_results.json", "test_results.json"):
                (output_dir / name).unlink(missing_ok=True)
    if training_args.push_to_hub and not training_args.hub_model_id:
        training_args.hub_model_id = DEFAULT_HF_MODEL_ID
    if training_args.load_best_model_at_end and training_args.metric_for_best_model in (None, "macro_f1"):
        training_args.metric_for_best_model = "plantdoc_macro_f1" if data_args.extra_data else "macro_f1"
        training_args.greater_is_better = True

    hf_token = get_hf_token()
    if training_args.push_to_hub:
        if hf_token and not training_args.hub_token:
            training_args.hub_token = hf_token
        if not training_args.hub_token:
            logger.warning("push_to_hub is set but no HF token found (HF_TOKEN env or `hf auth login`).")

    data_dir = resolve_data_dir(data_args.data_dir)
    logger.info("Data dir: %s", data_dir)
    if "/content/drive/" in str(data_dir.resolve()).replace("\\", "/"):
        logger.warning(
            "data_dir is on Google Drive (%s). Loading many small images through the Drive "
            "mount is slow. Zip the dataset, copy it to the Colab SSD, and unzip locally.",
            data_dir,
        )

    label_names, splits, datasets, extra_info, extra_dir, sampler_spec = build_datasets(
        data_dir, data_args, training_args.seed
    )
    if extra_dir:
        assert_training_protocol(data_dir, splits["train"], extra_dir, artifacts / "pv_phash_cache.json")
        # Keep exclusion counts from prepare; add PlantVillage split sizes.
        prepared = items_from_prepared(extra_dir)
        prep = {}
        if (extra_dir / "prep_report.json").exists():
            prep = json.loads((extra_dir / "prep_report.json").read_text())
        write_manifest(
            artifacts / "data_manifest.json",
            commit=prep.get("plantdoc_commit", ""),
            pinned=PINNED_COMMIT,
            exclusions=[],
            groups={
                "plantdoc_train": prepared["train"],
                "plantdoc_validation": prepared["validation"],
                "final_test": prepared["final_test"],
                "plantvillage_train": [Item(path=p, split="train", label=label_names[i], phash=0,
                                            group=f"pv:{p}", source="plantvillage")
                                       for p, i in splits["train"]],
            },
            params={"phash_dist": MATCH_DIST, "plantdoc_fraction": data_args.plantdoc_fraction,
                    "epoch_samples": data_args.epoch_samples},
            extra={"plantvillage_counts": {k: len(v) for k, v in splits.items()},
                   "exclusions": prep.get("exclusions", {})},
        )
    num_labels = len(label_names)
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in id2label.items()}
    logger.info("Classes (%d): %s", num_labels, ", ".join(label_names))
    logger.info("Split sizes (PlantVillage): %s", {k: len(v) for k, v in splits.items()})
    if extra_info:
        logger.info("Extra data: %s; mixed epoch %d draws, PlantDoc fraction %.2f",
                    extra_info, data_args.epoch_samples, data_args.plantdoc_fraction)

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
            "base_revision": model_args.model_revision,
            "extra_data": extra_info,
            "plantdoc_fraction": data_args.plantdoc_fraction if extra_info else None,
            "epoch_samples": data_args.epoch_samples if extra_info else None,
        },
    )

    pretrained_kwargs = {"token": model_args.token} if model_args.token else {}
    if model_args.model_revision:
        pretrained_kwargs["revision"] = model_args.model_revision
    base_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **pretrained_kwargs)
    base_labels = [base_config.id2label[i] for i in range(len(base_config.id2label))]
    if len(base_labels) == num_labels and base_labels != label_names:
        raise SystemExit(
            "The starting model already has a 38-class head whose label order differs from the training folders; "
            "continuing would scramble it. Compare id2label in the model's config.json with the data_dir class folders."
        )
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

    pv_tf = build_transforms(
        image_processor, data_args.image_size, is_training=True, rrc_scale=(0.7, 1.0),
        aug_policy=data_args.aug_policy,
    )
    pd_tf = build_transforms(
        image_processor, data_args.image_size, is_training=True, rrc_scale=(0.5, 1.0),
        aug_policy=data_args.aug_policy,
    )
    eval_tf = build_transforms(image_processor, data_args.image_size, is_training=False)

    def preprocess_train(examples):
        out = []
        for img, src in zip(examples["image"], examples["source"]):
            tf = pd_tf if src == "plantdoc" else pv_tf
            out.append(tf(img.convert("RGB")))
        return {"pixel_values": out, "labels": examples["label"]}

    def preprocess_eval(examples):
        return {"pixel_values": [eval_tf(img.convert("RGB")) for img in examples["image"]], "labels": examples["label"]}

    datasets["train"].set_transform(preprocess_train)
    for split_name in ("validation", "extra_validation", "test"):
        if split_name in datasets:
            datasets[split_name].set_transform(preprocess_eval)

    def cap(ds):
        return ds.select(range(min(data_args.max_eval_samples, len(ds)))) if ds is not None and data_args.max_eval_samples else ds

    if "extra_validation" in datasets:
        eval_dataset = {"plantvillage": cap(datasets.get("validation")), "plantdoc": cap(datasets["extra_validation"])}
        eval_dataset = {k: v for k, v in eval_dataset.items() if v is not None}
    else:
        eval_dataset = cap(datasets.get("validation"))

    mixed_sampler = None
    if sampler_spec:
        mixed_sampler = MixedIndexSampler(
            sampler_spec["pv_indices"], sampler_spec["pd_indices"], sampler_spec["pd_labels"],
            data_args.plantdoc_fraction, data_args.epoch_samples, training_args.seed,
        )

    trainer_cls = MixedTrainer if mixed_sampler else Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=datasets["train"] if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )
    trainer = trainer_cls(**trainer_kwargs, mixed_sampler=mixed_sampler) if mixed_sampler else Trainer(**trainer_kwargs)

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

    plantdoc_commit = ""
    if extra_dir and (extra_dir / "prep_report.json").exists():
        plantdoc_commit = json.loads((extra_dir / "prep_report.json").read_text()).get("plantdoc_commit", "")
    run_info = {
        "base_model": model_args.model_name_or_path,
        "base_revision": model_args.model_revision,
        "init_from": model_args.init_from or model_args.model_name_or_path,
        "extra_data": extra_info,
        "plantdoc_fraction": data_args.plantdoc_fraction if extra_info else None,
        "epoch_samples": data_args.epoch_samples if extra_info else None,
        "plantdoc_commit": plantdoc_commit,
        "plantdoc_commit_pinned": PINNED_COMMIT if extra_info else None,
        "data_manifest": (json.loads((artifacts / "data_manifest.json").read_text())
                          if extra_info and (artifacts / "data_manifest.json").exists() else None),
        **code_commit(),
        "augmentation": augmentation_record(data_args.aug_policy),
        "label_smoothing_factor": training_args.label_smoothing_factor,
        "train_rows": len(datasets["train"]),
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
        "bf16": getattr(training_args, "bf16", False),
        "dataloader_num_workers": training_args.dataloader_num_workers,
        "dataloader_pin_memory": training_args.dataloader_pin_memory,
        "seed": training_args.seed,
        "wall_time_sec": round(time.time() - started, 1),
        "device": str(training_args.device),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    card_kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": ["PlantVillage", "PlantDoc"] if extra_info else "PlantVillage",
        "tasks": "image-classification",
        "tags": ["image-classification", "plant-disease", "plantvillage", "mobilenetv2", "agriculture"]
        + (["plantdoc"] if extra_info else []),
    }
    if training_args.push_to_hub:
        logger.info("Pushing to https://huggingface.co/%s", training_args.hub_model_id)
        trainer.push_to_hub(**card_kwargs)
    else:
        trainer.create_model_card(**card_kwargs)
    logger.info("Done. Artifacts in %s", output_dir)


if __name__ == "__main__":
    main()
