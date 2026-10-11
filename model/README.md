# Phase 1: Disease detection model

Train MobileNetV2 on PlantVillage, evaluate it, test it on real-world photos, and publish it to
Hugging Face at **https://huggingface.co/neghani/plantvillage-mobilenetv2** for Phase 5.

Owners: Ganesh, Shravanya (proposal Annex A). Hub owner: `neghani`.

## Deliverables (proposal Section 9, Phase 1)

| Proposal activity | Where it is done |
|---|---|
| Train MobileNetV2 on PlantVillage in Colab | [`train.py`](train.py), [`../notebooks/phase1_plantvillage_mobilenetv2.ipynb`](../notebooks/phase1_plantvillage_mobilenetv2.ipynb) |
| Handle tilted and inverted images, scale inputs | Augmentations in `train.py` (`build_transforms`), EXIF orientation fix in `common.load_image` |
| Image quality check that suggests a better photo | `common.check_image_quality`, used by [`predict.py`](predict.py) and `real_world_eval.py`; same thresholds as `api/app/vision/quality.py` |
| Test on real-world images, target at least 50% correct | [`real_world_eval.py`](real_world_eval.py), images under [`data/real_world/`](data/real_world/README.md) |
| Report accuracy and F1 (plus per-class precision/recall, confusion matrix, top-3, threshold sweep) | [`evaluate.py`](evaluate.py) -> `outputs/<run>/reports/` |
| Try ResNet if MobileNetV2 falls short | `--model_name_or_path microsoft/resnet-50` (see below) |
| Deploy the model to Hugging Face, hand over to Phase 5 | `--push_to_hub` in `train.py`, `--push` in the evaluation scripts, tag `v1.0`, handover contract below |
| Record challenges and approach | "Challenges and approach" section below |

## Files

```text
model/
├── README.md                 this guide
├── requirements.txt          torch, transformers, datasets, scikit-learn, kagglehub, ...
├── common.py                 class names, quality gate, dataset scan/split, Predictor
├── train.py                  fine-tune (HF Trainer), writes splits.json + run_info.json, push_to_hub
├── evaluate.py               test-split metrics, confusion matrix, threshold sweep, model card
├── real_world_eval.py        internet-photo check against the 50% target
├── predict.py                Phase 5 reference inference (quality gate -> model -> not sure)
├── model_card_template.md    README pushed to the Hub, filled by evaluate.py
├── data/
│   ├── plantvillage/         (git-ignored) optional local copy of the Kaggle 'color' folder
│   └── real_world/           <ClassName>/*.jpg (images git-ignored), sources.csv
└── outputs/                  (git-ignored) trained models and reports
```

## Prerequisites

- Hugging Face account `neghani` with a **write** token: https://huggingface.co/settings/tokens
  - Colab: add it as secret `HF_TOKEN`. Locally: `hf auth login` or `export HF_TOKEN=...`.
- Kaggle dataset [`abdallahalidev/plantvillage-dataset`](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
  (public; `kagglehub` downloads it without credentials in most cases. If prompted, set
  `KAGGLE_USERNAME` / `KAGGLE_KEY`).
- GPU: Colab T4 is enough (~8-10 min per epoch on all 54k images with batch 64, fp16).

## Local vs Colab testing (do not upload files)

Edit code locally. Train on Colab. After every `git push`, **replace** Colab files from GitHub.
Do not use `files.upload()` for `.py` files — Colab often keeps the old copy and you re-test stale code.

| Step | Local (CPU) | Colab (T4 GPU) |
|---|---|---|
| Edit `model/*.py` | your editor | never; push first |
| Refresh code | n/a | re-run `notebooks/colab_sync.py` (hard-resets to `origin/main`) |
| Refresh one file | n/a | `python notebooks/colab_sync.py --file model/train.py` |
| Smoke test | tomato, 40 images/class, 1 epoch | optional; same flags on GPU |
| Full 38-class train | only if you have a GPU | recommended |
| Real-world images | `model/data/real_world/<Class>/` | Drive zip at `MyDrive/plant-disease/real_world.zip`, always extract-replace |

```bash
# Local: iterate, then push
python model/train.py --data_dir "$DATA_DIR" --crop_filter Tomato --max_samples_per_class 40 \
  --output_dir model/outputs/smoke --do_train --do_eval --num_train_epochs 1 \
  --save_strategy no --report_to none
git add model notebooks && git commit && git push

# Colab: always replace tracked files (keeps model/outputs and real_world images)
python notebooks/colab_sync.py
# or one file only:
python notebooks/colab_sync.py --file model/train.py --file model/evaluate.py
```

`colab_sync.py` refuses to `git reset --hard` on a local checkout unless you pass `--force`.

## Run in Colab (recommended)

Open [`notebooks/phase1_plantvillage_mobilenetv2.ipynb`](../notebooks/phase1_plantvillage_mobilenetv2.ipynb)
in Colab (File -> Open notebook -> GitHub -> paste the repo URL), select a GPU runtime and run the
cells top to bottom:

1. Sync repo from GitHub (`colab_sync.py` always replaces tracked files), `pip install -r model/requirements.txt`
2. Log in to HF from the Colab secret; the notebook checks you are `neghani` (or that you set `HF_MODEL_ID` to your own namespace)
3. `kagglehub.dataset_download(...)` and resolve the `color` folder
4. Optional smoke test (tomato classes, 60 images per class, 1 epoch)
5. Full training with `--push_to_hub` (model lands on the Hub as soon as training ends)
6. `evaluate.py --push` -> reports + model card on the Hub
7. Replace real-world images from the Drive zip, `real_world_eval.py --push`
8. Tag `v1.0`
9. Inference demo from the Hub
10. Handover values for Phase 5

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r model/requirements.txt

# dataset
python - <<'EOF'
import kagglehub; print(kagglehub.dataset_download("abdallahalidev/plantvillage-dataset"))
EOF
export DATA_DIR=<printed path>   # train.py finds the 'color' subfolder automatically

# smoke test (CPU ok, ~2-3 min): tomato only, 40 images per class
python model/train.py --data_dir "$DATA_DIR" --crop_filter Tomato --max_samples_per_class 40 \
  --output_dir model/outputs/smoke --do_train --do_eval \
  --num_train_epochs 1 --per_device_train_batch_size 16 --learning_rate 5e-4 \
  --report_to none --save_strategy no
python model/evaluate.py --model model/outputs/smoke
python model/predict.py some_leaf.jpg --model model/outputs/smoke --revision main
```

Full run (GPU):

```bash
python model/train.py \
  --model_name_or_path google/mobilenet_v2_1.0_224 \
  --data_dir "$DATA_DIR" \
  --output_dir model/outputs/plantvillage-mobilenetv2 \
  --do_train --do_eval --num_train_epochs 5 \
  --per_device_train_batch_size 64 --per_device_eval_batch_size 128 \
  --learning_rate 5e-4 --weight_decay 0.01 --warmup_steps 0.05 --lr_scheduler_type cosine \
  --fp16 --dataloader_num_workers 2 \
  --eval_strategy epoch --save_strategy epoch --save_total_limit 2 \
  --load_best_model_at_end --metric_for_best_model macro_f1 --greater_is_better True \
  --logging_steps 50 --report_to none --seed 42 \
  --push_to_hub --hub_model_id neghani/plantvillage-mobilenetv2 --hub_strategy end

python model/evaluate.py --model model/outputs/plantvillage-mobilenetv2 --push
python model/real_world_eval.py --model model/outputs/plantvillage-mobilenetv2 --push
```

`train.py` accepts every `transformers.TrainingArguments` flag plus:

| Flag | Meaning |
|---|---|
| `--data_dir` | PlantVillage `color` folder or the kagglehub root (auto-resolved) |
| `--crop_filter Tomato,Potato` | restrict classes to these crops (start with tomato, as the agentic plan suggests) |
| `--max_samples_per_class N` | cap per class for quick runs |
| `--val_fraction 0.1 --test_fraction 0.1` | stratified per-class split, seed = `--seed`; saved to `outputs/<run>/splits.json` |
| `--image_size` | override the 224 training crop |

## Push to Hugging Face by hand

`train.py --push_to_hub` already uploads the weights, `config.json` (with `id2label`),
`preprocessor_config.json`, `splits.json`, `run_info.json` and the metrics JSON files. If you need
to redo it, or to push a locally trained folder:

```bash
hf auth login                                   # paste the write token
hf auth whoami                                  # must print: neghani

hf repos create neghani/plantvillage-mobilenetv2 --type model --exist-ok
hf upload neghani/plantvillage-mobilenetv2 model/outputs/plantvillage-mobilenetv2 . \
  --exclude "checkpoint-*/**" --commit-message "Phase 1: MobileNetV2 on PlantVillage"

# after evaluate.py / real_world_eval.py (or just pass --push to them)
hf upload neghani/plantvillage-mobilenetv2 model/outputs/plantvillage-mobilenetv2/reports reports
hf upload neghani/plantvillage-mobilenetv2 model/outputs/plantvillage-mobilenetv2/README.md README.md

# freeze the release Phase 5 pins to
hf repos tag create neghani/plantvillage-mobilenetv2 v1.0 --message "Phase 1 release"
hf repos tag list neghani/plantvillage-mobilenetv2
```

Check the result at https://huggingface.co/neghani/plantvillage-mobilenetv2 : the model card shows the
metrics table, the "Files" tab has `reports/confusion_matrix.png`, and the hosted inference widget
classifies an uploaded leaf.

## ResNet fallback

If test macro-F1 or the real-world check is unsatisfactory, rerun the full command with:

```bash
--model_name_or_path microsoft/resnet-50 --output_dir model/outputs/plantvillage-resnet50 \
--hub_model_id neghani/plantvillage-resnet50
```

Everything downstream (`evaluate.py`, `real_world_eval.py`, `predict.py`, Phase 5) only depends on
`AutoModelForImageClassification` + `AutoImageProcessor`, so nothing else changes. Record both
results side by side in the model card and pick the better one for `HF_MODEL_ID`.

## Evaluation outputs

`evaluate.py` writes `outputs/<run>/reports/`:

| File | Contents |
|---|---|
| `metrics.json` | accuracy, top-3, macro/weighted precision-recall-F1, per-class metrics, confusion matrix, recommended threshold |
| `classification_report.txt` | sklearn per-class report (for the presentation) |
| `confusion_matrix.png` | row-normalised confusion matrix |
| `threshold_sweep.csv` | coverage and accuracy for thresholds 0.50-0.95 |
| `../README.md` | filled model card (also pushed to the Hub) |

`real_world_eval.py` adds `real_world_results.json` and `real_world_predictions.csv` and updates
the "Real-world images" section of the model card.

Confidence threshold: `metrics.json -> recommended_confidence_threshold` is the lowest threshold at
which predictions kept are at least 97% accurate while still covering at least half the test
images. Use it as `CONFIDENCE_THRESHOLD` in the API instead of the placeholder 0.6; below it the
agent answers "not sure".

## Challenges and approach

| Challenge | Approach |
|---|---|
| Tilted, inverted, partially framed photos | Random rotation up to 180 degrees, horizontal and vertical flips, random resized crop (70-100%); EXIF orientation applied at load time |
| Blur and poor light | Colour jitter and random Gaussian blur in training; a quality gate before inference tells the farmer to retake the photo |
| PlantVillage is lab-style (plain background); field photos differ | Separate real-world test set from the internet, reported next to the dataset metrics; ResNet-50 fallback if the gap is too large |
| Fake certainty on unknown crops/diseases | Softmax threshold from the sweep, "not sure" path, top-3 shown to the agent |
| Class imbalance (e.g. 150 vs 5,500 images per class) | Stratified split, macro-F1 for model selection, per-class report to spot weak classes |
| Colab disconnects | `--push_to_hub` with `--hub_strategy end` plus checkpoints each epoch; `--resume_from_checkpoint` to continue |
| Stale Colab files after a local edit | Never `files.upload()` code. Re-run `notebooks/colab_sync.py` so tracked files are always replaced from `origin/main` |
| Reproducibility | Fixed seed, `splits.json` with the exact file lists, `run_info.json` with hyperparameters, both pushed to the Hub |

## Handover contract for Phase 5

| Item | Value |
|---|---|
| Model | `HF_MODEL_ID=neghani/plantvillage-mobilenetv2` |
| Pinned revision | `HF_MODEL_REVISION=v1.0` |
| Loading | `transformers.pipeline("image-classification", model=HF_MODEL_ID, revision=HF_MODEL_REVISION)` or `common.Predictor` |
| Labels | `config.json` `id2label` = PlantVillage folder names = keys of `api/data/knowledge/diseases.json` (`Tomato___Early_blight`, ...). Extend the knowledge file to all 38 classes in Phase 5. |
| Threshold | `CONFIDENCE_THRESHOLD` = `reports/metrics.json -> recommended_confidence_threshold` |
| Quality gate | Keep `api/app/vision/quality.py`; thresholds already match `model/common.py` |
| Reference code | [`predict.py`](predict.py) `diagnose()` returns `status` in `ok` / `not_sure` / `retake_photo` plus top-k |
| Token | Not needed for a public repo; set `HF_TOKEN` only if the repo is made private |
