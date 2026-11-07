# Training story — lab 99% to field 48.2%

This is the deep-learning chapter of the capstone. Put **2–3 slides** here in the deck;
do not jump from “lab 99%” to “field 48%” without it.

Visuals to drop on slides live in [`assets/charts/`](assets/charts/). Narrative source of
truth for the locked field number: [`../model/final_report.md`](../model/final_report.md).

## One slide worth of arc

```text
ImageNet MobileNetV2
        │  full fine-tune, lr 5e-4, phone-style aug
        ▼
PlantVillage → Hub tag v1.0
   lab ~99%   ···   field one-shot  53/284 = 18.7%
        │  continue weights · lower lr 1e-4 · label smooth 0.1
        │  50/50 PlantVillage + Cropped-PlantDoc
        │  pick checkpoint by PlantDoc val macro-F1
        │  fit T + threshold on val only
        ▼
PlantDoc stage 2 → Hub tag v1.1
   lab still ~98%   ···   field  137/284 = 48.2%   (+29.6 pp)
        │  clean retrain + PlantDoc sampler fix
        ▼
Live pin v1.2  (same recipe; 284 set NOT re-run — quote 48.2% from v1.1)
```

Chart: [`assets/charts/training-stages.svg`](assets/charts/training-stages.svg) ·
[`assets/charts/field-pass-journey.svg`](assets/charts/field-pass-journey.svg) ·
[`assets/charts/lab-vs-field.svg`](assets/charts/lab-vs-field.svg)

## What we actually trained (honest wording)

| Stage | Init | Data | Key knobs | Outcome |
|-------|------|------|-----------|---------|
| **v1.0** | `google/mobilenet_v2_1.0_224` | PlantVillage color (~54k), stratified 80/10/10, seed 42 | Full fine-tune (backbone **not** frozen), AdamW **5e-4**, cosine + 5% warmup, 5 epochs, batch 64, **macro-F1** checkpoint | Lab ~99%; field **18.7%** |
| **v1.1** | published `v1.0` weights | 50/50 PV + Cropped-PlantDoc (27/38 classes) | **lr 1e-4**, 8 epochs, **label smoothing 0.1**, PlantDoc RRC (0.5–1.0), mixed sampler, checkpoint by **PlantDoc val macro-F1**, T + thr on val | Field **48.2%** (locked) |
| **v1.2** (live) | clean ImageNet → PV → PlantDoc | same shape as stage 2 | **Sampler fix** (`1/count**1.5` per row so class share ∝ `1/sqrt(count)`) | Hub + pin; **no new 284** |

**Say this out loud if asked about “gradual unfreezing”:** we did **multi-stage transfer**
(ImageNet → PlantVillage → PlantDoc) with a **lower LR when continuing**, not layer-wise
freeze/unfreeze. Layer-wise gradual unfreeze was a reasonable next experiment we did not
run; full fine-tune was fine with ~43k PV images.

## Why lab 99% lied (leakage + shortcuts)

1. **Near-duplicate leakage.** PlantVillage often has several shots of the same leaf. A
   **random per-image** split puts siblings in train and test → inflated accuracy. We did
   **not** fix this with pHash/cluster splits on PlantVillage (known limit).  
2. **Background shortcut.** Lab backgrounds are plain; nets can learn tray/cloth, not
   lesions. Published PlantVillage analyses show this.  
3. Softmax on 38 classes always looks “confident” — so lab accuracy alone cannot drive the
   “not sure” story.

**What we did about it:** phone-style augmentations; a **held-out field-style protocol**;
stage-2 PlantDoc; for PlantDoc prep we **did** use pHash / cluster splits and excluded
`real_world_set` near-dups (`model/data_protocol.py`).

Talking point: *“99% is the in-distribution ceiling. 48.2% is the number we defend.”*

## Augmentation (phone robustness)

Training transforms (`model` / demo guide Stage 3):

- RandomResizedCrop scale **0.7–1.0** (PlantDoc stage uses **0.5–1.0**)
- H/V flip, rotation **±180°** (upside-down phone photos)
- ColorJitter (hue kept small, **0.03**)
- Gaussian blur **p=0.25**

Eval path must match serve: leaf crop strategy **B** → 256 square → processor resize/centre-crop 224.
Train/serve skew is a silent accuracy killer — we matched the processor’s `shortest_edge`.

## Training “instability” and ops scars (Colab)

These are the scars that make the story real for trainers:

| Failure mode | What happened | Fix in the repo |
|--------------|---------------|-----------------|
| Runtime disconnect | Lost ~40 min of training | `--save_strategy epoch`, `--resume_from_checkpoint`, Hub push strategy |
| Stale Colab code | Edited locally, Colab ran old file | `notebooks/colab_sync.py` hard-reset to `origin/main` |
| `transformers` API drift | `size` as int / dict / `SizeDict` | `_size_value()` in `train.py` |
| `set_transform` + Trainer | Columns dropped → “no pixel_values” | `remove_unused_columns=False` |
| Huge checkpoint uploads | GBs of optimizer state | Hub `IGNORE` patterns |
| **PlantDoc sampler bug** | Per-row `1/sqrt(count)` made **common** classes *more* draws | Per-row `1/count**1.5` so class share ∝ `1/sqrt(count)` — fixed in **`v1.2`** |

The sampler bug is a great 30-second DL anecdote: “we thought we were upweighting rare
classes; we were doing the opposite until we wrote the weight math down.”

## PlantDoc stage 2 (the jump that mattered)

`v1.0` field pass **18.7%** → after PlantDoc mix **48.2%** (+29.6 pp, bootstrap CI
+23.9…+35.2). Same locked 284 images, strategy B.

Protocol discipline:

- Diagnostics / T / threshold on **PlantDoc validation only**
- Final test **once** per revision (`final_test.py` guard) — **do not retune after seeing 48.2%**
- Publish gate: PlantDoc val macro-F1 beats `v1.0` **and** PlantVillage test drop ≤ 2 pp
- PlantDoc covers **27/38** classes; the other 11 stay lab-only

Limits to say: PlantDoc is **web-sourced**; labels not pathologist-checked; gains **overstate
true phone-in-field**.

## Calibration and abstention

After stage 2: temperature **T=0.885**, confidence threshold **0.50** (PlantDoc val rule:
≥70% of “ok” answers correct among candidates). Live pin keeps these. Softmax still fails
OOD (shoe / wrong crop) — threshold + top-3 + crop hint + OOD margin are mitigation, not a
cure.

## Per-class field failures (put on a slide)

Chart: [`assets/charts/per-class-field-failures.svg`](assets/charts/per-class-field-failures.svg)

Zero or near-zero passes worth calling out (from `final_report.md`):

| Class | Pass |
|-------|------|
| Tomato bacterial spot | 0/11 |
| Cherry healthy | 0/12 |
| Tomato mosaic virus | 0/12 |
| Tomato leaf mold | 1/8 |

Stronger: grape healthy 12/14, strawberry healthy 9/10, corn common rust 9/12.

**Confusion matrix (lab):** generated by `model/evaluate.py` as `confusion_matrix.png` next to
the Hub reports for each tag — download from the MobileNet model card / `reports/` for the
slide. Typical look-alikes: tomato early blight ↔ Septoria ↔ target spot; corn gray leaf
spot ↔ northern leaf blight. Field confusion is the per-class table above when the PNG is
missing locally.

## Bake-off coda (one line on the training slide)

Same 284 protocol: MobileNet **48.2%** vs ResNet-18 ~**34%**. Bigger backbone did not
transfer. Live pin stays MobileNet. Talk sheet: `model-bakeoff.md`.

## What to say in 90 seconds

> We started from ImageNet MobileNet, full fine-tuned on PlantVillage with phone-style
> augmentations — lab held-out looked like 99%, but the locked field set was only 19%.
> That gap is near-duplicates and lab backgrounds. Stage 2 continued those weights on a
> 50/50 PlantVillage + PlantDoc mix, lower LR, label smoothing, checkpointed on PlantDoc
> macro-F1, calibrated on val only. Field pass jumped to 48.2% — still under the 50% goal,
> reported as-is. We later fixed a PlantDoc sampler bug that favoured common classes and
> published that as v1.2 without re-burning the final test.

## Pointers

| Artefact | Path |
|----------|------|
| Locked field report | `model/final_report.md` |
| Train / PlantDoc prep / sampling | `model/train.py`, `prepare_plantdoc.py`, `sampling.py` |
| Leakage asserts | `model/data_protocol.py` |
| Engineering log | `challenges-and-approach.md` §9–10 |
| Slide visuals + product shots | `10-slide-visuals.md` |
