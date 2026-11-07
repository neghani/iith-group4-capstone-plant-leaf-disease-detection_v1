# Slide visuals — charts + product screenshots

Drop these into the deck. Files are under [`assets/`](assets/). Training narrative
copy: [`09-training-story.md`](09-training-story.md).

## Must-have slides (order)

| # | Slide | Visual | Say |
|---|-------|--------|-----|
| A | Product in one shot | [`screenshots/03-results-ok.png`](assets/screenshots/03-results-ok.png) | “Farmer sees disease + care — MobileNet decides, KB/LLM explains.” |
| B | Lab vs field | [`charts/lab-vs-field.svg`](assets/charts/lab-vs-field.svg) (or `.png`) | Keep the two numbers apart. |
| C | **Training story** | [`charts/training-stages.svg`](assets/charts/training-stages.svg) + [`charts/field-pass-journey.svg`](assets/charts/field-pass-journey.svg) | ImageNet → PV → PlantDoc; 18.7% → 48.2%. |
| D | Failures | [`charts/per-class-field-failures.svg`](assets/charts/per-class-field-failures.svg) | Tomato bacterial spot / mosaic / cherry healthy. |
| E | Lab confusion | [`charts/confusion-matrix-lab.png`](assets/charts/confusion-matrix-lab.png) | Look-alikes (blights, corn spots) — in-distribution only. |
| F | Not-sure path | [`screenshots/02-not-sure.png`](assets/screenshots/02-not-sure.png) | Softmax abstention / threshold. |
| G | Quality gate | [`screenshots/04-retake.png`](assets/screenshots/04-retake.png) | Dark photo never reaches the model. |
| H | Upload UI | [`screenshots/01-check-upload.png`](assets/screenshots/01-check-upload.png) | Optional; A alone is enough for product. |

## Product screenshots (captured from live stack)

| File | Path shown |
|------|------------|
| `01-check-upload.png` | `/check` upload + guidelines |
| `02-not-sure.png` | Results — “I'm not confident yet” (demo button) |
| `03-results-ok.png` | Results — Squash powdery mildew ~98%, pin `@v1.2` |
| `04-retake.png` | Results — quality gate “too dark”, no model call |

Regenerate later: warm Compose → `/check` → **Show a retake** / **Show “I'm not sure”** for F/G;
for a confident result, run `POST /predict` on a strong web image and open `/results` after
saving the session (helper JSON kept as `screenshots/_session_ok.json` for tooling only —
not a farmer feature).

## Chart sources

| Chart | Data source |
|-------|-------------|
| Field journey / lab-vs-field / per-class | `model/final_report.md` (locked 284) |
| Training stages | Hub tags + `model/README.md` stage-2 recipe |
| Confusion matrix | Hub `neghani/plantvillage-mobilenetv2` `reports/confusion_matrix.png` (lab PV) |

Prefer **PNG** in PowerPoint/Google Slides; SVG is fine in Keynote / HTML decks.

## Demo-day timeline tweak

Insert **~2 minutes** after architecture for slides **B–E** (training story + charts) before
the live demo. Do **not** skip from “lab 99%” to “field 48%” in one breath.

Suggested spoken bridge:

> Lab accuracy looked solved. The field set showed it wasn’t. Here’s the work between those
> two numbers — augmentation, a PlantDoc stage, a sampler bug we fixed, and a locked test
> we refused to retune after seeing 48.2%.
