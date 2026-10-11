# Real-world test set (50 images)

Labelled field and web photos for testing the plant-disease model outside PlantVillage.

## What it is
- 50 images, 27 classes, sampled at random (seed 42) from **Cropped-PlantDoc** (Singh et al., 2020): https://github.com/pratikkayal/PlantDoc-Dataset
- 1 image per class plus a second image for 23 randomly chosen classes.
- Each image has a PlantVillage label in `labels.csv` (PlantDoc class names are mapped to the 38 PlantVillage class names).
- `split` column: `tune` (25) is for choosing the quality-gate and confidence thresholds. `final` (25) is for the number you report, once.
- `sources.csv` records the original path, class and licence of every image.
- Images are stored in folders named by label: `images/<label>/<file>`. Files larger than 1600 px were downscaled to 1600 px.

## How to use (Colab or local, from the repo root)
```bash
pip install -r model/requirements.txt scipy pydantic-settings
python model/eval_real_world_set.py --set_dir model/data/real_world_set --split tune  --predictor app_predictor:diagnose --strategy B --threshold 0.85
python model/eval_real_world_set.py --set_dir model/data/real_world_set --split final --predictor app_predictor:diagnose --strategy B --threshold 0.85 --push
```
`model/app_predictor.py` runs the **same chain as the live app**: the shared quality gate and crop strategy (`shared/vision/`), then the Hub model (`HF_MODEL_ID` @ `HF_MODEL_REVISION`) on the 256x256 crop, with `CROP_STRATEGY` and `CONFIDENCE_THRESHOLD`. `predict:diagnose` runs the same preprocessing on an in-memory image. Test the script without a model with `--selftest`. `--push` uploads the results and this folder's metadata (not the images) to the Hub model repo; only `--split final` rewrites the model card.

## Colab: publish on dataset change
`images/` is git-ignored (the GitHub repo is public). Zip this whole folder and put it on Drive as
`MyDrive/plant-disease/real_world_set.zip`; notebook section 7 copies it in (`colab_sync.py
--real-world-set-from`), runs `tune` and `final`, and republishes the Hub card only when the set
(labels.csv + every image), the weights or the threshold differ from `reports/real_world_set/<split>/run.json`.
To ship a **new** set: same layout (`labels.csv` with `file,label,split`, `images/`, `sources.csv`
with a `source` column), replace the zip on Drive, re-run section 7. Put set-specific limitations
and citation in `card_notes.md`; they go into the card verbatim.

Rebuild the images from PlantDoc (deterministic, seed 42):
```bash
python model/build_real_world_set.py <Cropped-PlantDoc dir with train/ and test/> model/data/real_world_set
```

## Licence and attribution
The PlantDoc repository is licensed CC BY 4.0. The images were collected from the web by the PlantDoc authors, and some carry stock-photo watermarks, so third-party rights may remain with the original owners. Keep the repository private or limited to coursework and reviewers, keep `sources.csv`, and cite:

Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., Batra, N. (2020). PlantDoc: A Dataset for Visual Plant Disease Detection. CoDS-COMAD 2020. https://doi.org/10.1145/3371158.3371196

## Limits to state in your report
- **Small.** With 50 images (25 per split) a pass rate has a wide confidence interval (about +/- 14 points at 50%). Add more images from the same source if you need a firmer number.
- **Cropped-PlantDoc images are already cropped around leaves.** They test the classifier on field-style leaves, not the full phone-photo path (3000x4000 upload, leaf finder, quality gate). Add some of your own full phone photos for an end-to-end check.
- **Label mapping.** A few PlantDoc classes are matched to the closest PlantVillage class (for example "Bell_pepper leaf spot" to "Pepper,_bell___Bacterial_spot"). Some web labels may be noisy. "Tomato two spotted spider mites leaf" was left out because PlantDoc has only 2 images of it.
- **Not the PlantVillage classes.** PlantDoc has no (or, for spider mites, only 2) images for 11 PlantVillage classes, such as orange citrus greening and tomato target spot, so those classes are not tested.
