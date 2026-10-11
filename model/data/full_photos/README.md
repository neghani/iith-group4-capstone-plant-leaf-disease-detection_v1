# Full phone photos (end-to-end set)

Our own uncropped phone photos, several leaves per frame. They test what Cropped-PlantDoc
cannot: the leaf finder, the crop strategy and the quality gate on a real upload.

## Collecting

- **30 or more photos**, at least 2 per crop we demo.
- Several leaves per photo. Include a few **healthy plants with one blemished leaf** and a few
  with **several diseased leaves**.
- Shoot as a farmer would: straight from the phone camera, no cropping, no editing. Keep the
  originals (EXIF orientation is applied by the loader).
- Put files under `images/<label>/` (git-ignored, the repo is public). Share the folder as a zip
  on Drive like `real_world_set`.

## labels.csv

| Column | Meaning |
|---|---|
| `file` | path relative to this folder, e.g. `images/Tomato___Early_blight/fp_001.jpg` |
| `label` | the photo's **main** disease as a PlantVillage class name (see `api/tests/model_labels.json`) |
| `plantdoc_class` | leave empty (kept so the format matches `real_world_set/labels.csv`) |
| `split` | `tune` or `final`, about half each |
| `notes` | free text, e.g. `healthy plant, one leaf with spots`, `5 leaves, 3 diseased`, `backlit` |
| `plant_id` | one id per physical plant, e.g. `tomato_bed2_p3`. All photos of one plant get the same id |

**Split by plant, not by photo:** every photo of one plant goes in the same split.
`model/eval_strategies.py` refuses to run the final split if a `plant_id` appears in both splits
or if two photos across splits are exact or near duplicates.

Decide the split **before** looking at any results, and do not move photos between splits later.
