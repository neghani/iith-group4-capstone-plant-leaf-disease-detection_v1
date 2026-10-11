---
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: image-classification
base_model: {{base_model}}
tags:
- image-classification
- plant-disease
- plantvillage
- agriculture
- mobilenetv2
datasets:
- abdallahalidev/plantvillage-dataset
metrics:
- accuracy
- f1
model-index:
- name: {{model_name}}
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      name: PlantVillage (color, held-out test split)
      type: plantvillage
    metrics:
    - type: accuracy
      value: {{accuracy}}
      name: Accuracy
    - type: f1
      value: {{macro_f1}}
      name: Macro F1
    - type: accuracy
      value: {{top3_accuracy}}
      name: Top-3 accuracy
---

# {{model_name}}

Plant leaf disease classifier for the capstone project **AI-Powered Plant Disease Detection and
Farmer Assistance** (IIIT-H, Group 4). A farmer uploads a photo of a crop leaf; this model returns
the most likely PlantVillage class and a confidence score. A LangGraph agent then explains the
result and suggests treatment from a curated agricultural knowledge base.

- Base model: [`{{base_model}}`](https://huggingface.co/{{base_model}}), fine-tuned end to end
- Classes: {{num_classes}} ({{crop_scope}})
- Trained: {{date}} · Hub id: `{{model_id}}`

## Intended use

- Input: a single RGB photo of one leaf, roughly centred, taken in daylight.
- Output: PlantVillage class name (for example `Tomato___Early_blight`) plus softmax probability.
- Treat a top-1 probability below **{{recommended_threshold}}** as "not sure" and ask for a clearer
  photo or expert confirmation (see threshold sweep below).
- Not a substitute for a plant pathologist. Treatment decisions must be confirmed with a local
  agricultural extension officer.

## Training data

- [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) `color`
  images, stratified per-class split with seed {{seed}}:
  train {{num_train}} · validation {{num_val}} · test {{num_test}}.
- Stage 2 (tag `v1.1` and later) also uses [Cropped-PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset)
  (Singh et al. 2020, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)): 27 of 38 classes;
  the other 11 stay lab-only. See `run_info.json` for the PlantDoc commit and data manifest.
- Augmentation (phone-photo robustness): random resized crop (70-100% PlantVillage, 50-100% PlantDoc),
  horizontal and vertical flip, rotation up to 180°, brightness/contrast/saturation jitter, random Gaussian blur.
- Evaluation preprocessing: resize shortest edge, centre crop 224, normalise (same as the bundled
  `preprocessor_config.json`).

## Results on the held-out PlantVillage test split

| Metric | Value |
|---|---|
| Accuracy | {{accuracy_pct}} |
| Macro precision | {{macro_precision_pct}} |
| Macro recall | {{macro_recall_pct}} |
| Macro F1 | {{macro_f1_pct}} |
| Top-3 accuracy | {{top3_accuracy_pct}} |
| Test images | {{num_test}} |

Confusion matrix: [`reports/confusion_matrix.png`](reports/confusion_matrix.png) ·
Full per-class report: [`reports/classification_report.txt`](reports/classification_report.txt)

### Per-class precision / recall / F1

{{per_class_table}}

### Confidence threshold sweep

Coverage is the share of test images with top-1 probability at or above the threshold; the
remaining images would be answered with "not sure". Recommended `CONFIDENCE_THRESHOLD` for the API:
**{{recommended_threshold}}** ({{threshold_rule}}).

{{threshold_table}}

<!-- REAL_WORLD_START -->
## Real-world images

Not evaluated yet. Run `python model/eval_real_world_set.py --set_dir model/data/real_world_set
--split final --predictor app_predictor:diagnose --strategy B --threshold 0.85 --push` to fill this section.
<!-- REAL_WORLD_END -->

## Limitations

- PlantVillage images are lab-style photos on plain backgrounds; accuracy on field photos is lower.
  The real-world section above reports the gap.
- Only the {{num_classes}} PlantVillage classes are known. Other crops, other diseases, nutrient
  deficiencies and pest damage will be forced into the nearest known class; rely on the threshold
  and the "not sure" path.
- One leaf per photo. Mixed infections and whole-plant shots are out of scope.
- Confidence is a softmax probability. Stage 2 ships a fitted temperature in `calibration.json`
  (NLL on PlantDoc validation). Treat remaining overconfidence as a limitation.
- PlantDoc and the 50-image real-world set are web photos; they overstate phone-photo performance.
  The one-shot comparison is `final_report_v1_1.md` (not used for any choice).

## Training details

{{training_details}}

## Use in the API (Phase 5)

```python
from transformers import pipeline

clf = pipeline("image-classification", model="{{model_id}}", revision="{{revision}}")
preds = clf("leaf.jpg", top_k=3)
# [{'label': 'Tomato___Early_blight', 'score': 0.93}, ...]
top = preds[0]
if top["score"] < {{recommended_threshold}}:
    print("Not sure - please retake the photo or consult an expert")
```

Class names match the keys of `api/data/knowledge/diseases.json` in the project repository, so
the prediction can be looked up directly for symptoms and treatment guidance.

Project repository: https://github.com/neghani/iiith-group4-capstone-plat-leaf-decease-detection

## Disclaimer

Treatment suggestions produced from this model's output are general guidance only and must be
validated with local agricultural experts before critical action.
