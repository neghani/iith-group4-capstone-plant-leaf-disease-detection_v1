<!-- REAL_WORLD_START -->
## Real-world images (field and web photos)

### How to read the numbers

PlantVillage photos are taken under controlled conditions: one leaf, plain background. Accuracy on
the held-out PlantVillage split above is therefore an **upper bound**, not what a farmer should
expect from a phone photo. Images of the same leaf can appear in both the training and test splits,
which inflates it further. The real-world figures below are the more honest ones.

### Result

One-shot final test of tag **`v1.1`** on 2026-10-11: **284** images (236 Cropped-PlantDoc `test/`
plus 48 from the 50-image real-world set after de-dup), 27 classes that map to PlantVillage labels.
Same chain as the live app (shared quality gate, crop strategy **B**, then this model on the
256×256 crop). Temperature **T=0.885**; confidence threshold **0.50**, selected on PlantDoc
validation only (125/302 pass rate among thresholds with at least 70% of "ok" answers correct).
This split was not used for any choice. Copied from
[`model/final_report.md`](https://huggingface.co/neghani/plantvillage-mobilenetv2/blob/stage2-runs/runs/stage2/final_test/final_report_v1_1.md);
the threshold and the split are frozen.

| Metric | Value |
|---|---|
| Rejected by the quality gate (retake photo) | 5 of 284 |
| Top-1 on images that reached the model | 159/279 (57.0%) |
| Top-3 on images that reached the model | 219/279 (78.5%) |
| "Not sure" answers | 91 of 284 |
| **End-to-end pass rate** (correct and confident, out of all images) | **137/284 (48.2%)** (95% CI 42.5% to 54.0%) |
| Pass rate, plantdoc_test (pristine held-out) | 116/236 (49.2%), 95% CI 42.8–55.5% |
| Pass rate, real_world_set | 21/48 (43.8%), 95% CI 30.7–57.7% (wide) |

Project target for the end-to-end pass rate: 50.0%. The point estimate is below the
target. **Reported as-is.** The threshold and the data were not changed after seeing this number.

v1.0 on the same 284 images (threshold 0.85, T=1): 53/284 (18.7%), not sure 146/284. Paired gain
+29.6 points (95% bootstrap interval +23.9 to +35.2).

Weakest classes on this split (0 passes): `Cherry_(including_sour)___healthy` 0/12,
`Tomato___Bacterial_spot` 0/11, `Tomato___Tomato_mosaic_virus` 0/12.

Per-class table: [`model/final_report.md`](https://github.com/neghani/iiith-group4-capstone-plat-leaf-decease-detection/blob/main/model/final_report.md).
Image list, labels and sources: [`real_world_set/`](real_world_set/) (images are not redistributed here).

All images are web-sourced; gains overstate phone-photo performance. The 50-image real_world_set
is all from Cropped-PlantDoc `train/` and half of it chose the v1.0 threshold, so it is not
pristine. `plantdoc_test` is the honest held-out stratum.

### Reproducing

Do not re-run `final_test.py` for `v1.1` (the one-shot guard is already written). Quote
`model/final_report.md`.
<!-- REAL_WORLD_END -->
