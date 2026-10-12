<!-- Copied from Hub stage2-runs/runs/stage2/final_test/final_report_v1_1.md. One-shot v1.1 report; threshold and split are frozen. Do not recompute. -->

# Final test report (v1.1 vs v1.0)

Run 2026-10-11T16:01:08. Images: 284. This split was not used for any choice.

v1.1 temperature T=0.885; threshold status **selected** (threshold 0.50 has the highest pass rate (125/302 (41.4%)) among those with >=70% of 'ok' answers correct.)

### v1.0 (threshold 0.85, T=1)

Threshold 0.85. With under 100 images the 95% intervals are wide; differences of a few images are within noise.

| Metric | Value |
|---|---|
| Pass rate (correct and ok, of all images) | 53/284 (18.7%), 95% CI 14.6-23.6% |
| Top-1 on images that reached the model | 75/279 (26.9%) |
| Top-3 on images that reached the model | 137/279 (49.1%) |
| Macro F1 (classes present) | 25.7% |
| Wrong but confident (of ok answers) | 80/133 (60.2%) |
| Expected calibration error | 49.2% |
| Not sure | 146/284 (51.4%) |
| Rejected by the gate | 5/284 (1.8%) |
| Pass rate, plantdoc_test (pristine held-out) | 48/236 (20.3%), 95% CI 15.7-25.9% |
| Pass rate, real_world_set | 5/48 (10.4%), 95% CI 4.5-22.2% (wide) |

### v1.1 (threshold 0.50, T=0.885)

Threshold 0.5. With under 100 images the 95% intervals are wide; differences of a few images are within noise.

| Metric | Value |
|---|---|
| Pass rate (correct and ok, of all images) | 137/284 (48.2%), 95% CI 42.5-54.0% |
| Top-1 on images that reached the model | 159/279 (57.0%) |
| Top-3 on images that reached the model | 219/279 (78.5%) |
| Macro F1 (classes present) | 54.5% |
| Wrong but confident (of ok answers) | 51/188 (27.1%) |
| Expected calibration error | 10.0% |
| Not sure | 91/284 (32.0%) |
| Rejected by the gate | 5/284 (1.8%) |
| Pass rate, plantdoc_test (pristine held-out) | 116/236 (49.2%), 95% CI 42.8-55.5% |
| Pass rate, real_world_set | 21/48 (43.8%), 95% CI 30.7-57.7% (wide) |

## Paired difference (v1.1 minus v1.0 pass rate)

Observed +29.6 points; 95% bootstrap interval +23.9 to +35.2 (1000 resamples, seed 0).

## Per-class (v1.1)

| Class | n | Top-1 | Pass |
|---|---|---|---|
| `Apple___Apple_scab` | 12 | 3/12 | 2/12 |
| `Apple___Cedar_apple_rust` | 11 | 5/11 | 4/11 |
| `Apple___healthy` | 11 | 8/11 | 7/11 |
| `Blueberry___healthy` | 13 | 9/13 | 6/13 |
| `Cherry_(including_sour)___healthy` | 12 | 0/12 | 0/12 |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 5 | 3/5 | 3/5 |
| `Corn_(maize)___Common_rust_` | 12 | 9/12 | 9/12 |
| `Corn_(maize)___Northern_Leaf_Blight` | 13 | 6/13 | 5/13 |
| `Grape___Black_rot` | 10 | 9/10 | 7/10 |
| `Grape___healthy` | 14 | 12/14 | 12/14 |
| `Peach___healthy` | 11 | 9/11 | 9/11 |
| `Pepper,_bell___Bacterial_spot` | 11 | 8/11 | 7/11 |
| `Pepper,_bell___healthy` | 10 | 7/10 | 6/10 |
| `Potato___Early_blight` | 10 | 6/10 | 5/10 |
| `Potato___Late_blight` | 9 | 7/9 | 4/9 |
| `Raspberry___healthy` | 9 | 8/9 | 8/9 |
| `Soybean___healthy` | 10 | 7/10 | 7/10 |
| `Squash___Powdery_mildew` | 8 | 8/8 | 7/8 |
| `Strawberry___healthy` | 10 | 9/10 | 9/10 |
| `Tomato___Bacterial_spot` | 11 | 0/11 | 0/11 |
| `Tomato___Early_blight` | 11 | 6/11 | 5/11 |
| `Tomato___Late_blight` | 11 | 4/11 | 3/11 |
| `Tomato___Leaf_Mold` | 8 | 1/8 | 1/8 |
| `Tomato___Septoria_leaf_spot` | 13 | 6/13 | 5/13 |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 7 | 5/7 | 4/7 |
| `Tomato___Tomato_mosaic_virus` | 12 | 2/12 | 0/12 |
| `Tomato___healthy` | 10 | 2/10 | 2/10 |

**The final pass rate is 137/284 (48.2%), below 50%.** Reported as-is. The threshold and the data are not changed after seeing this number.

The 50-image real_world_set is all from Cropped-PlantDoc `train/` and half of it chose
the v1.0 threshold, so it is not pristine. `plantdoc_test` is the honest held-out stratum.
All images are web-sourced; gains overstate phone-photo performance.

Per-image results: `final_test_per_image.csv`.
