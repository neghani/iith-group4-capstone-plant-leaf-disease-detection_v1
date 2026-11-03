# Final test report (v1.2 vs v1.0)

Run 2026-11-04T01:02:08. Images: 284. This split was not used for any choice.

v1.2 temperature T=0.996; threshold status **selected** (threshold 0.60 has the highest pass rate (91/302 (30.1%)) among those with >=70% of 'ok' answers correct.)

### v1.0 (threshold 0.85, T=1)

Threshold 0.85. With under 100 images the 95% intervals are wide; differences of a few images are within noise.

| Metric | Value |
|---|---|
| Pass rate (correct and ok, of all images) | 39/284 (13.7%), 95% CI 10.2-18.2% |
| Top-1 on images that reached the model | 65/279 (23.3%) |
| Top-3 on images that reached the model | 135/279 (48.4%) |
| Macro F1 (classes present) | 21.7% |
| Wrong but confident (of ok answers) | 68/107 (63.6%) |
| Expected calibration error | 48.3% |
| Not sure | 172/284 (60.6%) |
| Rejected by the gate | 5/284 (1.8%) |
| Pass rate, plantdoc_test (pristine held-out) | 36/236 (15.3%), 95% CI 11.2-20.4% |
| Pass rate, real_world_set | 3/48 (6.2%), 95% CI 2.1-16.8% (wide) |

### v1.2 (threshold 0.60, T=0.996)

Threshold 0.6. With under 100 images the 95% intervals are wide; differences of a few images are within noise.

| Metric | Value |
|---|---|
| Pass rate (correct and ok, of all images) | 98/284 (34.5%), 95% CI 29.2-40.2% |
| Top-1 on images that reached the model | 158/279 (56.6%) |
| Top-3 on images that reached the model | 229/279 (82.1%) |
| Macro F1 (classes present) | 55.6% |
| Wrong but confident (of ok answers) | 36/134 (26.9%) |
| Expected calibration error | 8.0% |
| Not sure | 145/284 (51.1%) |
| Rejected by the gate | 5/284 (1.8%) |
| Pass rate, plantdoc_test (pristine held-out) | 86/236 (36.4%), 95% CI 30.6-42.8% |
| Pass rate, real_world_set | 12/48 (25.0%), 95% CI 14.9-38.8% (wide) |

## Paired difference (v1.2 minus v1.0 pass rate)

Observed +20.8 points; 95% bootstrap interval +15.1 to +26.4 (1000 resamples, seed 0).

## Per-class (v1.2)

| Class | n | Top-1 | Pass |
|---|---|---|---|
| `Apple___Apple_scab` | 12 | 5/12 | 3/12 |
| `Apple___Cedar_apple_rust` | 11 | 6/11 | 4/11 |
| `Apple___healthy` | 11 | 6/11 | 5/11 |
| `Blueberry___healthy` | 13 | 4/13 | 2/13 |
| `Cherry_(including_sour)___healthy` | 12 | 5/12 | 3/12 |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 5 | 3/5 | 3/5 |
| `Corn_(maize)___Common_rust_` | 12 | 10/12 | 7/12 |
| `Corn_(maize)___Northern_Leaf_Blight` | 13 | 5/13 | 2/13 |
| `Grape___Black_rot` | 10 | 8/10 | 5/10 |
| `Grape___healthy` | 14 | 11/14 | 9/14 |
| `Peach___healthy` | 11 | 9/11 | 7/11 |
| `Pepper,_bell___Bacterial_spot` | 11 | 9/11 | 5/11 |
| `Pepper,_bell___healthy` | 10 | 5/10 | 3/10 |
| `Potato___Early_blight` | 10 | 5/10 | 2/10 |
| `Potato___Late_blight` | 9 | 6/9 | 2/9 |
| `Raspberry___healthy` | 9 | 7/9 | 5/9 |
| `Soybean___healthy` | 10 | 6/10 | 2/10 |
| `Squash___Powdery_mildew` | 8 | 7/8 | 7/8 |
| `Strawberry___healthy` | 10 | 9/10 | 8/10 |
| `Tomato___Bacterial_spot` | 11 | 1/11 | 0/11 |
| `Tomato___Early_blight` | 11 | 7/11 | 2/11 |
| `Tomato___Late_blight` | 11 | 4/11 | 1/11 |
| `Tomato___Leaf_Mold` | 8 | 1/8 | 1/8 |
| `Tomato___Septoria_leaf_spot` | 13 | 7/13 | 2/13 |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 7 | 6/7 | 5/7 |
| `Tomato___Tomato_mosaic_virus` | 12 | 3/12 | 0/12 |
| `Tomato___healthy` | 10 | 3/10 | 3/10 |

**The final pass rate is 98/284 (34.5%), below 50%.** Reported as-is. The threshold and the data are not changed after seeing this number.

The 50-image real_world_set is all from Cropped-PlantDoc `train/` and half of it chose
the v1.0 threshold, so it is not pristine. `plantdoc_test` is the honest held-out stratum.
All images are web-sourced; gains overstate phone-photo performance.

Per-image results: `final_test_per_image.csv`.
