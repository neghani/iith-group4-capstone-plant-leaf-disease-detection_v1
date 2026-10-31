<!-- REAL_WORLD_START -->
## Real-world images (field and web photos)

The 16% / 16-of-25 block that used to sit here was a `v1.0` eval on 25 images at threshold 0.85.
The live API pin is **[`shared/release.json`](../shared/release.json)** (`neghani/plantvillage-resnet18`).
The MobileNet **`v1.1`** one-shot field report below is **historical baseline** only until ResNet stage-2
is published. Quote from `model/final_report.md` (MobileNet Hub
[`stage2-runs/runs/stage2/final_test/final_report_v1_1.md`](https://huggingface.co/neghani/plantvillage-mobilenetv2/blob/stage2-runs/runs/stage2/final_test/final_report_v1_1.md)):

| Metric | Value |
|---|---|
| Rejected by the quality gate (retake photo) | 5 of 284 |
| Top-1 on images that reached the model | 159/279 (57.0%) |
| Top-3 on images that reached the model | 219/279 (78.5%) |
| "Not sure" answers | 91 of 284 |
| **End-to-end pass rate** (correct and confident, out of all images) | **137/284 (48.2%)** (95% CI 42.5% to 54.0%) |
| Pass rate, plantdoc_test (pristine held-out) | 116/236 (49.2%) |
| Pass rate, real_world_set | 21/48 (43.8%, wide) |

Project target: 50.0%. The point estimate is below the target. Reported as-is. v1.0 on the same
284 images was 53/284 (18.7%), not sure 146/284.

This `main` branch still holds the PlantVillage-only `v1.0` training table above. Real-photo
numbers belong to revision `v1.1`.
<!-- REAL_WORLD_END -->
