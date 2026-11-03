# Model bake-off (talk sheet)

**Winner: MobileNetV2 `@v1.1`.** Live pin is `shared/release.json`. Same 284-image field-style test for every row (strategy B). ResNet `@v1.2` used the fixed PlantDoc sampler; that set was already used for the `v1.1` decision.

## Numbers to say

| Model | Hub tag | Thr / T | End-to-end pass | Plantdoc test | Not sure |
|---|---|---|---|---|---|
| **MobileNetV2 (winner)** | `neghani/plantvillage-mobilenetv2@v1.1` | 0.50 / 0.885 | **137/284 (48.2%)** | 116/236 (49.2%) | 91/284 (32%) |
| ResNet-18 | `@v1.2` (sampler fix) | 0.60 / 0.996 | 98/284 (34.5%) | 86/236 (36.4%) | 145/284 (51%) |
| ResNet-18 | `@v1.1` | 0.55 / 0.936 | 96/284 (33.8%) | — | — |
| ResNet-18 | `@v1.0` (PV only) | 0.85 / 1.0 | 39/284 (13.7%) | 36/236 (15.3%) | 172/284 (61%) |

Source: `model/outputs/compare_v1_2.json`, `model/final_report.md`. Gate rejects were 5/284 for every run.

## Why MobileNet wins (30 seconds)

1. **Highest field pass rate** on the honest protocol (+39 images vs ResNet v1.2).
2. **Fewer “not sure”** answers (32% vs 51%) while staying better calibrated on wrong-crop cases.
3. ResNet is stronger on clean PlantVillage lab photos; that does **not** transfer to PlantDoc-style photos.
4. Next gap is still wrong-crop confidence (strawberry→tomato); OOD margin alone is not enough — see `model/outputs/ood/FALLBACK.md`.
