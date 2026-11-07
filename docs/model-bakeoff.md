# Model bake-off (talk sheet)

**Winner / demo pin: MobileNetV2.** Live file: `shared/release.json`
→ `neghani/plantvillage-mobilenetv2@v1.2`.

Field numbers below are the one-shot **284** protocol (strategy B). We quote **`v1.1`
137/284 (48.2%)** as the honest pass rate. `v1.2` is the same training shape + sampler fix;
final_test was not re-run on that held-out set.

## Numbers

| Model | Hub tag | Thr / T | End-to-end pass | Plantdoc test | Not sure |
|---|---|---|---|---|---|
| **MobileNetV2** | `@v1.1` | 0.50 / 0.885 | **137/284 (48.2%)** | 116/236 (49.2%) | 91/284 (32%) |
| ResNet-18 | `@v1.2` (sampler fix) | 0.60 / 0.996 | 98/284 (34.5%) | 86/236 (36.4%) | 145/284 (51%) |
| ResNet-18 | `@v1.1` | 0.55 / 0.936 | 96/284 (33.8%) | — | — |
| ResNet-18 | `@v1.0` (PV only) | 0.85 / 1.0 | 39/284 (13.7%) | 36/236 (15.3%) | 172/284 (61%) |

Gate rejects: 5/284 every run. Sources: `model/final_report.md`, compare artefacts under
`model/outputs/`.

## Why MobileNet for demo (30s)

1. Highest field pass on the protocol we agreed (+39 images vs ResNet v1.2).
2. Fewer “not sure” answers while still safer on wrong-crop smokes.
3. ResNet looks stronger on clean PlantVillage — that did not transfer.
4. Remaining hole: wrong-crop confidence (strawberry→tomato). OOD margin alone isn’t enough
   — `model/outputs/ood/FALLBACK.md`.

Don’t put ResNet on the live path for demo day.
