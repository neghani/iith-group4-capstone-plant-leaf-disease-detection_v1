# Weekly review — model bake-off & live pin (3 Nov 2026)

**Decision:** live pin is `neghani/plantvillage-mobilenetv2@v1.1`.  
**Why this week mattered:** we had three model lines and no fair way to compare MasterDataset against the old PlantVillage field set (284 images / different label space).

## Candidates

| ID | Model | Classes | Prior claim |
|---|---|---|---|
| A | `plantvillage-mobilenetv2@v1.1` | 38 / 14 crops | 137/284 (48.2%) on old PlantDoc final |
| B | `plantvillage-resnet18@v1.1` | 38 / 14 crops | 96/284 (33.8%); was rollback |
| C | `plant-leaf-masterdataset-resnet18@v1.0.1` | 86 / 19 crops | Lab test 96.15%; not scored on 284 |

## Why not reuse the 284-image set

- Labels are PlantVillage `Crop___Disease` ids.
- MasterDataset uses different ids (`healthy_apple`, `corn_common_rust`, …).
- Scoring C on 284 would either force a lossy remap or undervalue 19-crop coverage.
- **Parity rule:** same images, only classes all three models can predict, same serving chain.

## Shared-label field set

Built by `model/build_shared_set.py` → `model/data/shared_set/`.

- Sources: PlantDoc `test/` + `real_world_set` (shared classes only).
- **21 classes**, **226 images** (tune 116 / final 110).
- pHash exclusions from MasterDataset OOD report applied.
- Threshold frozen on **tune only**; final scored once.
- Decision metric: final **top-1**; if paired CI includes 0 → lower not-sure, then crop coverage.

Full numbers: [`model/outputs/bakeoff/bakeoff_report.md`](../model/outputs/bakeoff/bakeoff_report.md).

## Results (final split, n=110)

| Model | Top-1 | Pass (ok+correct) | Not sure | Crop match | Thr |
|---|---|---|---|---|---|
| **MobileNetV2 @v1.1 (winner)** | **60/110 (54.5%)** | 41/110 (37.3%) | **50.0%** | 82/110 (74.5%) | 0.50 |
| ResNet-18 PV @v1.1 | 63/110 (57.3%) | 27/110 (24.5%) | 65.5% | 85/110 (77.3%) | 0.55 |
| MasterDataset @v1.0.1 | 14/110 (12.7%) | 14/110 (12.7%) | 13.6% | 29/110 (26.4%) | 0.30 |

No-gate (threshold 0) top-1: MobileNet 54.5% · ResNet 57.3% · Master 12.7%.  
Paired Δ MobileNet − ResNet top-1: **−2.7 pts** (95% CI −10.9 to +4.5%) → not decisive → **tie-break: lower not-sure → MobileNet**.

## What we pinned

[`shared/release.json`](../shared/release.json) / [`hf_space/release.json`](../hf_space/release.json):

- **Live:** MobileNetV2 `@v1.1`, thr 0.50, T=0.885, knowledge `diseases.plantvillage.json`
- **Candidate:** PlantVillage ResNet `@v1.1`
- **Kept (not live):** MasterDataset `@v1.0.1` under `masterdataset` for the 19-crop lab story

Verified: `/health` → `neghani/plantvillage-mobilenetv2@v1.1`. UI crop list restored to 14 PlantVillage crops.

## Talking points for review

1. **Fair comparison beats raw lab accuracy.** MasterDataset’s 96% is in-distribution lab photos; on shared field images it was 12.7%.
2. **MobileNet vs ResNet on field photos is close on top-1;** MobileNet answers more often (lower not-sure / higher end-to-end pass).
3. **19-crop product claim is still valid as a separate track** — do not quote MasterDataset lab % next to MobileNet field %.
4. **Web smoke set (`data/web_test_set/`) is not a decision set** (n≈12, noisy downloads).

## Reproduce

```bash
model/.venv/bin/python model/build_shared_set.py
model/.venv/bin/python model/bakeoff_shared.py
# pin already applied; after editing release.json:
cp shared/release.json hf_space/release.json
docker compose up --build -d --force-recreate inference-service api-service
curl -s 'http://127.0.0.1:7860/health'
```

## Open / next week

- [ ] Grow shared final set if we want tighter CIs (already n=110; wider still than the old 284).
- [ ] If demo story needs sugarcane/rice/wheat: keep MasterDataset as a **second** endpoint or clearly labeled experiment — do not silently re-pin without a new field set in 86-class space.
- [ ] Optional: phone-photo collection for a true farmer-domain bar (PlantDoc is still web-sourced).
