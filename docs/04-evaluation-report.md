# Evaluation report (numbers to quote)

Capstone write-up summary. Full tables: Hub model card, `model/final_report.md`,
`docs/model-bakeoff.md`.

## Protocol we trust

| Split | What it is | How to use it |
|-------|------------|---------------|
| PlantVillage held-out test | Lab-style, stratified, seed 42 | In-distribution ceiling (~98–99%) |
| One-shot 284 (strategy B) | PlantDoc test + real_world_set stratum | **Honest field-style number** |
| `real_world_set` alone | 50 images, tune/final | Wide CI; part of the 284 |

**Do not** re-run `final_test.py` for `v1.1` / casually for `v1.2` — set was burned for the
`v1.1` decision. Live pin `@v1.2` = same training shape + PlantDoc sampler fix; quote
**48.2% from `v1.1`** as the field figure.

## MobileNet results

### Lab (PlantVillage test)

Order of magnitude from Hub cards: accuracy ~98–99%, macro-F1 ~97–99%, top-3 near saturated.
`@v1.2` card ~**98.2%** / macro-F1 ~**97.2%**. Treat as optimistic for phones (near-dups,
plain backgrounds).

### Field-style one-shot (`v1.1`, thr 0.50, T=0.885)

From `model/final_report.md`:

| Metric | v1.0 (PV only) | **v1.1** |
|--------|----------------|----------|
| Pass (correct & ok / all) | 53/284 (**18.7%**) | **137/284 (48.2%)** |
| 95% CI (v1.1) | — | 42.5–54.0% |
| Not sure | 146/284 | 91/284 |
| Gate rejects | 5/284 | 5/284 |
| plantdoc_test | 48/236 (20.3%) | 116/236 (49.2%) |
| real_world_set | 5/48 (10.4%) | 21/48 (43.8%, wide) |
| Paired gain | — | **+29.6 pp** |

**48.2% is below the 50% proposal target.** Reported as-is; threshold/data not changed after
seeing it.

Weak classes on that run (0 passes): e.g. cherry healthy, tomato bacterial spot, tomato
mosaic — see per-class table in `final_report.md`.

## Bake-off (same 284, strategy B)

| Model | Pass | Not sure |
|-------|------|----------|
| **MobileNetV2 `@v1.1`** | **137/284 (48.2%)** | 32% |
| ResNet-18 `@v1.2` | 98/284 (34.5%) | 51% |
| ResNet-18 `@v1.1` | 96/284 (33.8%) | — |
| ResNet-18 `@v1.0` | 39/284 (13.7%) | 61% |

**Live pin = MobileNet.** ResNet is comparison only.

## Live app decision settings

From `shared/release.json`:

- `confidence_threshold`: **0.5**  
- `calibration_temperature`: **0.885**  
- `crop_strategy`: **B**  
- `ood_threshold`: **0.05**  

Web/Local smoke tables use the same band: score &lt; 0.5 → abstain; optional GT crop hint →
crop_mismatch on wrong-crop highs.

## What evaluation is *not*

- DuckDuckGo `/web-test` match rate — GT is search intent, not verified labels.  
- Mixing ResNet lab numbers with MobileNet field claim.  
- Claiming phone-photo accuracy equal to PlantDoc gains (web-sourced).

## How to reproduce (high level)

```bash
# API / gate / agent tests
cd api && PYTHONPATH=..:. .venv/bin/python -m pytest -q

# Live pin health
curl -s 'http://localhost:8000/health?probe=true'

# Field report (read-only artefact)
cat model/final_report.md
```

Training / final_test commands: `model/README.md` and Colab notebook — don’t burn the
guarded final set for a demo slide.
