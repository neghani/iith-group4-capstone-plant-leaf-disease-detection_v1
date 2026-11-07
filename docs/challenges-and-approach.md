# Challenges and approach (team notes)

Working log for the write-up and demo. What broke, what we measured, what we shipped.
**Demo path we are finalising: MobileNetV2 only.** Live pin is
[`shared/release.json`](../shared/release.json) → `neghani/plantvillage-mobilenetv2@v1.2`
(threshold 0.5, T=0.885, crop strategy B, `ood_threshold=0.05`).

ResNet was an experiment. It lost the field bake-off. Do not demo it unless someone asks
about the comparison.

Numbers below: PlantVillage lab metrics from Hub cards; field pass rates from the one-shot
284-image test in `model/final_report.md` (MobileNet `@v1.1`). We did **not** re-run that
final test for `@v1.2` — same held-out set was already burned on `v1.1`.

## Snapshot for slides

| # | What bit us | What we did | Where it landed |
|---|-------------|-------------|-----------------|
| 1 | Sharp phone photo rejected as blurry | Scale-normalised, leaf-only sharpness | Simulated 12 MP: 0/40 → 40/40 accepted |
| 2 | Portrait-mode / cluttered background | Find leaf first (colour + focus), judge only it | Held leaf wins over soft grass |
| 3 | Smooth citrus/corn look “blurry” | Re-blur ratio + variance floor | Sharp PV leaves pass; real blur still fails |
| 4 | Wrong leaf / merged with background | Candidate ranking + square crop | Bridge to blurry grass fixed via focus drop |
| 5 | What size to send the model | Measure fill × size on Space | 256 px square, leaf fills ~90% |
| 6 | Browser storage quota on big uploads | Client downscale + small preview | Verified in build; still should smoke-test 12 MP |
| 7 | “Is HF actually answering?” | Provenance headers + `inference` block | `/health?probe=true` + results footer |
| 8 | Prediction ignored by the “agent” | Trace LangGraph; crop-hint tests | Hint mismatch / unsupported crop are real exits |
| 9 | Lab 99% → field ~19% (`v1.0`) | Stage-2 PlantDoc mix → `v1.1` | **137/284 (48.2%)** — below 50%, reported as-is |
| 10 | Tried ResNet for “more capacity” | Same 284 protocol bake-off | MobileNet 48.2% vs ResNet ~34% → **kept MobileNet** |
| 11 | Confident nonsense on unknown crops | Crop hint + OOD margin + Telugu/UI honesty | Partially mitigated; strawberry case still open |

---

## 1. Large, clean photo rejected as “blurry”

**What happened.** Reference phone photo (`IMG_5711.jpg`, hibiscus leaf in hand) got
“Photo looks blurry”.

**Evidence.** Old gate: Laplacian variance over the whole frame at native res, threshold 80.
Reference scored **39.8** at 768×1024. Same photo resampled:

| Long side | 1024 | 768 | 512 | 384 | 256 |
|-----------|------|-----|-----|-----|-----|
| Laplacian variance | 39.8 | 64.6 | 133.6 | 217.9 | 358.3 |

Simulated 12 MP (40 PV leaves on soil in 3024×4032): **old gate 0/40**. Also hard-rejected
sides > 4096 and files > 10 MB (48 MP phones).

**Cause.** High-res neighbours are correlated → second derivative shrinks. Gate tuned on
small images punished the sharpest photos.

**Fix.** Decode once (`load_working_copy`, JPEG draft, ≤2048). Sharpness on the leaf at
≤256 long side. Pixel bomb guard 120 MP. Upload cap in settings is still 10 MB default
(raise via env if demos need bigger).

**Result.** Upscaled reference 3024×4032 and 6048×8064 both accepted; ~27 KB to the model.
Blur sweep matches across resolutions (`tests/test_preprocess.py`).

## 2. Background clutter / portrait blur

Whole-frame score was dominated by deliberately soft background. Locate the leaf, measure
only its pixels (eroded 2 px so the leaf edge doesn’t inflate sharpness). Brightness on the
leaf too — bright sky / dark soil shouldn’t fail a good leaf.

## 3. Blur threshold (two tries)

**Try A — normalised Laplacian, thr 120.** Rejected 4/40 sharp PV images (smooth citrus/corn:
variance 34–97 vs textured leaves in the thousands).

**Try B — re-blur ratio.** `1 - var(lap(blur(g))) / var(lap(g))`. Sharp leaves lose most HF
energy; already-blurred lose little. Non-monotonic under heavy blur (JPEG noise also
“disappears”) → keep absolute floor `variance >= 25`. Final: reject if
`ratio < 0.78 or variance < 25`.

Calibration: 40 PV + one phone blur sweep. Floor margin over smoothest sharp leaf is thin —
revisit with real phones if we have time.

## 4. Which leaf? (we skipped a CNN detector)

Considered a CNN leaf box. Didn’t: torch in the API image, no labelled boxes, another
unmeasured failure mode. Classical path in `shared/vision/leaf.py` (~50 ms, scipy only):

- HSV vegetation mask, morph, hole fill (brown blight spots come back).
- Failure: held leaf merged with soft grass via a thin bridge (bbox ballooned).
  Erosion didn’t cut it. **Fix:** drop vegetation blurrier than 30% of the sharpest
  vegetation (local Laplacian).
- Rank up to a few candidates by `focus × sqrt(area) × centrality`.
- No green found → judge whole frame (`leaf_found: false`), don’t hard-reject.

Swap for a detector later behind `find_leaf_candidates` if multi-leaf field shots fail.

## 5. Inference size / framing

One decode → working copy (≤2048) + inference JPEG 256×256 q95 (leaf cropped, gray pad,
LANCZOS once). Model preprocessor is short-side 256 → centre 224; sending 256 avoids a
second resample. Old path sent 512 and the centre crop ate portrait tops.

Fill sweep (40 PV; lab vs pasted “field”): **90% fill @ 256** won. Framing mattered more
than pixel count. Caveat: synthetic field, n=40 — regression check, not a accuracy claim.

## 6. Browser quota

UI used to stash full base64 in `sessionStorage` (~5 MB) and upload twice. Quota error
*after* the API answered → “Something went wrong”. `web/src/lib/image.ts` downsizes once
(long ≤2048, JPEG 0.92; <1.5 MB untouched) + 640 preview. Still worth one real 12 MP
browser pass before demo day.

## 7. Proving the Hub model answered

Path: UI → `POST /predict` → gate → LangGraph `predict` → `hf_client` → `HF_ENDPOINT_URL`
→ `hf_space/` pipeline from `release.json`.

- Empty endpoint → no fabricated diagnosis (`test_no_hf_means_no_fabricated_diagnosis`).
- Space stamps `x-model-id` / `x-model-revision`; API returns `inference` (verified, host,
  latency, bytes, preprocess trace). Results page shows it.
- `GET /health?probe=true` must match `shared/release.json`.
- **Do not** leave stale `HF_MODEL_*` in `.env` — overrides the pin and desyncs Docker.

## 8. Prediction actually drives the agent

Graph (`api/app/agent/graph.py`):

```
predict → knowledge → generate_report → safety
         ↘ predict_error / unsupported_crop / retake / not_sure → safety
```

Covered in `tests/test_agent_flow.py` (vary only classifier output):

- disease_id / top-3 select `diseases.json` (38 labels ↔ Hub `config.json`).
- LLM gets text only (top-3, confidence band, KB). Never the image.
- Gaps we fixed in code: crop hint was prompt-only → now mismatch downgrades; unknown crop
  (hibiscus) → `unsupported_crop`, no disease name. UI has “Other crop”.

## 9. Stage 2 / field gap (PlantDoc)

`v1.0` (PlantVillage only): **53/284 (18.7%)** at thr 0.85, lots of “not sure”, softmax
confidently wrong.

Stage 2: continue from `v1.0` on 50/50 PV + Cropped-PlantDoc (Singh 2020, 27/38 classes),
pHash split, exclude `real_world_set` near-dups, fit T + threshold on **val only**, one-shot
final test.

**MobileNet `v1.1`:** **137/284 (48.2%)**, CI 42.5–54.0%. Plantdoc_test 116/236 (49.2%).
`real_world_set` 21/48 (43.8%, wide). Paired +29.6 vs `v1.0`. Below 50% — leave it.
PlantDoc is web-sourced; gains overstate phone photos.

## 10. MobileNet vs ResNet (why demo is MobileNet)

We did try ResNet-18. Brief / habit said “maybe more capacity for fine lesions.” On the
**same** 284 protocol:

| Model | Pass | Notes |
|---|---|---|
| **MobileNetV2 `@v1.1`** | **137/284 (48.2%)** | thr 0.50 / T=0.885 |
| ResNet-18 `@v1.2` (sampler fix) | 98/284 (34.5%) | thr 0.60 / T=0.996 |
| ResNet-18 `@v1.1` | 96/284 (33.8%) | |
| ResNet-18 `@v1.0` | 39/284 (13.7%) | PV only |

ResNet looks nicer on clean PlantVillage. That did **not** transfer. MobileNet also
abstained less (32% not-sure vs ~51%) and behaved better on the strawberry wrong-crop
smoke (see OOD notes). Live pin flipped back to MobileNet; ResNet stays off the demo path.

`v1.2` on MobileNet = clean ImageNet → PV → PlantDoc retrain (sampler fix in training),
Hub tag `v1.2`, pin updated 2026-11-07. **No new 284 one-shot** — quote **48.2% from
`v1.1`** as the honest field number; say `v1.2` is the same recipe with the sampler fix
and fresh weights. Hub PV test for `v1.2` card is ~98.2% / macro-F1 ~97.2% (lab).

## 11. OOD / unknown crop (still open)

Hibiscus historically → Tomato Early blight ~99%. Strawberry leaf + tomato hint:

| Setup | Behaviour |
|---|---|
| ResNet strategy B | Tomato Early blight ~97% confirmed — bad |
| MobileNet whole-frame | low conf → not_sure |
| MobileNet strategy B | often wrong crop → crop-hint downgrade helps |

Max-softmax / margin AUROC on ResNet for this mode ~0.36 (worse than chance). Live MobileNet
keeps `ood_threshold=0.05` (flat softmax → `not_sure`). Crop-level head or training
negatives still needed — `model/outputs/ood/FALLBACK.md`.

## Other scars worth remembering

- **PlantDoc sampler bug.** Per-row `1/sqrt(count)` made common classes *more* draws.
  Intended class share needs per-row `1/count**1.5`. Fix is in repo; applied in `v1.2` train.
- **Two quality gates.** `model/common.py` still has the old whole-frame gate (Phase 1 /
  some offline evals). Live API uses `shared/vision/` leaf-aware gate. Say API is source of
  truth if asked.
- **Telugu (not Hindi).** Proposal said regional language; we shipped EN/TE UI strings +
  glossary-backed LLM translate (`api/app/services/translate.py`, `web` LanguageToggle).
  Hindi pipeline still not built — don’t claim it.
- **Web smoke set.** `/web-test` + `data/web_test_set/` for tutor demos (search → classify
  live pin). GT is search target, not pathologist-verified.
- HEIC depends on browser conversion. Space cold start 503 → one retry after 2 s.
- KB entries still thin on citations.

## How we reproduce before demo

```bash
# pin check
cat shared/release.json   # expect mobilenetv2 @ v1.2

# tests
cd api && PYTHONPATH=..:. .venv/bin/python -m pytest -q

# stack
cp shared/release.json hf_space/release.json   # if you changed the pin
docker compose up --build
curl -s 'http://localhost:8000/health?probe=true' | jq .
# want: hf_reachable true, model neghani/plantvillage-mobilenetv2, revision v1.2
```

Ad-hoc blur / fill sweeps were against a running Space with PV test images — not checked in.
