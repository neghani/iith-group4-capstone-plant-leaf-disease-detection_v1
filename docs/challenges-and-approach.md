# Challenges faced and approach used

Engineering log for the capstone write-up. Each entry: what went wrong, the evidence, the root
cause, what we changed, and what it cost or left open. Numbers come from runs on 2026-10-11
against the real `neghani/plantvillage-mobilenetv2@v1.0` model (served by `hf_space/`) unless
stated otherwise.

## Summary

| # | Challenge | Approach | Outcome |
|---|-----------|----------|---------|
| 1 | Large, clean photo rejected as "blurry" | Scale-normalised, leaf-only sharpness | Simulated 12 MP photos: 0/40 -> 40/40 accepted |
| 2 | Phone portrait mode / background clutter | Find the leaf first (colour + focus), judge only it | Leaf chosen over blurred grass behind it |
| 3 | Smooth leaves look "blurry" to Laplacian variance | Re-blur ratio plus a low variance floor | All 40 PlantVillage images pass, blur still rejected |
| 4 | Which leaf, and where | Candidate ranking + square crop around the winner | Held leaf found in the reference photo |
| 5 | Best size for inference | Two derived versions; measured fill and size | 256 px square, leaf fills 90% |
| 6 | Browser upload and preview | Client-side downscale, small preview | Smaller upload, preview fits storage quota (browser path not yet exercised) |
| 7 | Is the Hugging Face model really used? | Traced the call path, added provenance | Yes; every response now proves which model answered |
| 8 | Is the prediction really used by the agent? | Traced the graph, tests that vary only the prediction | Yes; plus two real gaps fixed (crop hint) |
| 9 | Confidently wrong on a plant it does not know | Crop-hint check + "Other crop" option | Partially mitigated; **open problem** (see Open issues) |

---

## 1. Large, clean photo rejected as "blurry"

**Symptom.** A sharp, well-lit leaf photo (the reference photo, `IMG_5711.jpg`, a hibiscus leaf held
in hand) was refused with "Photo looks blurry".

**Evidence.**

- The old gate computed Laplacian variance over the whole frame at native resolution and
  required >= 80. The reference photo scored **39.8** at 768x1024.
- The score depends on resolution, not only on focus. The *same* photo re-sampled:

  | Long side | 1024 | 768 | 512 | 384 | 256 |
  |-----------|------|-----|-----|-----|-----|
  | Laplacian variance | 39.8 | 64.6 | 133.6 | 217.9 | 358.3 |

  A real 12 MP phone photo is 4x larger again, so it scores far lower. Simulated 12 MP photos
  (40 PlantVillage leaves pasted on a soil background in a 3024x4032 frame): **old gate passed 0/40**,
  every one rejected as blurry.
- Two more hard limits sat behind it: any side > 4096 px was rejected ("dimensions too large", so
  every 48 MP photo), and files > 10 MB were rejected.

**Root cause.** Neighbouring pixels of a high-resolution image are strongly correlated, so the
second derivative shrinks. A gate tuned on small images punishes exactly the photos with the most
detail.

**Approach.** Decode once, then judge a *normalised* version:

1. `load_working_copy` decodes with JPEG draft mode (a 48 MP file never becomes a full-size array)
   and downsizes to <= 2048 px. Pixel-bomb guard at 120 MP instead of the 4096 px side cap; size
   limit raised to 25 MB (`MAX_IMAGE_BYTES`).
2. Sharpness is measured on the leaf at <= 256 px long side (see entries 2-3), so a 1 MP and a
   48 MP photo of the same leaf score the same.

**Result.** The reference photo upscaled to 3024x4032 (1.9 MB) and 6048x8064 (5.6 MB): both
accepted, prepared in ~0.2 s, 27 KB sent to the model. Blur sweep of the reference photo, same
optical blur expressed at each resolution, gives matching verdicts (`tests/test_preprocess.py`).

## 2. Background clutter and portrait-mode blur

**Symptom.** Even after normalising scale, the whole-frame number is dominated by the background
(deliberately blurred by phone portrait mode, or full of other plants).

**Approach.** Locate the leaf first and measure only its pixels (eroded by 2 px so the strong
leaf/background edge does not inflate the score). Brightness is also measured on the leaf, so a
bright sky or dark soil does not fail an otherwise good leaf.

## 3. Calibrating the blur threshold (two attempts)

**Attempt A - normalised Laplacian variance, threshold 120.** Scale-independent, but rejected
4/40 real PlantVillage images that are in perfect focus: smooth, glossy leaves (citrus, corn) have
little texture, so their variance is 34-97 (sharp textured leaves are in the thousands). A
variance threshold cannot tell "smooth" from "blurred".

**Attempt B - re-blur ratio.** `1 - var(lap(blur(g))) / var(lap(g))`: blur the leaf a little more
and see how much high-frequency energy disappears. Sharp leaves lose most of it, already-blurred
leaves lose little. Reference photo blur sweep (768x1024, ratio / variance):

| Gaussian sigma | 0 | 1 | 1.5 | 2 | 3 | 4 | 8 |
|----------------|---|---|-----|---|---|---|---|
| ratio | 0.92 | 0.86 | 0.79 | 0.70 | 0.58 | 0.86 | 0.70 |
| variance | 465 | 224 | 113 | 58 | 29 | 4.9 | 4.8 |
| verdict | pass | pass | pass | blur | blur | blur | blur |

**New problem: the ratio is non-monotonic.** Under heavy blur only JPEG noise is left, and noise
also "disappears" when re-blurred, so the ratio climbs back to 0.86 and a destroyed photo passed.
Fix: keep a second, absolute criterion (variance >= 25). Final rule: reject if
`ratio < 0.78 or variance < 25`.

**Calibration data.** Sharp PlantVillage leaves: min ratio 0.88, min variance 34 (all 40 pass,
margin above 25 is thin, see Open issues). Moderate blur lands at ratio 0.70-0.77, heavy blur at
variance 3-29. The same sweep at 3024x4032 (blur x4) gives the same verdicts.

## 4. Choosing the leaf (the "CNN leaf detection" question)

**Idea considered.** Run a CNN leaf detector, then keep the clearest leaf.

**Decision.** Implemented the leaf selection with classical vision first (`vision/leaf.py`),
behind a single function (`find_leaf_candidates`) that a CNN detector can replace later:

- Why not a CNN yet: it needs torch (~1 GB) in the API image, which today is deliberately tiny
  (the model lives in the Space); we have no labelled leaf-box data to validate a detector, and a
  pretrained detector would add another model whose failures we cannot measure. The colour+focus
  approach runs in ~50 ms and has no extra dependencies beyond scipy.
- Mask: vegetation hue/saturation in HSV, opening/closing, hole filling (brown blight spots inside
  a leaf are recovered).
- **Failure we hit:** the held leaf merged with the out-of-focus grass behind it through a thin
  bridge, so the crop box covered both (bbox grew from the leaf's 574x430 px to 660x696 px).
  Attempt 1, erosion-based component splitting, did not help (the bridge was thicker than the
  radius). **Fix:** drop vegetation that is blurrier than 30% of the sharpest vegetation in the
  frame (local Laplacian energy). This also gives the "take the one that is clear" behaviour.
  If the whole frame is soft nothing is dropped, and the sharpness gate rejects the photo.
- Up to 4 candidates are ranked by `focus x sqrt(area) x centrality`; the best one goes forward.
- If nothing green is found (heavily diseased or unusual leaf) the whole frame is judged and sent
  (`leaf_found: false`) rather than rejecting the photo.

## 5. Best size for inference (two derived versions)

One decode produces two versions, each scaled once:

1. **Working copy**, long side <= 2048: leaf search and quality checks.
2. **Inference copy**, 256x256 JPEG (q95): the best leaf cropped from the working copy, padded to
   a square with neutral gray, resized once with LANCZOS.

Why 256: `preprocessor_config.json` of the model resizes the short side to 256 then centre-crops
224, so sending 256 means the Space does no second resample (the old path sent 512 px, which was
resampled twice, and the centre crop of a 3:4 portrait photo cut away about a third of its height).

Measured on the Space (40 PlantVillage test images; "lab" = original, "field" = pasted in a
simulated 12 MP frame), top-1 correct:

| Setting | lab | field |
|---------|-----|-------|
| leaf fills 80%, 256 px | 39/40 | 38/40 |
| **leaf fills 90%, 256 px (chosen)** | **40/40** | **39/40** |
| leaf fills 100%, 256 px | 37/40 | 36/40 |
| 80%, 224 px | 40/40 | 38/40 |
| 80%, 384 px | 40/40 | 38/40 |
| 80%, 512 px | 40/40 | 38/40 |

Image size barely matters (differences are one image); framing does. At 100% the centre crop clips
the leaf. Whole-frame vs leaf crop, final pipeline:

| | old pipeline (whole frame, 512 px) | new pipeline |
|---|---|---|
| lab photos accepted / correct | 40 / 40 | 40 / 40 |
| simulated 12 MP photos accepted | 0 / 40 | 40 / 40 |
| simulated 12 MP photos correct (old gate bypassed) | 29 / 40 | 39 / 40 |

**Caveat.** The "field" photos are synthetic (PlantVillage leaves on a generated background), n=40,
one run. Differences of one image are noise. Treat it as a regression check, not an accuracy claim;
real field accuracy needs the real-world set in `model/data/real_world/`.

## 6. Browser: big files and the storage quota

The UI read the original file into a base64 data URL and stored it in `sessionStorage` (about a 5 MB
quota). A large photo would throw `QuotaExceededError` after the API had already answered, so the
user saw "Something went wrong". It also uploaded the full file twice (`/quality-check`, `/predict`).

`web/src/lib/image.ts` now downsizes once in the browser (long side <= 2048, JPEG 0.92, EXIF
rotation applied; files under 1.5 MB are untouched) and builds a separate 640 px preview for
storage. Falls back to the original file if decoding fails, and the API still validates
everything. Verified: TypeScript + `next build`. **Not yet exercised in a real browser with a
12 MP file.**

## 7. Is the Hugging Face model really used?

**Call path (verified by reading and by running it):** UI -> `POST /predict` -> quality gate ->
LangGraph `predict` node -> `hf_client.classify_image` -> `HF_ENDPOINT_URL`.

- With Docker Compose, `HF_ENDPOINT_URL` defaults to `inference-service` (`hf_space/`), which loads
  `neghani/plantvillage-mobilenetv2` revision `v1.0` from the Hub with `transformers.pipeline`.
- If `HF_ENDPOINT_URL` is empty the API falls back to the serverless router
  (`router.huggingface.co/hf-inference/models/<id>`). That does **not** serve this fine-tune (a test
  call returns 401, and the repo notes "model not supported"), so the user gets an honest "Analysis
  unavailable" report. There is no mock/placeholder path left in the API; nothing invents a
  diagnosis when the model is down (`test_no_hf_means_no_fabricated_diagnosis`).
- Gap found: nothing *proved* which model answered. A misconfigured URL pointing at a different
  model would look identical. Fixes:
  - `hf_space/app.py` stamps every answer with `x-model-id` / `x-model-revision`.
  - The API returns `inference` in `/predict` (model id, revision, `model_verified`, host, latency,
    bytes, and the preprocessing trace), and the results page shows it.
  - `GET /health?probe=true` calls the Space's `/health` and reports the model actually served.
  - Structured logs (`leafcheck.hf`, `leafcheck.vision`) for every call.
- Run on 2026-10-11: Space on the real Hub model, API pointed at it. `/health?probe=true` ->
  `hf_served_model: neghani/plantvillage-mobilenetv2@v1.0`; `/predict` -> `model_verified: true`.

## 8. Is the prediction really used in the agentic flow?

Flow: `predict` (HF) -> `knowledge` -> `generate_report` -> `safety` (side exits: `predict_error`,
`unsupported_crop`).

Traced and now covered by `tests/test_agent_flow.py`, which changes only the classifier output:

- `disease_id`/confidence/top-3 from the HF call select the curated knowledge entry
  (`diseases.json`; all 38 model labels have an entry, checked against the Hub `config.json`).
- The text-only LLM prompt contains the top-3 with scores, the confidence category and a
  below-threshold warning; the LLM never receives the image.
- `safety` downgrades low-confidence results to "Possible match" and requires expert confirmation.
- The chat endpoint receives `session_context` built from the same prediction.
- Different predictions give different names, actions and prompts.

**Two real gaps found and fixed**

1. The crop hint was only pasted into the prompt. If the farmer said "potato" and the model said
   "tomato", the UI still showed a confident tomato result. Now the hint is compared with the
   predicted crop (`services/crops.py`, aliases such as maize/corn, plural handling): a mismatch
   downgrades to "Possible match", requires confirmation, adds a visible note and tells the LLM.
2. A hint for a crop outside the 14 the model knows (hibiscus) now routes to an
   `unsupported_crop` node: status "Crop not supported", no disease name, no LLM write-up of a
   guess, and the UI hides the model's matches. The UI dropdown gained "Other crop (not listed)".

## 9. Stage 2: lab photos vs field photos (PlantDoc)

**Symptom.** `v1.0` (PlantVillage only) passed 53/284 (18.7%) of the one-shot field-style final
test at threshold 0.85, with 146/284 "not sure". Softmax is overconfident on the wrong class.

**Approach.** Continue from the published `v1.0` weights on a 50/50 mix of PlantVillage and
Cropped-PlantDoc (Singh et al. 2020, CC BY 4.0; 27 of 38 classes). Split by pHash cluster, exclude
every `real_world_set` original path and near-duplicates of the held-out test, fit temperature and
the threshold on **validation only**, and run the final test once per revision (guard JSON on the
Hub `stage2-runs` branch or Drive, never local `outputs/`).

**Result (`v1.1`, from `model/final_report.md`).** End-to-end pass rate **137/284 (48.2%)**, 95% CI
42.5–54.0%. Pristine `plantdoc_test`: 116/236 (49.2%). `real_world_set` stratum: 21/48 (43.8%, wide).
Not sure 91/284; quality-gate retakes 5/284. Below the 50% target; reported as-is. Paired gain over
`v1.0` on the same 284 images: +29.6 points.

**How we keep ourselves honest.** Sample sizes sit next to every percentage. The final pass rate is
below 50% and is left as-is; the threshold and the data were not changed after seeing it. Publish
`v1.1` only if PlantDoc val macro-F1 beats v1.0 and PlantVillage accuracy drops by at most 2 points.

**Limit.** All of PlantDoc is web-sourced, so a gain overstates a farmer's phone photo.

## 10. Open issues and next steps

- **Confidently wrong on unknown plants (the big one).** The reference photo is a hibiscus leaf.
  The real model returns **Tomato - Early blight at 99.9%** for it, with the old preprocessing and
  the new one alike. Softmax confidence cannot flag out-of-distribution inputs, and the 0.5
  threshold never fires. Today we only catch it when the farmer names the crop. Options: add an
  "other / not a supported crop" class and retrain, or an OOD score (energy / Mahalanobis on the
  logits), or a crop-level pre-classifier. Not done.
- **CNN leaf detector.** Swap `find_leaf_candidates` for a detector if field photos with several
  overlapping leaves, yellow/brown leaves or very cluttered scenes fail the colour+focus approach.
  Known weaknesses: soil or objects with leaf-like hue, fully necrotic leaves, thin grass-like
  leaves (maize) at long range.
- **Thresholds rest on little data.** 40 PlantVillage images + one phone photo; the variance floor
  (25) has a thin margin over the smoothest sharp leaf (34). Re-calibrate with the real-world set.
- **Background masking** (set non-leaf pixels to neutral gray to mimic PlantVillage) is a likely
  accuracy win for field photos; not tried.
- `model/common.py` still holds the original whole-frame gate (Phase 1 reference); it now differs
  from the API.
- HEIC photos (iPhone default) are only supported if the browser converts them to JPEG.
- Space cold start returns 503; the client waits 2 s and retries once.
- The browser resize path has not been exercised in a real browser with a 12 MP file.

## How to reproduce

```bash
# API tests (47): preprocessing, agent flow, HF provenance
cd api && .venv/bin/python -m pytest -q

# Real model end to end
cd hf_space && uvicorn app:app --port 7860          # loads neghani/plantvillage-mobilenetv2@v1.0
cd api && HF_ENDPOINT_URL=http://127.0.0.1:7860/ uvicorn app.main:app --port 8000
curl "localhost:8000/health?probe=true"
curl -F file=@leaf.jpg localhost:8000/predict | jq '.inference, .quality'
```

Calibration experiments (blur sweeps, fill/size comparison, old-vs-new accuracy) were run as ad hoc
scripts against the running Space with PlantVillage test images from the Hugging Face dataset
`BrandonFors/Plant-Diseases-PlantVillage-Dataset`; they are not checked in.
