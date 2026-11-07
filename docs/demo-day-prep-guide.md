# Demo day prep — CropCare AI (team notes)

For Ganesh / Group 4. Capstone demo in front of DL / ML / Agentic trainers.
**We are demoing MobileNetV2 only** (`neghani/plantvillage-mobilenetv2@v1.2` in
`shared/release.json`). ResNet was tried and lost the field bake-off — keep those numbers
in your pocket if asked, don’t put ResNet on the happy path.

Companion docs: `docs/challenges-and-approach.md` (engineering scars),
`docs/phase2-demo-checklist.md` (day-of ticks).

---

## 60-second pitch (say this, then show it)

Farmer photographs a leaf. Quality gate checks the photo first (dark / blurry / tiny /
no leaf?) and asks for a retake instead of guessing. We fine-tuned **MobileNetV2** on
PlantVillage (38 classes, 14 crops), then continued on a PlantDoc mix. Lab held-out is
~98–99%; the honest field-style one-shot is **137/284 (48.2%)** — below our 50% goal, we
say that out loud. LangGraph looks up a curated KB and an LLM writes plain language.
**The LLM never decides the disease and never sees the image.** Low confidence, wrong crop,
and off-topic chat are explicit exits. Model is pinned on Hugging Face; whole stack is
`docker compose up`.

---

## What is actually built (don’t oversell)

Working:

- Train / eval / publish scripts under `model/` + Colab notebook for MobileNet `v1.2`
- Inference container `hf_space/` (pipeline from `release.json`)
- Leaf-aware quality gate (`shared/vision/`)
- LangGraph: predict → knowledge → report → safety (+ error / unsupported / retake / not_sure)
- KB for all 38 classes
- Next.js UI (`/check`, `/results`, chat, EN/TE toggle)
- Provenance on every `/predict` (`inference` block)
- `/web-test` for live smoke on web photos (tutor toy, not the graded metric)

Gaps / honesty list (say first if they look for them):

| Gap | What to say |
|---|---|
| Field pass **48.2% < 50%** | Measured once on 284; not tuned after. Paired +29.6 over `v1.0` (18.7%). |
| 99% lab is optimistic | Near-dup leaves + plain backgrounds. Real number is the 284 test. |
| Proposal said Streamlit / Grok / Hindi | We shipped Next.js, OpenAI-compatible client (Grok via `LLM_BASE_URL`), **Telugu** MVP not Hindi. |
| “Agentic” | Fixed graph with conditional exits — deliberate. LLM doesn’t pick tools in a loop. |
| Chat topic guard | Keyword list + system prompt. Brittle; second layer is the prompt. |
| KB citations | Still thin. |
| `v1.2` final_test | **Not re-run** (set burned on `v1.1`). Quote 48.2% from `v1.1`; `v1.2` = same recipe + sampler fix + new weights. |
| README / model README drift | Some files still say ResNet from the bake-off era. Pin file wins: `shared/release.json`. |

Never open `.env` on a shared screen.

---

## Architecture (one breath)

```
Browser (Next :3000)
  → POST /quality-check   (local gate, no model)
  → POST /predict         (gate again → LangGraph → HF container :7860)
  → POST /chat            (text + session_context from browser)

LLM ← text only (top-3, KB, confidence). Image never leaves the vision path.
```

Rules we stick to:

- Vision decides, language explains.
- Pin from `shared/release.json` (copy into `hf_space/release.json` after pin changes).
- Fail soft: model down → “Analysis unavailable”; LLM down → KB template text.
- Crop strategy **B** (sharpest leaf), OOD margin 0.05 → flat softmax becomes `not_sure`.

---

## Journey in short (what we actually struggled with)

Don’t narrate every stage unless asked. This is the story arc:

1. **Scope.** Classification, not detection. One leaf → one of 38 labels. Tomato-only smoke
   first (`--crop_filter`), then full.
2. **Data.** PlantVillage from Kaggle (`color/` only). Ugly class names, imbalance →
   stratified split + macro-F1. Near-dup leakage acknowledged, not fixed.
3. **Train.** Colab Pro, HF Trainer, MobileNetV2 ImageNet → PV. Publish tags on Hub.
   HF serverless won’t host the fine-tune → own Space/container.
4. **Gate pain.** Whole-frame Laplacian rejected good phone photos. Leaf-only re-blur
   ratio. Details in challenges doc.
5. **Field cliff.** `v1.0` 18.7% on 284. Stage-2 PlantDoc → MobileNet `v1.1` 48.2%.
6. **ResNet side quest.** Sampler-fixed ResNet still ~34% on same test. Went back to
   MobileNet for the live pin. That’s the bake-off.
7. **Agent safety.** Crop hint mismatch, unsupported crop, no image to LLM, Telugu
   translate with glossary.

Fresher traps we hit: stale Colab code (`colab_sync.py`), `remove_unused_columns=False`,
SizeDict shape churn in transformers, Next.js `NEXT_PUBLIC_*` bake-at-build, browser
localhost ≠ container DNS for the model.

---

## Bake-off numbers (if they ask “why not ResNet?”)

Same 284 images, strategy B:

| Model | Pass | Not sure |
|---|---|---|
| **MobileNetV2 `@v1.1` (winner)** | **137/284 (48.2%)** | 91/284 (32%) |
| ResNet `@v1.2` | 98/284 (34.5%) | 145/284 (51%) |
| ResNet `@v1.1` | 96/284 (33.8%) | — |
| ResNet `@v1.0` | 39/284 (13.7%) | 172/284 (61%) |

MobileNet: higher pass, fewer abstentions, better wrong-crop smoke behaviour.
ResNet wins clean lab photos — irrelevant for the demo claim.

---

## Decisions we own (one-liners)

| Choice | Why |
|---|---|
| Classification | Brief + one leaf one label |
| MobileNetV2 live | Field bake-off + CPU demo |
| Full fine-tune | Lesions ≠ ImageNet textures; 43k images |
| Macro-F1 for selection | Imbalance |
| Leaf-aware gate | Old gate failed real phones |
| Own inference container | Serverless doesn’t host our fine-tune |
| Next.js not Streamlit | Figma / UX |
| LangGraph workflow | Auditable exits > autonomous LLM |
| LLM explains only | Safety |
| Curated JSON KB | Deterministic actions, no free-form chemistry |
| Client sessionStorage | No auth / no PII store for demo |
| Telugu MVP | Regional language marks; glossary + LLM translate |

---

## Run locally (demo machine)

```bash
cp .env.example .env          # LLM_API_KEY optional
# Do NOT set HF_MODEL_ID / HF_MODEL_REVISION unless overriding on purpose

docker compose up --build
# UI  http://localhost:3000/check
# API http://localhost:8000/docs
# health: curl -s 'http://localhost:8000/health?probe=true'
```

Three-terminal debug if compose fights you:

```bash
# 1 model
cd hf_space && uvicorn app:app --port 7860
# 2 API
cd api && HF_ENDPOINT_URL=http://localhost:7860/ PYTHONPATH=..:. \
  uvicorn app.main:app --reload --port 8000
# 3 UI
cd web && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Warm the model once before the room fills. First Hub pull is slow.
Offline insurance: run once online, then `HF_HUB_OFFLINE=1` with a cached volume — rehearse
that exact mode. Record a 2‑min backup video.

API tests before you leave home:

```bash
cd api && source .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH=..:. pytest -q
```

---

## Demo script (~8–10 min)

| Time | Do | Say |
|---|---|---|
| 0–1 | Pitch | Farmers need fast, safe, plain guidance |
| 1–2.5 | Architecture | Vision decides, language explains |
| 2.5–4 | Model slide | MobileNet, lab ~98%, **field 48.2%**, why not ResNet |
| 4–5 | Gate | One live retake (dark/blur) |
| 5–7.5 | Live flows below | |
| 7.5–8.5 | Challenges | Gate, field cliff, bake-off, crop hint |
| 8.5–9.5 | Limits + next | OOD, phone photos, crop head |
| 9.5+ | Q&A | |

Live clicks (pre-test every file in a `demo_images/` folder):

1. `/health?probe=true` → MobileNet `@v1.2`, reachable
2. Clear tomato blight → diagnosis + top-3 + actions
3. Healthy leaf → “Looks healthy”
4. Dark/blur → retake, no model call
5. Crop = Other / hibiscus → Crop not supported
6. Tomato leaf + Potato hint → Possible match + mismatch note
7. One hard real/web photo (failure is fine) → explain top-3 / not_sure
8. Chat: “Should I remove lower leaves?” → contextual
9. Chat: cricket score → polite decline
10. Optional: language toggle → Telugu guidance (needs LLM key)
11. Optional: `/web-test` one row — live pin latency badge

Also exercise UI helpers on `/check`: **Show a retake**, **Show “I'm not sure”**.

---

## Risk list (day of)

| Risk | Plan |
|---|---|
| Cold start / network | Warm 10 min early; local compose; recorded video |
| LLM quota | Fall back to KB text — rehearse that mode |
| Wrong live answer | Pre-tested images; talk top-3 + threshold |
| Wi-Fi | Hotspot; offline cache if you practised it |
| Stale Next env | `docker compose up --build`, hard refresh |
| `.env` on screen | Close it; clean terminal |
| Sleep / battery | Plug in |

---

## Trainer Q&A prep (rehearse out loud)

Answers match the **MobileNet demo path** (`@v1.2` pin; field number from `@v1.1`
one-shot **137/284**). If you blank, say the number and the limit — don’t invent.

### Deep learning / ML

**Q1. Why MobileNetV2?**
Small, CPU-friendly, depthwise-separable, in the brief. We also ran ResNet-18 through the
same 284 field protocol and MobileNet won (48.2% vs ~34%). Demo pin is MobileNet; ResNet
is the comparison story, not the live path.

**Q2. What is transfer learning and why use it?**
Start from ImageNet weights (edges / textures), replace the head with 38-way, fine-tune on
PlantVillage then PlantDoc mix. Converges in a few epochs on ~43k images; better than from
scratch for us.

**Q3. Did you freeze layers?**
No — full fine-tune. Lesions aren’t ImageNet textures. Head-only warmup then unfreeze with
a smaller LR is a fair next experiment; we didn’t run it.

**Q4. Why macro-F1 and not accuracy?**
Classes are imbalanced (~150 to ~5.5k). Accuracy is dominated by big classes. Macro-F1
weights each class equally; we select checkpoints on val macro-F1 (PlantDoc val in stage 2).

**Q5. 99% sounds too good. Is it?**
Real on the PlantVillage held-out split, optimistic for the field: near-duplicate leaf shots
and uniform backgrounds. Say it yourself: lab ceiling vs honest **137/284 (48.2%)** on the
one-shot field-style set. `v1.2` Hub card is ~98% lab — same story.

**Q6. What is in your augmentation and why?**
Random resized crop, H/V flips, rotation up to 180°, colour jitter, random Gaussian blur.
Meant to imitate tilted / inverted / partial / dim / soft phone photos. PlantDoc path uses a
wider RRC scale than PlantVillage.

**Q7. Explain the confusion matrix. What gets confused?**
Rows true, columns predicted. Look-alikes: tomato early blight vs Septoria vs target spot;
corn leaf spots / northern blight. Tiny-support classes (e.g. potato healthy on PV test)
make recall noisy — quote support next to the %.

**Q8. What is top-3 accuracy and why show it?**
Correct if the true label is in the top three scores. Lab top-3 is basically saturated. We
show top-3 to the farmer and the LLM so ambiguity is visible.

**Q9. What learning rate and why?**
Stage 1 style: AdamW ~5e-4, cosine, warmup. Stage 2 (PlantDoc mix): lower LR (~1e-4) from
`v1.0` / stage-1 weights, label smoothing 0.1. We didn’t publish a full LR sweep — say
“common fine-tune practice, not heavily tuned.”

**Q10. What is softmax and why is it overconfident?**
Softmax forces scores to sum to 1; it always picks something. No “none of the above.”
Mitigations: confidence threshold, top-3, crop-hint check, quality gate, OOD margin
(`ood_threshold=0.05` → flat softmax → `not_sure`). Still fails some wrong-crop cases
(strawberry → tomato on ResNet; MobileNet is better but not solved — see OOD notes).

**Q11. How did you pick the 0.5 threshold?**
Sweep; pick by a coverage / precision-style rule on validation (PlantDoc val for stage 2).
Live pin: **0.50**, T=**0.885**. Not retuned after seeing the 48.2% final number.

**Q12. How do you prevent train/serve skew?**
Same processor config: resize short side 256, centre-crop 224, ImageNet mean/std. API crops
the leaf, pads square, resizes once to 256 so the Space does a clean crop.

**Q13. What would you do to improve real-world accuracy?**
Measure first (we did). Then: more phone-like data (keep test held out), background-robust
aug, crop-level head / negatives for OOD, calibration refresh — only then a bigger backbone.
We already tried ResNet; it lost on our field set.

**Q14. Why not object detection or segmentation?**
One leaf, one label. Boxes/masks are expensive. We do cheap colour+focus localisation only
for crop + gate.

**Q15. Why a stratified split with a seed?**
Every class in train/val/test in proportion; `splits.json` + seed 42 for reproducibility.
Near-dup leakage across splits is a known limit (random per-image, not per-leaf cluster).

**Q16. Why did ResNet lose the bake-off?**
Same 284 images, strategy B. MobileNet 137/284; ResNet v1.2 (sampler fix) 98/284; more
“not sure” on ResNet. Stronger on clean lab photos didn’t transfer. Talk sheet:
`docs/model-bakeoff.md`.

**Q17. What is `v1.2` vs `v1.1`?**
`v1.1` = first PlantDoc stage-2 release with the locked 284 one-shot (48.2%). `v1.2` =
clean ImageNet → PV → PlantDoc retrain with the PlantDoc sampler fix; Hub tag + live pin.
We did **not** re-burn the final test set — quote 48.2% from `v1.1` as the field number.

### Image quality gate

**Q18. How do you detect blur?**
Laplacian variance is resolution- and background-sensitive. We find the leaf, fix scale
(≤256), use a **re-blur ratio** plus an absolute variance floor (reject if
`ratio < 0.78 or variance < 25`).

**Q19. How do you find the leaf without a neural net?**
HSV vegetation mask, drop out-of-focus vegetation, morph, connected components, rank by
focus × size × centrality. Keeps the API image torch-free (~50 ms). CNN detector can
replace the same function later.

**Q20. What fails?**
Heavily brown/yellow leaves can fool the colour mask → whole-frame fallback. Thresholds
hand-calibrated on a blur sweep + ~40 PV images, not a labelled good/bad phone set.

**Q21. Why run the gate twice?**
`/quality-check` for fast UX; `/predict` re-runs server-side because the client isn’t
trusted.

**Q22. Why did the old gate reject sharp phone photos?**
Variance computed at native resolution. Same sharp leaf scored ~40 at 1024 px and ~360 at
256 px. Simulated 12 MP: old gate 0/40. Details in `docs/challenges-and-approach.md`.

### MLOps / publishing

**Q23. Why Hugging Face and not Kaggle for publishing?**
Data from Kaggle. Weights need versioned tags + `from_pretrained(..., revision=…)` + a free
container host (Spaces / our Docker). Kaggle Models doesn’t replace that for our loader.

**Q24. How do you version and reproduce?**
Seed, `splits.json`, `run_info.json`, model card, Hub tags (`v1.0` baseline, `v1.1` field
report, `v1.2` live). API/Space read `shared/release.json` (copy to `hf_space/release.json`
after pin changes). Don’t leave stale `HF_MODEL_*` in `.env`.

**Q25. Why not HF serverless inference?**
It doesn’t host this custom fine-tune (“Model not supported by provider hf-inference”).
We serve the pinned revision from `hf_space/`.

**Q26. How do you handle secrets?**
`.env` git-ignored; `.env.example` committed; Colab secrets; `publish.py --check` for write
token / owner. Never open `.env` on the projector.

**Q27. How did you avoid Colab’s stale-code problem?**
`notebooks/colab_sync.py` hard-resets to `origin/main`; don’t `files.upload()` training code.

**Q28. How do you prove which model answered?**
Space stamps `x-model-id` / `x-model-revision`. `/predict` returns `inference` with
`model_verified`. `/health?probe=true` must match the pin.

### Agentic / LLM

**Q-A1. Is this really agentic?**
Honest: orchestrated **workflow** with conditional routing (error, unsupported crop, retake,
not_sure, low confidence / mismatch), deterministic tools (classifier, KB lookup), one
generative step. We did **not** let the LLM choose tools in a loop — predictability over
autonomy for advice-like output. Stronger agentic option later: tool-calling chat.

**Q-A2. Why LangGraph?**
Explicit nodes/edges, conditional exits, typed state (`DiagnosisState`). Easy to show and
test (`test_agent_flow.py`).

**Q-A3. How do you stop hallucination?**
(1) LLM never diagnoses, (2) grounded in curated `diseases.json`, (3) prompt forbids product
names/doses, (4) low conf / mismatch → possible match + expert flag, (5) disclaimer,
(6) template fallback if LLM is down.

**Q-A4. How do you keep chat on-topic?**
Keyword pre-filter + system prompt that declines non-agri in one line. Keyword layer is
brittle (“bugs eating my cabbage” can miss); prompt is the second layer. Classifier guard =
future work.

**Q-A5. Why does the LLM never see the image?**
Separation + safety: specialised model decides, language model explains. Avoids vision-LLM
hallucination/cost. `generate_text` is text-only by construction.

**Q-A6. How is the LLM swappable?**
OpenAI-compatible client; change `LLM_MODEL` / `LLM_BASE_URL` (Grok: `https://api.x.ai/v1`).
Proposal said Grok first; default is OpenAI-shaped.

**Q-A7. What if the LLM is down?**
Structured fields from KB; `initial_message` falls back to template; chat has a fallback
reply. Rehearse demo **with and without** `LLM_API_KEY`.

**Q-A8. Where is memory/state?**
Browser `sessionStorage` sends `session_context` on each `/chat`. Server is stateless — no
diagnosis DB for the demo.

**Q-A9. How do you evaluate the LLM output?**
API tests with a fake LLM; provenance tracks whether the summary was rewritten. No formal
human LLM eval yet — say that if asked. Optional next: evaluator node vs KB.

**Q-A10. How would you add RAG?**
Replace dict lookup with embeddings over extension docs; cite sources. Corpus is small
enough for exact lookup today; KB still needs `sources` fields.

**Q-A11. Regional language?**
Shipped **Telugu** MVP (UI strings + glossary-backed translate), not the proposal’s full
Hindi/IndicTrans stack. Toggle in the UI; needs LLM for diagnosis TE. Don’t claim Hindi.

### System design

**Q29. Why separate inference-service from the API?**
Keeps torch out of the API image, scales independently, swap host via `HF_ENDPOINT_URL`.

**Q30. What if the model server is down?**
Graph → error node → “Analysis unavailable”. API doesn’t crash. Checklist item: stop
inference and verify.

**Q31. How would you scale this?**
Stateless API behind a load balancer, model replicas, queue for spikes, GPU or ONNX/int8
later, CDN for UI.

**Q32. Security / privacy?**
Demo: no auth, no image store, history only in the browser. Production: tight CORS,
rate-limit, auth, upload scanning, retention. Compose CORS is wide open on purpose for demo.

**Q33. Main risk of wrong advice?**
Misdiagnosis → wrong spray. Mitigations: expert-confirmation flags, disclaimer, no doses in
LLM text, top-3, crop mismatch / unsupported exits.

**Q34. How do you test?**
`api` pytest with fake classifier / fake LLM: gate blocks before model, healthy / low-conf /
HF failure / text-only prompt, release pin tests. Model quality: `evaluate.py`,
`final_test.py` (guarded — don’t re-run casually), compare bake-off artefacts.

**Q35. Browser vs Docker networking?**
Browser hits `http://localhost:8000`. API hits `http://inference-service:7860/` inside
compose. Different “localhost”s — classic trap. `NEXT_PUBLIC_API_URL` is a **build arg**;
change needs rebuild.

### Process / honesty

**Q36. How much did AI write?**
Be truthful about what you can explain line-by-line. Self-test at the bottom of this doc.
Trainers prefer honest + understood over polished fog.

**Q37. Biggest weakness?**
Field generalisation of a lab-trained model, and wrong-crop confidence. Both measured;
next step is crop head / negatives / more phone data — not flipping back to ResNet.

**Q38. What changed from the proposal?**
Next.js instead of Streamlit; OpenAI-compatible LLM; leaf-aware gate; Telugu MVP not full
Hindi; MobileNet kept after ResNet bake-off; compose has three services not “two.”

**Q39. What would you do with two more weeks?**
Cluster-based splits (leakage), real phone set, crop-level head, KB source citations,
learned gate thresholds, Hindi if marks demand it.

**Q40. Which numbers should I trust?**
Lab accuracy/F1 = in-distribution ceiling. **137/284 (48.2%)** = deployment-shaped honesty
for MobileNet stage-2. Threshold sweep = how often we say “not sure.” Don’t mix ResNet
lab flex with MobileNet field claim.

---

## Self-check night before

Without notes, can you:

1. Draw the three boxes (UI / API / inference) and the graph exits?
2. Explain re-blur ratio in one sentence?
3. Say why the LLM never sees the image?
4. Quote lab vs 137/284 without mixing them up?
5. Explain why MobileNet beat ResNet on *our* test?

If any of those stick, re-read challenges doc + this pitch, then re-run the checklist.
