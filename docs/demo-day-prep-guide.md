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

## Trainer questions — short answers

**Why MobileNet?** Brief preference, CPU-friendly, and it won our field bake-off vs ResNet.

**Why not freeze the backbone?** Full fine-tune; lesions differ from ImageNet. Two-stage
warmup is a fair next experiment; we didn’t run it.

**Why macro-F1?** Class counts ~150–5.5k. Accuracy hides weak small classes.

**99% too good?** Yes for deployment. Near-dups + lab background. Trust 48.2% on 284.

**Softmax overconfident?** Always picks something. Threshold, top-3, crop hint, OOD margin,
gate. Still fails some wrong-crop cases (strawberry).

**Threshold 0.5?** Sweep on in-distribution / PlantDoc val rules. Tuned before final test;
not retuned after seeing 48.2%.

**Train/serve skew?** API leaf crop → 256 square; processor centre-crops 224. Same as
`preprocessor_config.json`.

**Blur detection?** Leaf-only re-blur ratio + variance floor at fixed scale. Whole-frame
Laplacian depended on resolution.

**Leaf without a CNN?** HSV + focus filter + components. Fast, keeps API torch-free.
Fails on fully brown leaves → whole-frame fallback.

**Why HF not Kaggle Models for publish?** Versioned tags + `from_pretrained(revision=…)` +
Spaces. Data came from Kaggle.

**Why own container?** HF serverless: “model not supported” for this fine-tune.

**Is it agentic?** Orchestrated workflow with tools (classifier, KB) and conditional exits.
We didn’t give the LLM free tool choice on purpose — advice-like domain.

**LLM never sees the image?** Specialised vision model decides; language model explains.
Also cost / hallucination control. Enforced in code.

**LLM down?** Structured fields from KB; template summary; chat fallback.

**Biggest weakness?** Field generalisation + wrong-crop confidence. Measured; next step is
crop head / negatives, not another backbone flip.

**Changed from proposal?** Next.js, OpenAI-compatible LLM, leaf-aware gate, Telugu not
Hindi, MobileNet kept after ResNet bake-off.

**Two more weeks?** Cluster splits for leakage, phone-photo set, crop-level head, KB
sources, Hindi if marks need it.

---

## Self-check night before

Without notes, can you:

1. Draw the three boxes (UI / API / inference) and the graph exits?
2. Explain re-blur ratio in one sentence?
3. Say why the LLM never sees the image?
4. Quote lab vs 137/284 without mixing them up?
5. Explain why MobileNet beat ResNet on *our* test?

If any of those stick, re-read challenges doc + this pitch, then re-run the checklist.
