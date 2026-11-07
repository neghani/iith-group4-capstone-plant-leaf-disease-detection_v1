# Project overview

**CropCare AI** — Capstone Project 7, Group 4 (IIIT-H).

A farmer photographs a crop leaf. We check the photo is usable, classify the disease with a
fine-tuned **MobileNetV2**, then turn that prediction into plain-language guidance with a
curated knowledge base and an optional text-only LLM. Follow-up chat stays on agriculture.

## Problem

Field diagnosis is slow and uneven. Phone photos are messy (blur, glare, clutter, wrong crop).
Lab-trained classifiers look strong on PlantVillage and fall over on web/field photos. Letting
an LLM “look at the leaf” invites hallucinated treatments.

## Goal (what we aimed for)

1. Image classification over 38 PlantVillage classes (14 crops).  
2. Honest handling of bad photos and low confidence (retake / not sure).  
3. Farmer-facing guidance grounded in curated facts — LLM explains, does not diagnose.  
4. Reproducible model pin on Hugging Face; runnable stack with Docker Compose.  
5. Report lab accuracy **and** a real-world / field-style number (proposal target ≥ 50%).

## What we shipped

| Layer | Deliverable |
|-------|-------------|
| Model | MobileNetV2 ImageNet → PlantVillage → PlantDoc mix; Hub `neghani/plantvillage-mobilenetv2` tags `v1.0` / `v1.1` / live **`v1.2`** |
| Inference | `hf_space/` FastAPI + `transformers.pipeline` (CPU) |
| Preprocess | Leaf-aware quality gate + crop strategy B (`shared/vision/`) |
| Agent | LangGraph: predict → knowledge → report → safety (+ error / unsupported / retake / not_sure) |
| API | FastAPI `/health`, `/quality-check`, `/predict`, `/chat`, smoke `/web-test`, `/local-test` |
| UI | Next.js CropCare AI (Figma-aligned); EN / **Telugu** toggle |
| Ops | `docker compose` three services; optional one-container profile |

**Out of scope for this demo:** login, image storage, server-side history DB.

## Numbers we stand behind

| Claim | Number | Notes |
|-------|--------|-------|
| Lab (PlantVillage held-out) | ~98–99% | In-distribution ceiling; near-dup leakage possible |
| Field-style one-shot | **137/284 (48.2%)** | MobileNet `@v1.1`, strategy B; below 50% — reported as-is |
| vs PlantVillage-only `v1.0` | 18.7% → 48.2% (+29.6 pp) | Same 284 protocol |
| vs ResNet on same test | MobileNet wins (~34% ResNet) | See `model-bakeoff.md` |
| Live pin | `@v1.2` | Same recipe + sampler fix; **no new 284 run** |

## Proposal vs what you see in the demo

| Proposal-ish | What we actually did |
|--------------|----------------------|
| Streamlit UI | Next.js + Tailwind (closer to Figma) |
| Grok first | OpenAI-compatible client; Grok via `LLM_BASE_URL` |
| Hindi / IndicTrans | **Telugu** MVP (UI + glossary translate); Hindi not built |
| Two services | Three Compose services (UI / API / inference) |
| MobileNet (brief) | Kept after ResNet bake-off lost on field set |

## Repo map

```text
api/          FastAPI + LangGraph + KB + smoke tests
web/          Next.js UI
hf_space/     inference container (loads Hub pin)
model/        train / eval / publish / real_world_set
shared/       release.json + vision gate/crop (API + offline share this)
docs/         these deliverables
notebooks/    Colab drivers
```

## Team note

Owners for Phase 1 model work are noted in `model/README.md` (proposal Annex A). Demo path
finalised as MobileNet only — do not put ResNet on the happy path unless someone asks about
the bake-off.
