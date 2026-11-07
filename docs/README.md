# Capstone deliverables (Group 4)

IIIT-H Capstone Project 7 — **AI-Powered Plant Disease Detection and Farmer Assistance**
(CropCare AI). Live demo path: **MobileNetV2** only
(`neghani/plantvillage-mobilenetv2@v1.2` in [`shared/release.json`](../shared/release.json)).

These files are what we hand in / point trainers at. Keep numbers honest: lab ~98–99%,
field one-shot **137/284 (48.2%)** from MobileNet `@v1.1` (locked set; `@v1.2` not re-burned).

## What to submit / show

| # | Document | Use |
|---|----------|-----|
| 0 | [00-executive-summary.md](00-executive-summary.md) | One-pager for coordinators |
| 1 | [01-project-overview.md](01-project-overview.md) | Problem, scope, pitch, what shipped |
| 2 | [02-architecture.md](02-architecture.md) | System + request flow diagrams |
| 3 | [03-design-decisions.md](03-design-decisions.md) | Why MobileNet, HF, LangGraph, Next.js, … |
| 4 | [challenges-and-approach.md](challenges-and-approach.md) | Engineering log (gate, field cliff, bake-off) |
| 5 | [model-bakeoff.md](model-bakeoff.md) | MobileNet vs ResNet talk sheet |
| 6 | [04-evaluation-report.md](04-evaluation-report.md) | Metrics to quote; what not to mix |
| 7 | [09-training-story.md](09-training-story.md) | **Lab → PlantDoc → field** (DL audience) |
| 8 | [10-slide-visuals.md](10-slide-visuals.md) | Charts + product screenshots for the deck |
| 9 | [05-api-and-user-guide.md](05-api-and-user-guide.md) | Farmer UI + API contract |
| 10 | [06-setup-and-runbook.md](06-setup-and-runbook.md) | Docker / local run, pin rules |
| 11 | [07-limitations-and-future-work.md](07-limitations-and-future-work.md) | Honest gaps + next steps |
| 12 | [08-submission-checklist.md](08-submission-checklist.md) | Hand-in / packaging ticks |
| 13 | [demo-day-prep-guide.md](demo-day-prep-guide.md) | Pitch, Q&A, rehearsal |
| 14 | [phase2-demo-checklist.md](phase2-demo-checklist.md) | Day-of ticks |
| — | [Plant_Disease_Detection_Proposal_v1.docx](Plant_Disease_Detection_Proposal_v1.docx) | Original proposal (note drift in overview) |

## Source of truth (when docs disagree)

1. **`shared/release.json`** — live model id, revision, threshold, T, crop strategy, OOD  
2. **Code** under `api/`, `shared/vision/`, `hf_space/`, `web/`  
3. **`model/final_report.md`** — locked 284-image field numbers  
4. These `docs/` notes  

Root [`README.md`](../README.md) matches the MobileNet pin; if anything drifts, trust `shared/release.json`.

## One-line story for the folder

Farmer photo → quality gate → MobileNet (pinned) → LangGraph + curated KB → text-only LLM.
Vision decides; language explains. We measured field accuracy and kept MobileNet after a
ResNet bake-off.
