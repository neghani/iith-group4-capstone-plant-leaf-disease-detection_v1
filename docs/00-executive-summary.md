# Executive summary

**CropCare AI** (IIIT-H Capstone Project 7, Group 4) helps a farmer diagnose a crop leaf from a
phone photo and get plain-language next steps.

**Pipeline:** quality gate → pinned **MobileNetV2** (`neghani/plantvillage-mobilenetv2@v1.2`) →
LangGraph + curated disease KB → optional **text-only** LLM. The vision model decides the
class; the LLM never sees the image and never invents the disease name.

**Results to quote**

| Setting | Number |
|---------|--------|
| Lab (PlantVillage held-out) | ~98–99% |
| Field-style locked set | **137/284 (48.2%)** (`@v1.1`; below 50% goal — reported as-is) |
| Bake-off | MobileNet beat ResNet (~34%) on the same 284 protocol |

**Stack:** FastAPI, LangGraph, Next.js (EN + Telugu MVP), Docker Compose, Hugging Face Hub pin
in `shared/release.json`.

**Honest limits:** lab ≠ field; soft OOD / wrong-crop confidence; KB citations thin; Hindi not
shipped; 48.2% &lt; proposal 50%.

**Training arc (for DL judges):** ImageNet → PlantVillage (`v1.0`, field 18.7%) → PlantDoc
stage 2 (`v1.1`, **48.2%**) → sampler-fixed live `v1.2`. Full write-up:
[09-training-story.md](09-training-story.md). Deck assets:
[10-slide-visuals.md](10-slide-visuals.md).

**Where to read more:** [docs/README.md](README.md).
