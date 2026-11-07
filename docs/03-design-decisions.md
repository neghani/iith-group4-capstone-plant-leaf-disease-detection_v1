# Design decisions

Why we built it this way. Short answers for the write-up and viva. Details and scars live in
`challenges-and-approach.md` and `demo-day-prep-guide.md`.

| ID | Decision | Chose | Why | Trade-off |
|----|----------|-------|-----|-----------|
| D1 | Task | Image classification | One leaf → one of 38 labels; matches brief | No lesion localisation |
| D2 | Data | PlantVillage + PlantDoc stage-2 | Coordinator data + field-style mix | Lab bias; PlantDoc is web-sourced |
| D3 | Train env | Kaggle data, Colab Pro train | GPU + asked-for notebook | Session resets; mitigated with sync + Hub push |
| D4 | Publish | HF Hub tags + Spaces/container | `revision=` pin + free host | Weights off GitHub |
| D5 | Backbone (live) | **MobileNetV2** | Brief + **won 284 bake-off** vs ResNet | ResNet nicer on clean PV; lost on field set |
| D6 | Train stack | HF `Trainer` | Checkpoints, fp16, push-to-hub | Less “from scratch” transparency |
| D7 | Fine-tune | Full network | Lesions ≠ ImageNet textures; enough data | Overfit risk; watch val |
| D8 | Selection metric | Macro-F1 (PlantDoc val in stage 2) | Class imbalance | Tiny support classes noisy |
| D9 | Confidence | Threshold 0.5 + top-3 + possible match | Honesty | Tuned before final; not retuned after 48.2% |
| D10 | Quality gate | Leaf-aware re-blur | Whole-frame Laplacian rejected sharp phones | Hand-calibrated thresholds |
| D11 | Inference host | Own container / Space | Serverless won’t host this fine-tune | Cold starts |
| D12 | UI | Next.js | Figma / richer UX | Differs from Streamlit proposal |
| D13 | LLM | OpenAI-compatible | Swap provider via env | Proposal said Grok first |
| D14 | Agent | LangGraph workflow | Auditable exits | Not autonomous tool-picking |
| D15 | LLM role | Explains only | Safety | Can’t use visual cues classifier missed |
| D16 | Knowledge | Curated JSON | Deterministic actions | Citations still thin |
| D17 | State | Client sessionStorage | No auth / PII store for demo | Lost on tab close |
| D18 | Packaging | Compose 3 services | Separate torch lifecycle | Multi-container ops |
| D19 | Versioning | `shared/release.json` | One pin for API + Space | Must copy to `hf_space/` after bump |
| D20 | Regional language | Telugu MVP | Marks + shippable | Not proposal Hindi/IndicTrans |
| D21 | Crop strategy | B (sharpest leaf) | Measured vs A–E; kept unless large gain | Multi-leaf scenes still hard |
| D22 | OOD | Margin threshold 0.05 | Flat softmax → not_sure | Doesn’t catch all wrong-crop highs |

## Notes we keep repeating

**Why not ResNet on the demo path?** Same 284 protocol: MobileNet 48.2%, ResNet ~34%. Lab
flex doesn’t transfer. Talk sheet: `model-bakeoff.md`.

**Why separate inference?** API stays numpy/scipy; torch image stays small and restartable.

**Why not let the LLM see the image?** Medical-advice-like risk. Classifier is the specialist;
LLM rewrites grounded text.

**Why “agentic” without free tool loops?** Predictable routing for a domain where wrong
spray advice hurts. Next step toward more agency: tool-calling chat — not for this demo pin.
