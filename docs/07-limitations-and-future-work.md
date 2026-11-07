# Limitations and future work

Say these first in the demo. Trainers will find them anyway.

## Known limitations

1. **Field pass rate 48.2% &lt; 50% goal** on the locked 284 one-shot. Reported honestly;
   not retuned after seeing the number.  
2. **Lab ~99% overstates phones** — PlantVillage near-duplicates + plain backgrounds.  
3. **PlantDoc / real_world_set are web photos** — gains overstate true phone-in-field.  
4. **Softmax is always confident** — threshold, top-3, OOD margin, crop hint help; they do
   not fully stop wrong-crop highs (e.g. strawberry→tomato style cases).  
5. **Healthy-leaf crop confusion** — grape↔soybean etc. on green leaves.  
6. **Lookalike diseases across crops** — tomato early blight ↔ potato early blight.  
7. **Tomato bacterial spot** weak on the field protocol (0/11 in that report).  
8. **Colour leaf finder** struggles on fully brown / yellow leaves → whole-frame fallback.  
9. **Gate thresholds** hand-calibrated (blur sweep + ~40 PV leaves), not a labelled ROC.  
10. **Two gate implementations** — live `shared/vision/` vs older `model/common.py` for some
    offline scripts. API is source of truth.  
11. **“Agentic” = fixed workflow** with conditional exits, not autonomous tool loops.  
12. **Chat topic guard** is keyword + prompt — brittle.  
13. **KB citations thin** — proposal promised stronger sourcing.  
14. **Telugu MVP, not Hindi** — differs from proposal language stack.  
15. **No auth / retention** — demo only.  
16. **HEIC** depends on browser conversion.  
17. **`v1.2` has no new 284 score** — don’t invent one.

## What we would do next (priority)

| Priority | Work | Why |
|----------|------|-----|
| 1 | Crop-level head or training negatives | Wrong-crop confidence |
| 2 | More real phone photos (held-out) | Close the web→phone gap |
| 3 | Cluster / pHash splits on PlantVillage | Leakage honesty |
| 4 | KB `sources` per disease | Proposal + trust |
| 5 | Learned gate thresholds on labelled good/bad phones | Replace hand floors |
| 6 | Tool-calling chat (lookup KB / prediction) | Stronger “agentic” story |
| 7 | Hindi path if marks require it | Beyond Telugu MVP |
| 8 | Background masking / field-style aug | Framing robustness |
| 9 | ONNX / int8 if latency/cost matters | Ops stretch |

**Not next:** flipping live pin back to ResNet without beating MobileNet on the same field
protocol.

## Open engineering notes

See `challenges-and-approach.md` (gate, sampler bug fix in `v1.2` train, OOD FALLBACK) and
`model/outputs/ood/FALLBACK.md` for the strawberry case study.
