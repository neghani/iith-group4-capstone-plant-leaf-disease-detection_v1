# Demo acceptance checklist

## Startup

- [ ] Copy `.env.example` to `.env` (optional `LLM_API_KEY` for richer text)
- [ ] Set `HF_ENDPOINT_URL` to the deployed Space, or leave it empty to use the local `inference-service`
- [ ] Run `docker compose up --build`
- [ ] `GET /health?probe=true` shows `hf_model_id` / `hf_model_revision` matching `shared/release.json` (currently `neghani/plantvillage-mobilenetv2@v1.1`), `hf_reachable: true` and `hf_served_model` the same pin
- [ ] Open UI at http://localhost:3000/check
- [ ] Open API docs at http://localhost:8000/docs

## End-to-end flows

- [ ] Upload a clear leaf image → quality check passes, then structured diagnosis with the model's top-3 matches, actions and monitoring
- [ ] Upload a large phone photo (10+ MB, 12 MP or more) of a clear leaf → accepted; results page footer shows the model id/revision ("verified by server")
- [ ] Pick "Other crop" and type a plant the model does not know (e.g. hibiscus) → "Crop not supported", no disease name
- [ ] Pick a crop that differs from the leaf (e.g. Potato for a tomato leaf) → "Possible match" with a mismatch note
- [ ] Upload a very dark, tiny or genuinely blurry image → retake message from `/quality-check`; `/predict` is never called
- [ ] Healthy leaf → “Looks healthy”, no expert-confirmation warning
- [ ] Low-confidence case (confidence below 0.5, e.g. a non-PlantVillage crop) → “Possible match” / confirmation warning
- [ ] Stop the inference service → “Analysis unavailable”, no crash
- [ ] Follow-up chat with agriculture question → contextual reply
- [ ] Follow-up chat with non-agriculture question → polite decline

## Out of scope (verify absent)

- [ ] No login or signup
- [ ] No image persistence, S3, or diagnosis history database

## Presentation script (2–3 min)

1. Show Figma alignment: upload panel + diagnosis panel + chat.
2. Analyze a tomato leaf sample; walk through signs, actions, limitations.
3. Ask one follow-up (“Should I remove lower leaves?”).
4. Explain the flow: quality gate → Hugging Face MobileNetV2 (99.1% test accuracy) → text-only LLM summary grounded in the prediction.
