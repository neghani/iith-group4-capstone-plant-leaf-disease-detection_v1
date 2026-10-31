# Demo acceptance checklist

## Startup

- [ ] Copy `.env.example` to `.env` (optional `LLM_API_KEY` for richer text)
- [ ] Set `HF_ENDPOINT_URL` to the deployed Space, or leave it empty to use the local `inference-service`
- [ ] Run `docker compose up --build`
- [ ] `GET /health?probe=true` shows `hf_model_id` / `hf_model_revision` matching `shared/release.json` (currently `neghani/plantvillage-mobilenetv2@v1.1`), `hf_reachable: true` and `hf_served_model` the same pin
- [ ] Do **not** set `HF_MODEL_ID` / `HF_MODEL_REVISION` / `CALIBRATION_TEMPERATURE` in `.env` unless overriding the pin (stale values make Docker serve the wrong model)
- [ ] Strawberry demo photo with crop=tomato must **not** return `confirmed_likely` (expect `not_sure` or `possible`)
- [ ] Open UI at http://localhost:3000/check
- [ ] Open API docs at http://localhost:8000/docs

## End-to-end flows

- [ ] Upload a clear leaf image → quality check passes, then structured diagnosis with the model's top-3 matches, actions and monitoring
- [ ] Upload a large phone photo (10+ MB, 12 MP or more) of a clear leaf → accepted; results page footer shows the model id/revision ("verified by server")
- [ ] Pick "Other crop" and type a plant the model does not know (e.g. hibiscus) → "Crop not supported", no disease name
- [ ] Pick a crop that differs from the leaf (e.g. Potato for a tomato leaf) → "Possible match" with a mismatch note
- [ ] Upload a very dark, tiny or genuinely blurry image → results page “Please retake the photo”; `/predict` is never called
- [ ] On `/check`, **Show a retake** → dark demo photo through `/quality-check` → retake screen, no disease name
- [ ] On `/check`, **Show “I'm not sure”** → `/demo/not-sure.jpg` through `/predict` → “I'm not confident — please retake the photo.”, top matches as possibilities, no chat
- [ ] Healthy leaf → “Looks healthy”, no expert-confirmation warning
- [ ] Low-confidence case (confidence below 0.5) → status `not_sure`, not a disease write-up
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
4. Click **Show a retake**, then go back and click **Show “I'm not sure”** — abstaining is a feature.
5. Explain the flow: quality gate → Hugging Face MobileNetV2 (`v1.1`, 137/284 = 48.2% real-photo pass rate) → text-only LLM summary grounded in the prediction. Do not present ResNet as the live model: on the same 25-image final split it scores 2/25 against MobileNet's 10/25, and `neghani/plantvillage-resnet18` has no `v1.1` tag yet.
