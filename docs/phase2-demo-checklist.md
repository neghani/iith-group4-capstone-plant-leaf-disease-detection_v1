# Demo acceptance checklist

MobileNet path only. Pin must match `shared/release.json`
(`neghani/plantvillage-mobilenetv2@v1.2`).

## Startup

- [ ] Copy `.env.example` → `.env` (optional `LLM_API_KEY` for richer text / Telugu)
- [ ] Leave `HF_ENDPOINT_URL` empty for local `inference-service`, or point at the Space
- [ ] Do **not** set `HF_MODEL_ID` / `HF_MODEL_REVISION` / `CALIBRATION_TEMPERATURE` in `.env`
      unless you mean to override the pin (stale values desync Docker from `release.json`)
- [ ] If the pin changed: `cp shared/release.json hf_space/release.json` then rebuild
- [ ] `docker compose up --build`
- [ ] `GET /health?probe=true` → `hf_model_id` / `hf_model_revision` =
      `neghani/plantvillage-mobilenetv2` / `v1.2`, `hf_reachable: true`, `hf_served_model` same
- [ ] Strawberry (or similar) + crop=tomato must **not** land as `confirmed_likely`
      (expect `not_sure` / `possible` via margin or crop mismatch)
- [ ] UI http://localhost:3000/check
- [ ] API docs http://localhost:8000/docs

## End-to-end flows

- [ ] Clear leaf → quality pass → structured diagnosis, top-3, actions, monitoring
- [ ] Large phone photo (10+ MB / 12 MP if you have one) → accepted; results footer shows
      model id/revision verified
- [ ] “Other crop” + hibiscus (or anything outside the 14) → “Crop not supported”, no disease name
- [ ] Wrong crop hint (Potato on a tomato leaf) → “Possible match” + mismatch note
- [ ] Dark / tiny / genuinely blurry → retake UI; `/predict` never needed for that path
- [ ] `/check` → **Show a retake** → retake screen, no disease name
- [ ] `/check` → **Show “I'm not sure”** → `/demo/not-sure.jpg` → not confident / retake copy,
      top matches as possibilities, no chat push
- [ ] Healthy leaf → “Looks healthy”, no expert-confirmation scare
- [ ] Low confidence (< 0.5) → `not_sure`, not a full disease write-up
- [ ] Stop inference service → “Analysis unavailable”, API stays up
- [ ] Chat: agriculture follow-up → contextual
- [ ] Chat: non-agriculture → polite decline
- [ ] Language toggle → Telugu on a successful diagnosis (needs LLM; else English stays + note)
- [ ] Optional tutor toy: `/web-test` one row → live MobileNet badge + latency
- [ ] Optional: `/local-test` → 10 random `real_world_set` images (gate + abstain + GT crop hint)

## Out of scope (confirm absent)

- [ ] No login / signup
- [ ] No image persistence, S3, or diagnosis history DB

## Day-of script (2–3 min live)

1. Figma alignment: upload + diagnosis + chat.
2. Tomato sample through MobileNet `@v1.2`; walk signs / actions / limits.
3. One follow-up (“Should I remove lower leaves?”).
4. **Show a retake** then **Show “I'm not sure”** — abstaining is a feature.
5. Say the flow out loud: quality gate → **MobileNetV2 `v1.2`** (field honesty number is
   still the `v1.1` one-shot **137/284 = 48.2%**; we did not burn that set again for `v1.2`)
   → text-only LLM grounded in the prediction. ResNet lost the same-test bake-off (~34%) —
   only mention if asked.

## Night-before

- [ ] `pytest` green in api venv (`PYTHONPATH=..:.`)
- [ ] Demo image folder pre-tested on this machine
- [ ] Backup screen recording if Wi-Fi dies
- [ ] Pitch + 48.2% + “LLM never sees the image” without notes
