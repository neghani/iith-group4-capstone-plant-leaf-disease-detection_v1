# Phase 2 demo acceptance checklist

## Startup

- [ ] Copy `.env.example` to `.env` (optional `LLM_API_KEY` for richer text)
- [ ] Run `docker compose up --build`
- [ ] Open UI at http://localhost:3000/check
- [ ] Open API docs at http://localhost:8000/docs

## End-to-end flows

- [ ] Upload a clear leaf image → structured diagnosis with actions and monitoring
- [ ] Upload a very dark or tiny image → quality message, no crash
- [ ] Low-confidence case (retry uploads) → “Possible disease” / confirmation warning
- [ ] Follow-up chat with agriculture question → contextual reply
- [ ] Follow-up chat with non-agriculture question → polite decline

## Out of scope (verify absent)

- [ ] No login or signup
- [ ] No image persistence, S3, or diagnosis history database

## Presentation script (2–3 min)

1. Show Figma alignment: upload panel + diagnosis panel + chat.
2. Analyze a tomato leaf sample; walk through signs, actions, limitations.
3. Ask one follow-up (“Should I remove lower leaves?”).
4. Mention placeholder model (Phase 5 swap) and optional LLM key.
