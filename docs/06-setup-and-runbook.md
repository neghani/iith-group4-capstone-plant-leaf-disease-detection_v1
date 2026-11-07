# Setup and runbook

How we run CropCare AI for demos and local work. Pin must match `shared/release.json`.

## Prerequisites

- Docker + Docker Compose **or** Python 3.11+, Node 20+  
- Optional: `LLM_API_KEY` (OpenAI-compatible) for polished summaries / Telugu / chat  
- Optional: Hugging Face token only if the Space/endpoint is private  

## Configure

```bash
cp .env.example .env
# Set LLM_API_KEY if you want LLM text
# NEXT_PUBLIC_API_URL=http://localhost:8000 for Compose UI → host API
```

**Do not** set `HF_MODEL_ID` / `HF_MODEL_REVISION` / `CALIBRATION_TEMPERATURE` in `.env`
unless you intentionally override the pin. Stale values desync Docker from `release.json`.

After changing the pin:

```bash
cp shared/release.json hf_space/release.json
docker compose build --no-cache inference-service   # or full --build
docker compose up
```

## Option A — Docker (demo default)

```bash
docker compose up --build
```

| URL | What |
|-----|------|
| http://localhost:3000/check | Farmer UI |
| http://localhost:8000/docs | API docs |
| http://localhost:8000/health?probe=true | Pin + inference alive |

First start pulls Hub weights into the inference container — give it a minute.

`real_world_set` is mounted for Local Test. Images are often git-ignored; keep them on disk.

One-box deploy profile:

```bash
docker compose --profile all up --build cropcare
```

## Option B — three terminals (debug)

```bash
# 1) inference
cd hf_space && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 7860

# 2) API
cd api && source .venv/bin/activate   # pip install -r requirements.txt first
HF_ENDPOINT_URL=http://localhost:7860/ PYTHONPATH=..:. \
  uvicorn app.main:app --reload --port 8000

# 3) UI
cd web && npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## Tests before demo

```bash
cd api && source .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH=..:. pytest -q
```

## Day-of ops

1. Warm stack 10+ minutes early (`/health?probe=true`).  
2. Pre-test demo images on this machine.  
3. Decide LLM-on vs KB-fallback and rehearse that mode.  
4. Never open `.env` on the projector.  
5. Backup: screen recording of a good run.  
6. Local Test nav: `localStorage.setItem("local-test", "true")` then refresh.  

Full tick list: `phase2-demo-checklist.md`.

## Common traps

| Symptom | Likely cause |
|---------|----------------|
| Wrong model in UI footer | Stale `HF_MODEL_*` in `.env` or forgot copy to `hf_space/release.json` |
| UI still hits old API URL | `NEXT_PUBLIC_*` baked at build — rebuild UI |
| 503 then success | Space cold start — client retries once; warm earlier |
| Local Test “no images” | `images/` missing under `model/data/real_world_set` |
| Browser can’t reach model | Browser uses `localhost:8000`; API uses `inference-service:7860` |

## Training (not needed for demo)

Smoke / Colab / publish: see `model/README.md` and `notebooks/`. Demo only needs inference CPU.
