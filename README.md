# AI-Powered Plant Disease Detection and Farmer Assistance

A farmer uploads a photo of a crop leaf. A quality gate checks the photo, the fine-tuned ResNet-18 on Hugging Face predicts the disease, and a LangGraph agent turns the prediction into treatment guidance with an optional text-only LLM. Follow-up chat stays in the agriculture domain.

Capstone Project 7, Group 4.

## UI Prototype

- Figma: [CropCare AI design](https://www.figma.com/design/jAzrVWopu8ohqh0vKY3ejr/Untitled?node-id=0-1)
- Interactive prototype: [Check a crop flow](https://www.figma.com/proto/jAzrVWopu8ohqh0vKY3ejr/Untitled?node-id=2-593)

## Project Status

- **Phase 1 (model):** Live pin is ResNet-18 `@v1.1` on [`neghani/plantvillage-resnet18`](https://huggingface.co/neghani/plantvillage-resnet18) (`shared/release.json`, threshold 0.55, T=0.936). MobileNetV2 `@v1.1` remains the previous pin; see [model/README.md](model/README.md)
- **Phase 2 (demo):** FastAPI + LangGraph agent, **Next.js (CropCare AI) UI**, Docker Compose — implemented
- **Phase 5:** Phase 1 Hugging Face model wired into the API (two-step quality gate, remote classifier, prediction-driven LLM) — implemented
- **Hindi UI / translation:** planned
- **Out of scope for demo:** authentication, persistent image storage, server-side history

## Tech Stack

| Layer | Choice | Role |
|---|---|---|
| API and core logic | [FastAPI](https://fastapi.tiangolo.com/) | `/health`, `/quality-check`, `/predict`, `/chat` |
| Agent flow | [LangGraph](https://langchain-ai.github.io/langgraph/) | Predict → knowledge → report → safety (quality gate runs before the graph) |
| Data validation | [Pydantic](https://docs.pydantic.dev/) | API and diagnosis report schemas |
| User interface | [Next.js](https://nextjs.org/) + Tailwind | Figma-aligned CropCare AI shell |
| Disease model | ResNet-18 `@v1.1` live pin ([Hugging Face](https://huggingface.co/neghani/plantvillage-resnet18)); MobileNetV2 `@v1.1` previous pin | 38 classes; pin + `ood_threshold` in `shared/release.json`; served by `hf_space/` |
| LLM | OpenAI ChatGPT API (optional) | Text only: farmer summary from the prediction, and chat; template fallback without key |
| Containers | Docker Compose | `inference-service` + `api-service` + `ui-service` |

## Architecture

1. Next.js (`/check`) shrinks big phone photos in the browser (long side 2048 px), then sends the image to `POST /quality-check`. The API decodes once, finds the sharpest leaf (colour + focus), and checks brightness and sharpness on that leaf at a fixed scale, so large photos are not penalised. On failure the farmer gets retake guidance and nothing else runs.
2. On pass, the UI sends the image to `POST /predict`. The API re-runs the same gate, then crops the best leaf to a 256 px square JPEG (the size the classifier's own preprocessing expects).
3. The API calls the Hugging Face classifier at `HF_ENDPOINT_URL` and gets the top-3 labels with scores. The response's `inference` block records which model/revision answered (`model_verified` = reported by the server itself). Below `CONFIDENCE_THRESHOLD` (0.55, the ResNet v1.1 threshold) the result is marked "Possible match" and needs expert confirmation.
4. Curated `api/data/knowledge/diseases.json` (all 38 classes) grounds symptoms, actions and monitoring. If the farmer's crop hint disagrees with the predicted crop the result is downgraded; a crop the model was never trained on gets a "Crop not supported" report instead of a disease name.
5. The LLM gets only text: the top-3 predictions, the confidence level, the knowledge entry, and the farmer's notes. It writes the farmer summary. The image is never sent to the LLM.
6. Results and history live in **sessionStorage** only; `POST /chat` uses the prediction context (diagnosis, confidence, top-3) from the client.

## Repository Layout

```text
.
├── api/
├── web/                 # Next.js CropCare AI UI
├── hf_space/            # inference server for the HF model (Space or local container)
├── model/               # Phase 1: train / evaluate / push MobileNetV2 to Hugging Face
├── notebooks/           # Colab notebook driving model/ scripts
├── docs/
├── docker-compose.yml
├── .env.example
└── README.md
```

## Getting Started

### Prerequisites

- Docker and Docker Compose, or Python 3.11+ and Node 20+
- Optional: `LLM_API_KEY` from [OpenAI](https://platform.openai.com/)

### Configuration

```bash
cp .env.example .env
```

Set `LLM_API_KEY` for LLM-polished summaries and chat. Set `NEXT_PUBLIC_API_URL=http://localhost:8000` so the browser can reach the API when using Docker.

The disease model is called over HTTP at `HF_ENDPOINT_URL`. Hugging Face serverless inference does not host this fine-tune, so either deploy [`hf_space/`](hf_space/README.md) as a Space and set `HF_ENDPOINT_URL=https://<space-host>/`, or leave it empty with Docker Compose to use the local `inference-service`.

### Run with Docker

Three containers (UI :3000, API :8000, model :7860):

```bash
docker compose up --build
```

The inference container loads **`shared/release.json`** (copied into `hf_space/release.json` at build time). Do not set `HF_MODEL_ID` / `HF_MODEL_REVISION` in `.env` unless you are deliberately overriding that pin — stale values will make Docker serve the wrong Hub model while the API pin still reads `release.json`.

After you change the live pin:

```bash
cp shared/release.json hf_space/release.json
docker compose build --no-cache
docker compose up
```

Check: `curl -s http://localhost:7860/health` and `curl -s 'http://localhost:8000/health?probe=true'` should show the same `hf_model_id` / revision as `shared/release.json`.

All three services in **one** container (one port, same as Render):

```bash
docker compose --profile all up --build cropcare
```

- **Farmer UI:** http://localhost:3000 — `/check` upload, `/results` diagnosis + chat
- **API:** http://localhost:8000 (three-container) or http://localhost:3000/docs (one-container)

### Deploy one machine on Render

Language **Docker**, Dockerfile Path **`Dockerfile`**, context **repo root**. Or connect the repo and apply `render.yaml`. First build pulls the Hub model into the image; use a **Standard** (2 GB) instance — starter 512 MB will OOM. After deploy, open `https://<service>.onrender.com/check`.

### Deploy one machine on Railway

New project → this GitHub repo. [`railway.toml`](railway.toml) selects the root Dockerfile, health-checks `/health`, and waits up to 5 minutes for the model to load. Generate a domain, set RAM to **at least 2 GB**, and optionally add `LLM_API_KEY`. Open `https://<service>.up.railway.app/check`.

### Run locally (without Docker)

```bash
# Terminal 1 — model inference server (needs torch + transformers)
cd hf_space && pip install -r requirements.txt && uvicorn app:app --port 7860

# Terminal 2 — API
cd api && source .venv/bin/activate  # create venv + pip install -r requirements.txt first
HF_ENDPOINT_URL=http://localhost:7860/ PYTHONPATH=. uvicorn app.main:app --reload --port 8000

# Terminal 3 — UI
cd web && npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open http://localhost:3000

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health, LLM key configured, HF model id / revision / endpoint configured; `?probe=true` also asks the inference server which model it serves |
| POST | `/quality-check` | Step 1: multipart image → `{ ok, message, brightness, blur_score, leaf_found, leaf_coverage, original_size }` (no model call) |
| POST | `/predict` | Step 2: multipart image → gate re-check, HF prediction (`top_k`), structured diagnosis report, `inference` provenance |
| POST | `/chat` | JSON `{ "message", "session_context" }` → agriculture-focused reply |

## Tests

```bash
cd api && pip install -r requirements-dev.txt
PYTHONPATH=..:. pytest -q
```

## Design notes

Challenges hit and the approach taken (large-photo handling, HF model verification, how the prediction drives the agent, open issues): [docs/challenges-and-approach.md](docs/challenges-and-approach.md).

## Demo checklist

See [docs/phase2-demo-checklist.md](docs/phase2-demo-checklist.md).

## Disclaimer

Treatment suggestions are for general guidance only and must be validated with local agricultural experts before critical action.
