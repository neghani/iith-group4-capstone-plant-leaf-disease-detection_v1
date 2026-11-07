# AI-Powered Plant Disease Detection and Farmer Assistance

A farmer uploads a photo of a crop leaf. A quality gate checks the photo, a fine-tuned
**MobileNetV2** on Hugging Face predicts the disease, and a LangGraph agent turns the
prediction into treatment guidance with an optional text-only LLM. Follow-up chat stays in
the agriculture domain.

Capstone Project 7, Group 4 (CropCare AI).

**Documentation pack:** [docs/README.md](docs/README.md) — overview, architecture, evaluation,
API/user guide, runbook, limitations, demo Q&A.

## UI Prototype

- Figma: [CropCare AI design](https://www.figma.com/design/jAzrVWopu8ohqh0vKY3ejr/Untitled?node-id=0-1)
- Interactive prototype: [Check a crop flow](https://www.figma.com/proto/jAzrVWopu8ohqh0vKY3ejr/Untitled?node-id=2-593)

## Project Status

- **Phase 1 (model):** Live pin is **MobileNetV2 `@v1.2`** on
  [`neghani/plantvillage-mobilenetv2`](https://huggingface.co/neghani/plantvillage-mobilenetv2)
  (`shared/release.json`: threshold **0.5**, T=**0.885**, crop strategy **B**, OOD **0.05**).
  ResNet-18 was tried in a bake-off and lost on the field set — see
  [docs/model-bakeoff.md](docs/model-bakeoff.md).
- **Field-style result (locked):** **137/284 (48.2%)** on MobileNet `@v1.1` (same protocol;
  `@v1.2` not re-burned). Lab PlantVillage ~98–99% — do not mix the two.
- **Phase 2 (demo):** FastAPI + LangGraph, Next.js UI, Docker Compose — implemented
- **Localisation:** English + **Telugu** MVP (Hindi not shipped)
- **Out of scope for demo:** authentication, persistent image storage, server-side history

## Tech Stack

| Layer | Choice | Role |
|---|---|---|
| API and core logic | [FastAPI](https://fastapi.tiangolo.com/) | `/health`, `/quality-check`, `/predict`, `/chat` |
| Agent flow | [LangGraph](https://langchain-ai.github.io/langgraph/) | Predict → knowledge → report → safety (quality gate runs before the graph) |
| Data validation | [Pydantic](https://docs.pydantic.dev/) | API and diagnosis report schemas |
| User interface | [Next.js](https://nextjs.org/) + Tailwind | Figma-aligned CropCare AI shell; EN / Te |
| Disease model | MobileNetV2 `@v1.2` ([Hugging Face](https://huggingface.co/neghani/plantvillage-mobilenetv2)) | 38 classes; pin in `shared/release.json`; served by `hf_space/` |
| LLM | OpenAI-compatible API (optional) | Text only: farmer summary + chat; KB/template fallback without key |
| Containers | Docker Compose | `inference-service` + `api-service` + `ui-service` |

## Architecture (short)

1. Next.js (`/check`) shrinks large photos in the browser, then `POST /quality-check`. The API
   finds the sharpest leaf and checks brightness / blur; on fail → retake, no model call.
2. On pass, `POST /predict` re-runs the gate, crops strategy **B** to a 256 px square, and
   calls the inference service (`HF_ENDPOINT_URL`).
3. Top-3 + scores come back; below `confidence_threshold` (0.5) → “Possible match” /
   not-sure path. Crop-hint mismatch or unsupported crop downgrades the report.
4. Curated `api/data/knowledge/diseases.json` grounds symptoms and actions.
5. The LLM receives **text only** (never the image). Chat is topic-guarded to agriculture.
6. Results / history stay in **sessionStorage** on the client.

Full diagrams: [docs/02-architecture.md](docs/02-architecture.md).

## Repository Layout

```text
.
├── api/                 # FastAPI + LangGraph + KB
├── web/                 # Next.js CropCare AI UI
├── hf_space/            # inference server (Hub pin)
├── model/               # train / evaluate / push MobileNetV2
├── shared/              # release.json + vision gate/crop
├── notebooks/           # Colab drivers
├── docs/                # capstone deliverables
├── docker-compose.yml
├── .env.example
└── README.md
```

## Getting Started

### Prerequisites

- Docker and Docker Compose, or Python 3.11+ and Node 20+
- Optional: `LLM_API_KEY` from [OpenAI](https://platform.openai.com/) (or compatible base URL)

### Configuration

```bash
cp .env.example .env
```

Set `LLM_API_KEY` for polished summaries and chat. Set
`NEXT_PUBLIC_API_URL=http://localhost:8000` so the browser reaches the API under Docker.

**Do not** set `HF_MODEL_ID` / `HF_MODEL_REVISION` / `CALIBRATION_TEMPERATURE` in `.env`
unless you intentionally override the pin — stale values desync Docker from `release.json`.

### Run with Docker

```bash
docker compose up --build
```

| URL | What |
|-----|------|
| http://localhost:3000/check | Farmer UI |
| http://localhost:8000/docs | OpenAPI |
| http://localhost:8000/health?probe=true | Pin + inference alive |

After changing the live pin:

```bash
cp shared/release.json hf_space/release.json
docker compose build --no-cache
docker compose up
```

One-container profile (Render-style):

```bash
docker compose --profile all up --build cropcare
```

Longer ops notes: [docs/06-setup-and-runbook.md](docs/06-setup-and-runbook.md).

### Deploy (one machine)

- **Render:** Docker, root `Dockerfile`, ≥ **2 GB** RAM (starter OOM).
- **Railway:** `railway.toml`; generate domain; ≥ 2 GB; optional `LLM_API_KEY`.

### Run locally (without Docker)

```bash
# Terminal 1 — inference
cd hf_space && pip install -r requirements.txt && uvicorn app:app --port 7860

# Terminal 2 — API
cd api && source .venv/bin/activate
HF_ENDPOINT_URL=http://localhost:7860/ PYTHONPATH=..:. \
  uvicorn app.main:app --reload --port 8000

# Terminal 3 — UI
cd web && npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health + pin; `?probe=true` verifies inference server |
| POST | `/quality-check` | Gate only (no model) |
| POST | `/predict` | Gate + classify + diagnosis report |
| POST | `/chat` | Agriculture-focused follow-up |

Details: [docs/05-api-and-user-guide.md](docs/05-api-and-user-guide.md).

## Tests

```bash
cd api && pip install -r requirements-dev.txt
PYTHONPATH=..:. pytest -q
```

## Capstone docs

| Doc | Purpose |
|-----|---------|
| [docs/README.md](docs/README.md) | Index of all deliverables |
| [docs/00-executive-summary.md](docs/00-executive-summary.md) | One-pager |
| [docs/01-project-overview.md](docs/01-project-overview.md) | Problem / scope / what shipped |
| [docs/02-architecture.md](docs/02-architecture.md) | System diagrams |
| [docs/03-design-decisions.md](docs/03-design-decisions.md) | Why these choices |
| [docs/09-training-story.md](docs/09-training-story.md) | Lab → PlantDoc → field (DL story) |
| [docs/10-slide-visuals.md](docs/10-slide-visuals.md) | Charts + product screenshots |
| [docs/04-evaluation-report.md](docs/04-evaluation-report.md) | Metrics to quote |
| [docs/05-api-and-user-guide.md](docs/05-api-and-user-guide.md) | UI + API |
| [docs/06-setup-and-runbook.md](docs/06-setup-and-runbook.md) | Run / demo ops |
| [docs/07-limitations-and-future-work.md](docs/07-limitations-and-future-work.md) | Gaps + next |
| [docs/08-submission-checklist.md](docs/08-submission-checklist.md) | Hand-in ticks |
| [docs/challenges-and-approach.md](docs/challenges-and-approach.md) | Engineering log |
| [docs/demo-day-prep-guide.md](docs/demo-day-prep-guide.md) | Pitch + Q&A |
| [docs/phase2-demo-checklist.md](docs/phase2-demo-checklist.md) | Day-of checklist |

## Disclaimer

Treatment suggestions are for general guidance only and must be validated with local
agricultural experts before critical action.
