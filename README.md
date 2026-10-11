# AI-Powered Plant Disease Detection and Farmer Assistance

A farmer uploads a photo of a crop leaf. The system predicts the disease (placeholder classifier in Phase 2) and provides treatment guidance via a LangGraph agent and optional LLM. Follow-up chat stays in the agriculture domain.

Capstone Project 7, Group 4.

## UI Prototype

- Figma: [CropCare AI design](https://www.figma.com/design/jAzrVWopu8ohqh0vKY3ejr/Untitled?node-id=0-1)
- Interactive prototype: [Check a crop flow](https://www.figma.com/proto/jAzrVWopu8ohqh0vKY3ejr/Untitled?node-id=2-593)

## Project Status

- **Phase 1 (model):** MobileNetV2 on PlantVillage — training/evaluation/real-world scripts and Colab notebook ready, publishes to [`neghani/plantvillage-mobilenetv2`](https://huggingface.co/neghani/plantvillage-mobilenetv2); see [model/README.md](model/README.md)
- **Phase 2 (demo):** FastAPI + LangGraph agent, **Next.js (CropCare AI) UI**, Docker Compose — implemented
- **Phase 5:** Replace placeholder prediction with the Phase 1 Hugging Face model — planned
- **Hindi UI / translation:** planned
- **Out of scope for demo:** authentication, persistent image storage, server-side history

## Tech Stack

| Layer | Choice | Role |
|---|---|---|
| API and core logic | [FastAPI](https://fastapi.tiangolo.com/) | `/health`, `/predict`, `/chat` |
| Agent flow | [LangGraph](https://langchain-ai.github.io/langgraph/) | Quality → predict → knowledge → report → safety |
| Data validation | [Pydantic](https://docs.pydantic.dev/) | API and diagnosis report schemas |
| User interface | [Next.js](https://nextjs.org/) + Tailwind | Figma-aligned CropCare AI shell |
| Disease model | MobileNetV2 fine-tuned on PlantVillage, hosted on [Hugging Face](https://huggingface.co/neghani/plantvillage-mobilenetv2) (Phase 1, `model/`) | API still uses a hash-based placeholder until the Phase 5 swap |
| LLM | OpenAI ChatGPT API (optional) | Summary polish and chat; template fallback without key |
| Containers | Docker Compose | `api-service` + `ui-service` |

## Architecture

1. Next.js (`/check`) sends the image to `POST /predict`.
2. API validates file type, size, blur, and brightness.
3. Placeholder predictor returns disease id + confidence.
4. Curated `api/data/knowledge/diseases.json` grounds actions and symptoms.
5. Optional LLM rewrites the farmer summary; chat uses the same provider when configured.
6. Results and history live in **sessionStorage** only; `POST /chat` uses diagnosis context from the client.

## Repository Layout

```text
.
├── api/
├── web/                 # Next.js CropCare AI UI
├── ui/                  # legacy Streamlit (deprecated)
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

### Run with Docker

```bash
docker compose up --build
```

- **Farmer UI:** http://localhost:3000 — `/check` upload, `/results` diagnosis + chat
- **API:** http://localhost:8000 — `/docs`, `/health`

### Run locally (without Docker)

```bash
# Terminal 1 — API
cd api && source .venv/bin/activate  # create venv + pip install -r requirements.txt first
PYTHONPATH=. uvicorn app.main:app --reload --port 8000

# Terminal 2 — UI
cd web && npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open http://localhost:3000

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health + whether LLM key is configured |
| POST | `/predict` | Multipart image upload → quality + structured diagnosis report |
| POST | `/chat` | JSON `{ "message", "session_context" }` → agriculture-focused reply |

## Demo checklist

See [docs/phase2-demo-checklist.md](docs/phase2-demo-checklist.md).

## Disclaimer

Treatment suggestions are for general guidance only and must be validated with local agricultural experts before critical action.
