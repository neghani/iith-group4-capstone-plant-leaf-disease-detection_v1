> **Obsolete planning draft.** Do not use this file to run the app. Two-service Compose, threshold 0.6, and `API_URL` are out of date. The current runbook is [README.md](README.md) (three services, live pin `shared/release.json` / `v1.1`, threshold 0.5).

# AI-Powered Plant Disease Detection and Farmer Assistance

A farmer uploads a photo of a crop. The system identifies the disease with a MobileNetV2 model, then uses an LLM agent to explain the disease and suggest treatment in plain language. The farmer can ask follow-up questions in a chat. Hindi support is planned.

Capstone Project 7, Group 4.

> Status: planning and early implementation. Items marked **(planned)** are not built yet.

## Tech stack

| Layer | Choice | Role |
|---|---|---|
| API and core logic | [FastAPI](https://fastapi.tiangolo.com/) | Endpoints for image upload, prediction and chat. Generates interactive API docs at `/docs`. |
| Agent flow | [LangGraph](https://langchain-ai.github.io/langgraph/) | Pipeline of connected steps: image check, prediction, knowledge lookup, LLM explanation, optional translation. |
| Data validation | [Pydantic](https://docs.pydantic.dev/) | Typed request, response and agent state models. |
| User interface | [Next.js](https://nextjs.org/) | Image upload and chat window, built from the Figma design. |
| Disease model | MobileNetV2 (hosted on [Hugging Face](https://huggingface.co/)) | Trained on PlantVillage. Called by the API service for predictions. |
| LLM | Grok (free tier) first, other providers compared | Used behind a provider-independent interface. |
| Translation (planned) | Multilingual BERT-based summariser, IndicTrans, LLM expansion | Hindi explanations. English uses the same pipeline with a language parameter. |
| Containers | Docker Compose | Two services: `api-service` and `ui-service`. |
| Training | Google Colab | Model training, evaluation and results notebook. |

## Architecture

```
┌──────────────┐  HTTP   ┌─────────────────────────────────────┐
│  ui-service  │ ──────► │            api-service              │
│  Next.js     │         │  FastAPI + LangGraph agent          │
└──────────────┘         │                                     │
                         │  1. image quality check             │
                         │  2. prediction ───────────────┐     │
                         │  3. knowledge lookup          │     │
                         │  4. LLM explanation           │     │
                         │  5. translation (Hindi)       │     │
                         └───────────────────────────────┼─────┘
                                                         ▼
                                             MobileNetV2 on Hugging Face
```

The two services talk over the internal Docker network. The UI never calls the model or the LLM directly.

### Request flow

1. The farmer uploads a crop image in the UI.
2. The API checks image quality and asks for a better photo if the image is too blurry or dark.
3. The model returns the likely disease and a confidence score. A low score produces a "not sure" response.
4. The agent looks up the disease in the agricultural knowledge file.
5. The LLM writes the explanation and treatment advice from that entry. It answers agriculture questions only and declines other topics briefly and politely.
6. If Hindi is selected, the translation pipeline is applied **(planned)**.
7. The farmer asks follow-up questions in the chat.

## Repository layout (proposed)

```
.
├── api/
│   ├── app/
│   │   ├── main.py            # FastAPI app and routes
│   │   ├── schemas.py         # Pydantic models
│   │   ├── agent/             # LangGraph graph, nodes and prompts
│   │   ├── vision/            # model loading, preprocessing, quality check
│   │   ├── llm/               # provider interface and implementations
│   │   └── translation/       # glossary and Hindi pipeline (planned)
│   ├── data/
│   │   └── knowledge/         # agricultural knowledge file, one entry per disease class
│   ├── requirements.txt
│   └── Dockerfile
├── web/                       # Next.js UI
├── notebooks/                 # Colab notebooks: training, evaluation, real-world tests
├── docs/                      # architecture, user flow, API notes
├── docker-compose.yml
├── .env.example
└── README.md
```

## Getting started

### Prerequisites

- Docker and Docker Compose
- Python 3.11 or later (only for running without Docker)
- An API key for the chosen LLM provider
- Access to the model on Hugging Face

### Configuration

Copy `.env.example` to `.env` and fill in the values. Never commit `.env`.

```env
# LLM
LLM_PROVIDER=grok
LLM_MODEL=<model-name>
LLM_API_KEY=<your-key>

# Disease model
HF_MODEL_ID=<huggingface-user>/<model-name>
HF_TOKEN=<token-if-the-model-is-private>
CONFIDENCE_THRESHOLD=0.6

# App
DEFAULT_LANGUAGE=en
API_URL=http://api-service:8000
```

`CONFIDENCE_THRESHOLD` is a starting value. We tune it using the test results.

### Run with Docker

```bash
docker compose up --build
```

- UI: http://localhost:3000
- API: http://localhost:8000
- API docs: http://localhost:8000/docs

### Run without Docker

```bash
# API
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# UI (in a second terminal)
cd web
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

### docker-compose.yml (reference)

```yaml
services:
  api-service:
    build: ./api
    env_file: .env
    ports:
      - "8000:8000"

  ui-service:
    build: ./web
    ports:
      - "3000:3000"
    depends_on:
      - api-service
```

## API (proposed contract)

The final contract is generated by FastAPI at `/docs` once the endpoints exist.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service health check. |
| `POST` | `/predict` | Upload an image, receive disease, confidence and image quality result. |
| `POST` | `/chat` | Send a message with the current diagnosis context, receive the agent reply. |

Example `/predict` response:

```json
{
  "disease": "Tomato___Late_blight",
  "confidence": 0.91,
  "quality": { "ok": true, "message": null },
  "low_confidence": false
}
```

## Model and data

- **Training data:** [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data) (Kaggle).
- **Model:** MobileNetV2, trained in Colab and hosted on Hugging Face. If it does not perform well enough, a ResNet (CNN) alternative will be tried.
- **Real-world check:** images collected from the internet, covering the same crops and diseases.
- **Metrics:** accuracy, per-class precision, recall and F1, and a confusion matrix.
- **Target:** at least 50% of real-world images predicted correctly in the demo.
- **Knowledge file:** disease descriptions, symptoms, treatments and prevention, one entry per class the model predicts.

## Image quality handling

The API checks images before prediction (for example blur and brightness) and asks for a better photo when needed. During training we use augmentation with rotation, flips, blur and brightness changes to cope with tilted, inverted and low-quality photos.

## Development phases

| Phase | Focus |
|---|---|
| 1 | Disease detection model: train, evaluate, test on real images, deploy to Hugging Face |
| 2 | LLM agent flow and API with a placeholder prediction |
| 3 | Final interface and integration testing |
| 4 | Documentation and presentation |
| 5 | Replace the placeholder prediction with our own model |
| 6 | Hindi support, and a second regional language if time allows |

## Contributing

1. Create a branch from `main`: `git checkout -b feature/<short-name>`.
2. Keep secrets out of the repository. Use `.env`.
3. Open a pull request and ask a teammate to review it before merging.

## Disclaimer

Treatment suggestions are for general guidance only and may be wrong. Farmers should confirm important decisions with a local agricultural expert.

## References

- [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data)
- [FastAPI](https://fastapi.tiangolo.com/), [LangGraph](https://langchain-ai.github.io/langgraph/), [Next.js](https://nextjs.org/)
- IndicTrans translation models (optional, for Hindi)
