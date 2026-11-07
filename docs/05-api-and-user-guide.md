# API and user guide

For graders, teammates, and demo operators. Interactive OpenAPI: `http://localhost:8000/docs`.

## Farmer UI (Next.js)

| Route | Purpose |
|-------|---------|
| `/check` | Upload leaf, crop hint, symptoms → quality then predict |
| `/results` | Diagnosis, top-3, actions, monitoring, chat |
| `/history` | This-tab history (`sessionStorage`) |
| `/guide` | Static growing tips |
| `/web-test` | Tutor smoke: web search → live pin (GT = search target) |
| `/local-test` | Tutor smoke: random `real_world_set` (nav gated by `localStorage["local-test"]="true"`) |

Language toggle: English / Telugu (diagnosis TE needs `LLM_API_KEY`).

### Typical farmer path

1. Open `/check`, pick crop (or Other), optional notes, upload photo.  
2. App may shrink the file in-browser (long side ≤ 2048).  
3. Quality check — fail → retake message, no model call.  
4. Predict → results: status, top-3, KB actions, LLM summary if key present.  
5. Chat with session context; off-topic declined.

Helpers on `/check`: **Show a retake**, **Show “I'm not sure”** for demos.

## HTTP API

Base URL (Compose): `http://localhost:8000`.

| Method | Path | Body | Result |
|--------|------|------|--------|
| GET | `/health?probe=true` | — | Pin, LLM configured, inference reachability + served model |
| POST | `/quality-check` | multipart `file` | `{ ok, message, brightness, blur_score, leaf_found, … }` |
| POST | `/predict` | multipart `file` + optional crop / symptoms / language | Report + `top_k` + `inference` + provenance |
| POST | `/chat` | JSON `message`, `session_context`, optional `history`, `language` | Reply; may decline |
| GET | `/web-test/classes` | — | Demo pool + pin meta |
| POST | `/web-test/run` | `{ n, use_gt_crop_hint?, class_id? }` | NDJSON stream of smoke rows |
| GET | `/local-test/meta` | — | real_world_set availability |
| POST | `/local-test/run` | `{ n, use_gt_crop_hint?, split? }` | NDJSON stream |

### Predict behaviour (worth stating)

- Re-runs quality gate (never trust client alone).  
- Crop strategy B → 256² JPEG → HF endpoint.  
- Below confidence threshold / OOD margin → not_sure path.  
- Crop hint mismatch → possible match + note.  
- Unsupported crop → no disease name.  
- LLM never receives the image.

### Smoke tables (Web / Local)

Shared scoring (`smoke_common`):

- **Yes** — top-1 = GT, score ≥ threshold, crop hint OK  
- **Abstain** — score &lt; threshold or model `not_sure`  
- **Crop mismatch** — GT crop hint disagrees with predicted crop (high conf)  
- **No** — confident wrong label  

## Privacy (demo)

No accounts. No server image store. History only in the browser tab.
