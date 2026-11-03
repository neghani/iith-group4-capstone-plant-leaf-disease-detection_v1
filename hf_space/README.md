---
title: Plant Leaf Disease Inference
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# Plant leaf disease inference server

Serves the Hub model pinned in `release.json` (copied from `shared/release.json`).
The live pin is MasterDataset ResNet-18 `@v1.0.1`
(`neghani/plant-leaf-masterdataset-resnet18`). PlantVillage ResNet-18 `@v1.1` remains
the rollback candidate. Hugging Face serverless inference does not host this fine-tune
("Model not supported by provider hf-inference"), so the API calls this Space or a dedicated
Inference Endpoint through `HF_ENDPOINT_URL`.

Contract (same as an image-classification Inference Endpoint):

| Method | Path | Body | Response |
|---|---|---|---|
| `GET` | `/health` | — | `{status, model, revision}` |
| `POST` | `/` | raw image bytes | `[{label, score}, ...]` (`?top_k=`) |

After changing `shared/release.json`, copy it here and rebuild the Space / Compose inference image.
