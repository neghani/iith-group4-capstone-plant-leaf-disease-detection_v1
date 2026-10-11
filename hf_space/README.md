---
title: PlantVillage MobileNetV2 Inference
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# PlantVillage MobileNetV2 inference server

Serves [`neghani/plantvillage-mobilenetv2`](https://huggingface.co/neghani/plantvillage-mobilenetv2)
at the tag in `release.json` (copied from `shared/release.json`, the repo-wide live pin).
Hugging Face serverless inference does not host this
fine-tune ("Model not supported by provider hf-inference"), so the API calls this Space or a
dedicated Inference Endpoint through `HF_ENDPOINT_URL`.

Contract (same as an image-classification Inference Endpoint):

```bash
curl -X POST "https://<space-host>/?top_k=3" -H "Content-Type: image/jpeg" --data-binary @leaf.jpg
# [{"label": "Tomato___Early_blight", "score": 0.97}, ...]
```

## Deploy

```bash
hf repos create neghani/plantvillage-inference --repo-type space --space-sdk docker
hf upload neghani/plantvillage-inference hf_space . --repo-type space
```

Then set `HF_ENDPOINT_URL=https://neghani-plantvillage-inference.hf.space/` in the repo-root `.env`.

## Run locally

```bash
cd hf_space && pip install -r requirements.txt
uvicorn app:app --port 7860
# HF_ENDPOINT_URL=http://localhost:7860/
```
