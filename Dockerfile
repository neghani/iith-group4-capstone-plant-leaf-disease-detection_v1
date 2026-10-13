# All three services in one container (local `docker run` or a single Render Web Service).
#   docker build -t cropcare .
#   docker run --rm -p 3000:3000 --env-file .env cropcare
#
# Public port: $PORT (default 3000). Browser hits UI; /health /predict /docs go to the API.
# Inference stays on 127.0.0.1:7860, same as Compose's inference-service.

FROM node:20-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
# same-origin: browser calls /predict on this host; nginx proxies to the API.
ARG NEXT_PUBLIC_API_URL=same-origin
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

FROM python:3.11-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gettext-base \
    libgomp1 \
    libjpeg62-turbo \
    nginx \
    && rm -rf /var/lib/apt/lists/*

COPY --from=web /usr/local/bin/node /usr/local/bin/node

WORKDIR /app

COPY hf_space/requirements.txt /tmp/inference-requirements.txt
COPY api/requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir \
    -r /tmp/api-requirements.txt \
    -r /tmp/inference-requirements.txt \
    && rm /tmp/api-requirements.txt /tmp/inference-requirements.txt

COPY shared ./shared
COPY api/app ./app
COPY api/data ./data
COPY hf_space/app.py hf_space/release.json ./inference/
COPY deploy/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY deploy/start.sh /app/start.sh
RUN chmod +x /app/start.sh \
    && rm -f /etc/nginx/sites-enabled/default \
    && ln -sf /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default

COPY --from=web /web/.next/standalone /app/web
COPY --from=web /web/.next/static /app/web/.next/static
COPY --from=web /web/public /app/web/public

# Pull Hub weights at build so first Render boot does not wait on a Hub download.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from transformers import pipeline;\
import json;\
pin=json.load(open('/app/inference/release.json'));\
pipeline('image-classification', model=pin['hf_model_id'], revision=pin['hf_model_revision'], device='cpu');\
print('cached', pin['hf_model_id'], pin['hf_model_revision'])\
"

ENV PYTHONPATH=/app
ENV PORT=3000
EXPOSE 3000

CMD ["/app/start.sh"]
