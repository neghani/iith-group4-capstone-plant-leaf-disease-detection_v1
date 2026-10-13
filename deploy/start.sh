#!/bin/sh
# Start inference + API + UI on loopback, then nginx on $PORT (Render / docker run).
set -eu

PORT="${PORT:-3000}"
export PYTHONPATH=/app
if [ -z "${HF_ENDPOINT_URL:-}" ]; then
  export HF_ENDPOINT_URL="http://127.0.0.1:7860/"
fi

echo "cropcare: inference on 127.0.0.1:7860"
cd /app/inference
uvicorn app:app --host 127.0.0.1 --port 7860 &
INF_PID=$!

i=0
until curl -sf "http://127.0.0.1:7860/health" >/dev/null; do
  i=$((i + 1))
  if [ "$i" -gt 60 ]; then
    echo "cropcare: inference did not become healthy"
    exit 1
  fi
  sleep 3
done

echo "cropcare: API on 127.0.0.1:8000"
cd /app
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

i=0
until curl -sf "http://127.0.0.1:8000/health" >/dev/null; do
  i=$((i + 1))
  if [ "$i" -gt 30 ]; then
    echo "cropcare: API did not become healthy"
    exit 1
  fi
  sleep 1
done

echo "cropcare: UI on 127.0.0.1:3000"
cd /app/web
HOSTNAME=127.0.0.1 PORT=3000 node server.js &
UI_PID=$!

envsubst '${PORT}' < /etc/nginx/templates/default.conf.template \
  > /etc/nginx/sites-available/default

echo "cropcare: nginx on 0.0.0.0:${PORT} (UI + API)"
nginx -g 'daemon off;' &
NGINX_PID=$!

term() {
  kill "$NGINX_PID" "$UI_PID" "$API_PID" "$INF_PID" 2>/dev/null || true
  wait || true
}
trap term INT TERM

# Exit if any child dies so Render restarts the whole box.
while true; do
  for pid in "$INF_PID" "$API_PID" "$UI_PID" "$NGINX_PID"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "cropcare: process $pid exited"
      term
      exit 1
    fi
  done
  sleep 2
done
