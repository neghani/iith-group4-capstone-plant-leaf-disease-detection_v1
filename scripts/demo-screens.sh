#!/usr/bin/env bash
# demo-screens.sh — set up the demo screen: container logs on one side, the app on the other.
#
#   ./scripts/demo-screens.sh             start the stack (if needed) + tile 2 log windows (right side)
#   ./scripts/demo-screens.sh --rebuild   rebuild images first, then tile
#   ./scripts/demo-screens.sh --logs-only tile 3 log columns across the whole screen, no browser
#
# Terminals (Terminal.app):
#   split mode : right ~60% of the screen -> api-service logs (top), inference-service logs (bottom);
#                browser with the CropCare demo opens on the left half.
#   logs-only  : three equal columns: api-service | inference-service | ui-service.
#
# macOS will ask for Terminal automation permission the first time — allow it.

set -euo pipefail
cd "$(dirname "$0")/.."

REBUILD=0
LOGS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --rebuild)   REBUILD=1 ;;
    --logs-only) LOGS_ONLY=1 ;;
    -h|--help)
      sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

echo "==> Starting the stack (docker compose up -d)..."
if [ "$REBUILD" = 1 ]; then
  docker compose up -d --build
else
  docker compose up -d
fi

echo "==> Waiting for api-service :8000/health ..."
ok=0
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then ok=1; break; fi
  sleep 2
done
if [ "$ok" = 1 ]; then
  echo "==> API is up. Probing the inference pin:"
  curl -fsS "http://localhost:8000/health?probe=true" || true
  echo
else
  echo "!! API did not become healthy in 2 minutes — logs will still open so you can see why."
fi

PROJ="$(pwd)"

if [ "$LOGS_ONLY" = 0 ]; then
  echo "==> Opening the demo UI at http://localhost:3000 (put it on the LEFT half)..."
  open http://localhost:3000
fi

echo "==> Opening + tiling log terminals..."
MODE=$([ "$LOGS_ONLY" = 1 ] && echo logs || echo split) PROJ="$PROJ" osascript <<'EOF'
on run argv
  set mode to system attribute "MODE"
  set proj to system attribute "PROJ"

  tell application "Finder"
    set d to bounds of window of desktop
  end tell
  set sx to item 1 of d
  set sy to item 2 of d
  set sw to item 3 of d
  set sh to item 4 of d
  set w to sw - sx
  set h to (sh - sy) - 25 -- menu bar
  set pad to 8

  tell application "Terminal"
    activate

    if mode is "split" then
      -- right 60% of the screen, stacked: api on top, inference below
      set lx to sx + (w * 2 div 5) + pad
      set rx to sw - pad
      set midY to sy + (h * 3 div 5)
      do script "cd " & quoted form of proj & " && docker compose logs -f --tail=60 api-service"
      set bounds of front window to {lx, sy + pad, rx, midY}
      set custom title of tab 1 of front window to "API · gate + agent + LLM"

      do script "cd " & quoted form of proj & " && docker compose logs -f --tail=60 inference-service"
      set bounds of front window to {lx, midY + pad, rx, sy + h - pad}
      set custom title of tab 1 of front window to "INFERENCE · classifier"
    else
      -- three equal columns across the whole screen
      set c1 to sx + pad
      set c2 to sx + (w div 3)
      set c3 to sx + (w * 2 div 3)
      set c4 to sw - pad

      do script "cd " & quoted form of proj & " && docker compose logs -f --tail=60 api-service"
      set bounds of front window to {c1, sy + pad, c2, sy + h - pad}
      set custom title of tab 1 of front window to "API · gate + agent + LLM"

      do script "cd " & quoted form of proj & " && docker compose logs -f --tail=60 inference-service"
      set bounds of front window to {c2, sy + pad, c3, sy + h - pad}
      set custom title of tab 1 of front window to "INFERENCE · classifier"

      do script "cd " & quoted form of proj & " && docker compose logs -f --tail=60 ui-service"
      set bounds of front window to {c3, sy + pad, c4, sy + h - pad}
      set custom title of tab 1 of front window to "UI"
    end if

    activate
  end tell
end run
EOF

echo
echo "Demo layout ready:"
if [ "$LOGS_ONLY" = 1 ]; then
  echo "  [ api logs | inference logs | ui logs ]  <- three columns, full screen"
else
  echo "  [ browser demo :3000 | api-service logs (top) / inference-service logs (bottom) ]"
fi
echo "Every upload will now narrate itself: QUALITY -> AGENT (LangGraph) -> INFERENCE -> LLM -> SAFETY."
