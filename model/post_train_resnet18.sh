#!/usr/bin/env bash
# Score and publish an existing ResNet output directory.
# TAG=v1.2 (or v1.1) runs calibrate then publish without --skip_gate.
# Never writes shared/release.json or hf_space/release.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT="${OUT:-model/outputs/plantvillage-resnet18}"
HUB="${HUB:-neghani/plantvillage-resnet18}"
TAG="${TAG:-v1.0}"

if [ ! -d "$OUT" ]; then
  echo "Missing $OUT. Train first." >&2
  exit 1
fi

python3 -m model.evaluate --model "$OUT" --hub_model_id "$HUB" --push
case "$TAG" in
  v1.1|v1.2|v1.[3-9]*)
    python3 -m model.calibrate --model "$OUT"
    python3 -m model.publish \
      --model "$OUT" \
      --hub_model_id "$HUB" \
      --tag "$TAG" \
      --diagnostics model/outputs/diagnostics_v1_0.json
    ;;
  *)
    python3 -m model.publish --model "$OUT" --hub_model_id "$HUB" --tag "$TAG"
    ;;
esac
echo "Published ${HUB} tag ${TAG}. Live pin was not changed."
echo "Fitted temperature, if any, is ${OUT}/calibration.json only."
