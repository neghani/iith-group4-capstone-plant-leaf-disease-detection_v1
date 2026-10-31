#!/usr/bin/env bash
# Stage 2: PlantVillage + PlantDoc mix, starting from ResNet v1.0.
# Tags v1.1 only if publish.py's gate passes. Does not edit the live release pin.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATA_DIR="${DATA_DIR:-$(python3 -c "import kagglehub; print(kagglehub.dataset_download('abdallahalidev/plantvillage-dataset'))")}"
HUB="${HUB:-neghani/plantvillage-resnet18}"
PD_FRACTION="${PD_FRACTION:-0.5}"
EPOCH_SAMPLES="${EPOCH_SAMPLES:-4000}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-8}"
STAGE2_BATCH="${STAGE2_BATCH:-64}"
STAGE2_LRS="${STAGE2_LRS:-1e-4,5e-5}"
STAGE2_ROOT="${STAGE2_ROOT:-model/outputs/stage2-resnet18}"

GPU_FLAGS=()
while IFS= read -r line; do
  [ -n "$line" ] && GPU_FLAGS+=("$line")
done < <(python3 -c "from model.common import gpu_train_flags; print('\n'.join(gpu_train_flags()))")

echo "Preparing Cropped-PlantDoc..."
python3 -m model.prepare_plantdoc

# Gate baseline: PlantDoc val macro-F1 of the published v1.0, not of the stage-2 run.
echo "Diagnostics on ${HUB}@v1.0 (writes model/outputs/diagnostics_v1_0.json)"
python3 -m model.diagnostics --model "$HUB" --revision v1.0 --out model/outputs

best_dir=""
best_f1="-1"
IFS=',' read -r -a LRS <<< "$STAGE2_LRS"
for lr in "${LRS[@]}"; do
  lr="$(printf '%s' "$lr" | tr -d ' ')"
  out="${STAGE2_ROOT}/lr${lr}"
  echo "=== Stage 2 lr=${lr} -> ${out} ==="
  python3 -m model.train \
    --init_from "$HUB" \
    --model_revision v1.0 \
    --data_dir "$DATA_DIR" \
    --extra_data plantdoc \
    --plantdoc_fraction "$PD_FRACTION" \
    --epoch_samples "$EPOCH_SAMPLES" \
    --artifacts_dir model/outputs \
    --output_dir "$out" \
    --resume_from_checkpoint auto \
    --do_train --do_eval \
    --num_train_epochs "$STAGE2_EPOCHS" \
    --per_device_train_batch_size "$STAGE2_BATCH" \
    --per_device_eval_batch_size 128 \
    --learning_rate "$lr" \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1 \
    --lr_scheduler_type cosine \
    "${GPU_FLAGS[@]}" \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --metric_for_best_model plantdoc_macro_f1 \
    --greater_is_better True \
    --logging_steps 50 \
    --report_to none \
    --seed 42
  f1="$(python3 -c "import json; print(json.load(open('${out}/eval_results.json')).get('eval_plantdoc_macro_f1', -1))")"
  echo "PlantDoc val macro-F1 @ ${lr}: ${f1}"
  better="$(python3 -c "print(1 if float('${f1}') > float('${best_f1}') else 0)")"
  if [ "$better" = "1" ]; then
    best_f1="$f1"
    best_dir="$out"
  fi
done

if [ -z "$best_dir" ]; then
  echo "No stage-2 run produced a score." >&2
  exit 1
fi
echo "Best stage-2 dir: $best_dir (PlantDoc val macro-F1 ${best_f1})"

python3 -m model.calibrate --model "$best_dir"
python3 -m model.evaluate --model "$best_dir" --hub_model_id "$HUB" --data_dir "$DATA_DIR"
python3 -m model.publish \
  --model "$best_dir" \
  --hub_model_id "$HUB" \
  --tag v1.1 \
  --diagnostics model/outputs/diagnostics_v1_0.json
echo "Tagged v1.1 on ${HUB}. calibration.json is only in ${best_dir}."
echo "Live pin was not changed. Run final_test once, check the strawberry photo, then promote Docker."
