#!/usr/bin/env bash
# Stage 1 only: PlantVillage, 5 epochs, tag v1.0 on neghani/plantvillage-resnet18.
# Does not edit shared/release.json. Stage 2: model/train_resnet18_stage2.sh or the Colab notebook.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATA_DIR="${DATA_DIR:-$(python3 -c "import kagglehub; print(kagglehub.dataset_download('abdallahalidev/plantvillage-dataset'))")}"
OUT="${OUT:-model/outputs/plantvillage-resnet18}"
HUB="${HUB:-neghani/plantvillage-resnet18}"
EPOCHS="${STAGE1_EPOCHS:-5}"
LR="${STAGE1_LR:-5e-4}"
BATCH="${STAGE1_BATCH:-64}"

GPU_FLAGS=()
while IFS= read -r line; do
  [ -n "$line" ] && GPU_FLAGS+=("$line")
done < <(python3 -c "from model.common import gpu_train_flags; print('\n'.join(gpu_train_flags()))")

echo "Data: $DATA_DIR"
echo "Output: $OUT"
echo "Hub: $HUB tag v1.0 (live API pin is not changed)"
echo "GPU flags: ${GPU_FLAGS[*]-}"

python3 -m model.train \
  --model_name_or_path microsoft/resnet-18 \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT" \
  --do_train --do_eval \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH" \
  --per_device_eval_batch_size 128 \
  --learning_rate "$LR" \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine \
  --warmup_steps 500 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model macro_f1 \
  --logging_steps 50 \
  --report_to none \
  --seed 42 \
  --push_to_hub \
  --hub_model_id "$HUB" \
  "${GPU_FLAGS[@]}" \
  "$@"

python3 -m model.evaluate --model "$OUT" --hub_model_id "$HUB" --push
python3 -m model.publish --model "$OUT" --hub_model_id "$HUB" --tag v1.0
echo "Stage 1 tagged v1.0. Next: bash model/train_resnet18_stage2.sh"
echo "Docker keeps serving the current shared/release.json pin."
