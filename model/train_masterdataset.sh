#!/usr/bin/env bash
# Train ResNet-18 on the pre-split MasterDataset and push to a new Hub repo.
# Does not edit shared/release.json. Pin only after model/pin_masterdataset.py passes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DEST="${DEST:-data/masterdataset}"
python3 model/download_masterdataset.py --dest "$DEST"
DATA_DIR="${DATA_DIR:-$DEST}"
OUT="${OUT:-model/outputs/masterdataset-resnet18}"
HUB="${HUB:-neghani/plant-leaf-masterdataset-resnet18}"
EPOCHS="${STAGE1_EPOCHS:-5}"
LR="${STAGE1_LR:-5e-4}"
BATCH="${STAGE1_BATCH:-64}"

GPU_FLAGS=()
while IFS= read -r line; do
  [ -n "$line" ] && GPU_FLAGS+=("$line")
done < <(python3 -c "from model.common import gpu_train_flags; print('\n'.join(gpu_train_flags()))")

echo "Data: $DATA_DIR"
echo "Output: $OUT"
echo "Hub: $HUB (live API pin is not changed)"
echo "GPU flags: ${GPU_FLAGS[*]-}"

python3 -m model.train \
  --dataset masterdataset \
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
  --greater_is_better True \
  --logging_steps 50 \
  --seed 42 \
  --report_to none \
  "${GPU_FLAGS[@]}" \
  ${PUSH:+--push_to_hub --hub_model_id "$HUB" --hub_strategy end}

python3 model/plantdoc_ood.py --data_dir "$DATA_DIR" --labels_from "$OUT" --out "$OUT/plantdoc_ood.json" || true
echo "Test metrics are in $OUT/test_results.json. Pin with model/pin_masterdataset.py only after you choose threshold and temperature on validation."
