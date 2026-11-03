#!/usr/bin/env bash
# Local single-factor sweep helper (optional). Preferred path: Colab notebook section 11
# (RUN_V12=True) which retrains stage 1 from ImageNet into v1.2/stage1, then sweeps from that.
#
# This script continues from a LOCAL stage-1 folder (default: model/outputs/v1.2/stage1).
# It does NOT start from Hub v1.0/v1.1 and never writes the final-test guard.
#
#   bash model/sweep_mobilenet_stage2.sh            # all variants
#   bash model/sweep_mobilenet_stage2.sh ctrl field # subset
#   INIT_FROM=model/outputs/v1.2/stage1 bash model/sweep_mobilenet_stage2.sh
#
# Outputs: model/outputs/v1.2/sweep/<variant>/
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="${SWEEP_ROOT:-model/outputs/v1.2/sweep}"
INIT_FROM="${INIT_FROM:-model/outputs/v1.2/stage1}"
DATA_DIR="${DATA_DIR:-/Users/ganeshkumar/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage dataset/color}"
mkdir -p "$ROOT/artifacts" "$ROOT/logs"
if [ ! -f "$INIT_FROM/model.safetensors" ]; then
  echo "Missing $INIT_FROM/model.safetensors. Train stage 1 first (notebook section 11b) or set INIT_FROM=..." >&2
  exit 1
fi
[ -f "$ROOT/artifacts/pv_phash_cache.json" ] || {
  [ -f model/outputs/pv_phash_cache.json ] && cp model/outputs/pv_phash_cache.json "$ROOT/artifacts/" || true
}

export PYTHONPATH=.
export CODE_COMMIT="$(git rev-parse HEAD)"

VARIANTS=(
  "ctrl|"
  "warm|--warmup_steps 0.05"
  "robust|--aug_policy robust"
  "field|--aug_policy field"
  "frac70|--plantdoc_fraction 0.7"
)

want=("$@")
for spec in "${VARIANTS[@]}"; do
  name="${spec%%|*}"
  extra="${spec#*|}"
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " =~ " ${name} " ]]; then
    continue
  fi
  out="$ROOT/$name"
  echo "=== $name ($extra) -> $out  init_from=$INIT_FROM"
  if [ -f "$out/model.safetensors" ] && [ -f "$out/eval_results.json" ]; then
    echo "already finished, skipping (delete $out to rerun)"
    continue
  fi
  # shellcheck disable=SC2086
  python3 -m model.train \
    --init_from "$INIT_FROM" \
    --data_dir "$DATA_DIR" \
    --extra_data plantdoc --plantdoc_fraction 0.5 --epoch_samples 4000 \
    --artifacts_dir "$ROOT/artifacts" \
    --output_dir "$out" \
    --do_train --do_eval \
    --num_train_epochs 8 \
    --per_device_train_batch_size 64 --per_device_eval_batch_size 128 \
    --learning_rate 1e-4 --weight_decay 0.01 --label_smoothing_factor 0.1 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 4 \
    --eval_strategy epoch --save_strategy epoch --save_total_limit 2 \
    --load_best_model_at_end --metric_for_best_model plantdoc_macro_f1 --greater_is_better True \
    --logging_steps 50 --report_to none --seed 42 \
    $extra 2>&1 | tee "$ROOT/logs/$name.log"
done
echo "sweep done; compare with: python -m model.compare_stage2_sweep --root $ROOT"
