#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=datasets/ADNI/MRI-PET
SPLIT_DIR=datasets/MRI_PET_split
OUT_DIR_ROOT=results/MRI_PET_mmfusion_sweeps
MODEL_DEPTHS=(200)
SPLITS=(all balanced)
EPOCHS=200
LR=1e-5

for depth in "${MODEL_DEPTHS[@]}"; do
  for split in "${SPLITS[@]}"; do
    SPLIT_JSON="$SPLIT_DIR/${split}_split.json"
    SAVE_PATH="${OUT_DIR_ROOT}/depth${depth}_${split}"
    python MRI_PET_mmfusion_per_epoch.py \
      --data-dir "$DATA_DIR" \
      --save-path "$SAVE_PATH" \
      --load-patient-ids "$SPLIT_JSON" \
      --model-depth "$depth" \
      --epochs "$EPOCHS" \
      --lr "$LR"
  done
done
