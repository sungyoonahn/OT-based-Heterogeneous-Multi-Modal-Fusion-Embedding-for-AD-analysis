#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=datasets/ADNI/MRI-PET
OUT_ROOT=results/3DResNet_pretraining
WEIGHT_ROOT=weights
SPLIT_DIR=datasets/MRI_PET_split
MODEL_DEPTHS=(101 152 200)
SPLITS=(all balanced)
MODALITIES=(pet)

for modality in "${MODALITIES[@]}"; do
  for depth in "${MODEL_DEPTHS[@]}"; do
    for split in "${SPLITS[@]}"; do
      if [[ "$split" == "balanced" ]]; then
        SPLIT_JSON="$SPLIT_DIR/balanced_split.json"
      else
        SPLIT_JSON="$SPLIT_DIR/all_split.json"
      fi

      SAVE_PATH="$OUT_ROOT/${modality}_depth${depth}_${split}"
      python 3D_resnet.py \
        --data-dir "$DATA_DIR" \
        --save-path "$SAVE_PATH" \
        --classes AD CN \
        --modality "$modality" \
        --model-depth "$depth" \
        --load-patient-ids "$SPLIT_JSON"

      cp "$SAVE_PATH/best_model.pth" "$WEIGHT_ROOT/${modality}_resnet${depth}_${split}_backbone.pth"
    done
  done
done
