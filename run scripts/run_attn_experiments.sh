#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_attn_experiments.sh [both|with_pretrain|no_pretrain]

Controls whether MRI/PET backbones are pretrained before running the OT model:
  both (default)      run every configuration twice (first without, then with pretraining)
  with_pretrain       only run configs that load pretrained MRI/PET backbones
  no_pretrain         only run configs that start from random initialization
EOF
}

PRETRAIN_CHOICE="${1:-both}"
RUN_NO=false
RUN_WITH=false
case "$PRETRAIN_CHOICE" in
  both) RUN_NO=true; RUN_WITH=true ;;
  with_pretrain) RUN_WITH=true ;;
  no_pretrain) RUN_NO=true ;;
  -h|--help) usage; exit 0 ;;
  *) usage; echo "Invalid pretrain choice: $PRETRAIN_CHOICE" >&2; exit 1 ;;
esac

DATA_DIR=datasets/ADNI/MRI-PET
SPLIT_DIR=datasets/MRI_PET_split
MODEL_DEPTHS=(101 152 200)
DROPOUTS=(0.3 0.2 0.1 none)

run_sweep_for_mode() {
  local mode="$1"
  for depth in "${MODEL_DEPTHS[@]}"; do
    for drop in "${DROPOUTS[@]}"; do
      DROP_FLAG=("--projection-dropout" "$drop")
      if [[ "$drop" == "none" ]]; then
        DROP_FLAG=()
      fi

      for split in all balanced; do
        SPLIT_JSON="$SPLIT_DIR/${split}_split.json"

        MRI_PRETRAIN_ARG=()
        PET_PRETRAIN_ARG=()
        if [[ "$mode" == "with_pretrain" ]]; then
          MRI_PRETRAIN_ARG=("--mri-pretrained" "weights/mri_resnet${depth}_${split}_backbone.pth")
          PET_PRETRAIN_ARG=("--pet-pretrained" "weights/pet_resnet${depth}_${split}_backbone.pth")
        fi

        OUT_DIR="results/MRI_PET_OT_attention/mdepth${depth}_drop${drop}_${split}_${mode}"
        python MRI_PET_OT_OT_per_epoch_attn.py \
          --data-dir "$DATA_DIR" \
          --save-path "$OUT_DIR" \
          --load-patient-ids "$SPLIT_JSON" \
          --model-depth "$depth" \
          "${MRI_PRETRAIN_ARG[@]}" \
          "${PET_PRETRAIN_ARG[@]}" \
          "${DROP_FLAG[@]}"
      done
    done
  done
}

if $RUN_NO; then
  run_sweep_for_mode no_pretrain
fi

if $RUN_WITH; then
  run_sweep_for_mode with_pretrain
fi
