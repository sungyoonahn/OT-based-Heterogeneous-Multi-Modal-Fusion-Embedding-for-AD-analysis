#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=datasets/ADNI/MRI-PET
SPLIT_DIR=datasets/MRI_PET_split

# 1. MRI-PET OT balanced-fixed
python MRI_PET_OT_OT_per_epoch_attn.py \
  --data-dir "$DATA_DIR" \
  --save-path results/MRI_PET_OT_OT_per_epoch_attn/balanced \
  --load-patient-ids "$SPLIT_DIR/balanced_split.json"

# 2. MRI-PET OT all-fixed
python MRI_PET_OT_OT_per_epoch_attn.py \
  --data-dir "$DATA_DIR" \
  --save-path results/MRI_PET_OT_OT_per_epoch_attn/all \
  --load-patient-ids "$SPLIT_DIR/all_split.json"

# 3. 3D ResNet MRI balanced-fixed
python 3D_resnet.py \
  --data-dir "$DATA_DIR" \
  --save-path results/3DResNet/MRI_balanced \
  --classes AD CN \
  --modality mri \
  --load-patient-ids "$SPLIT_DIR/balanced_split.json"

# 4. 3D ResNet MRI all-fixed
python 3D_resnet.py \
  --data-dir "$DATA_DIR" \
  --save-path results/3DResNet/MRI_all \
  --classes AD CN \
  --modality mri \
  --load-patient-ids "$SPLIT_DIR/all_split.json"

# 5. 3D ResNet PET balanced-fixed
python 3D_resnet.py \
  --data-dir "$DATA_DIR" \
  --save-path results/3DResNet/PET_balanced \
  --classes AD CN \
  --modality pet \
  --load-patient-ids "$SPLIT_DIR/balanced_split.json"

# 6. 3D ResNet PET all-fixed
python 3D_resnet.py \
  --data-dir "$DATA_DIR" \
  --save-path results/3DResNet/PET_all \
  --classes AD CN \
  --modality pet \
  --load-patient-ids "$SPLIT_DIR/all_split.json"
