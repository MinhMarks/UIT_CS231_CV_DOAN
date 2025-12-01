#!/bin/bash

# =============================================================================
# Training script for SALAD with Cross-Image Learning
# =============================================================================

# Option 1: Train with frozen base (only train CrossImageEncoder)
# Recommended for first experiments
python train_cross_image.py \
    --pretrained_path pretrainedWeight/Salad/last.ckpt \
    --freeze_base \
    --batch_size 20 \
    --img_per_place 4 \
    --epochs 4 \
    --lr 6e-5 \
    --log_dir ./logs/cross_image_frozen/

# Option 2: Fine-tune entire model (uncomment to use)
# python train_cross_image.py \
#     --pretrained_path pretrainedWeight/Salad/last.ckpt \
#     --batch_size 20 \
#     --img_per_place 4 \
#     --epochs 4 \
#     --lr 1e-5 \
#     --log_dir ./logs/cross_image_finetune/

# Option 3: Train from scratch without pretrained (uncomment to use)
# python train_cross_image.py \
#     --pretrained_path "" \
#     --batch_size 20 \
#     --img_per_place 4 \
#     --epochs 10 \
#     --lr 6e-5 \
#     --log_dir ./logs/cross_image_scratch/

# =============================================================================
# Evaluation script
# =============================================================================

# Evaluate on MSLS validation set
# python evaluate.py \
#     --checkpoint ./logs/cross_image_frozen/cross_image_salad/freeze_True_lr_6e-05/checkpoints/last.ckpt \
#     --val_sets msls_val \
#     --batch_size 32

# Evaluate on Pittsburgh
# python evaluate.py \
#     --checkpoint ./logs/cross_image_frozen/cross_image_salad/freeze_True_lr_6e-05/checkpoints/last.ckpt \
#     --val_sets pitts30k_val,pitts30k_test \
#     --batch_size 32
