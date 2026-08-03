#!/usr/bin/env bash
set -euo pipefail

source /home/curtis/miniforge3/etc/profile.d/conda.sh
conda activate temprmot
cd /home/curtis/projects/TempRMOT

output_dir=/home/curtis/experiments/TempRMOT/rk2_official
mkdir -p "$output_dir"

python inference.py \
  --meta_arch temp_rmot \
  --dataset_file e2e_rmot \
  --with_box_refine \
  --lr_drop 100 \
  --lr 2e-4 \
  --lr_backbone 2e-5 \
  --batch_size 1 \
  --sample_mode random_interval \
  --sample_interval 1 \
  --sampler_steps 60 80 90 \
  --sampler_lengths 5 5 5 5 \
  --update_query_pos \
  --merger_dropout 0 \
  --dropout 0 \
  --random_drop 0.1 \
  --fp_ratio 0.3 \
  --query_interaction_layer QIM \
  --hist_len 4 \
  --rmot_path /home/curtis/datasets/refer-kitti-v2 \
  --text_encoder_path /home/curtis/checkpoints/roberta-base \
  --resume /home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth \
  --num_inference_workers 1 \
  --gpu_id 0 \
  --output_dir "$output_dir" \
  "$@" 2>&1 | tee "$output_dir/inference.log"
