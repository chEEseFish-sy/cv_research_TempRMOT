#!/usr/bin/env bash
set -euo pipefail

source /home/curtis/miniforge3/etc/profile.d/conda.sh
conda activate temprmot
cd /home/curtis/projects/TempRMOT

snapshot_dir=.planning/2026-07-31-refer-kitti-v1-official-reproduction/phase2_runtime_snapshot
output_root=/home/curtis/experiments/TempRMOT/rk_official
mkdir -p "$output_root"

python tools/run_refer_kitti_v1.py \
  --checkpoint /home/curtis/checkpoints/TempRMOT/checkpoint_rk.pth \
  --data-root /home/curtis/datasets/refer-kitti \
  --seqmap /home/curtis/datasets/refer-kitti/seqmap.txt \
  --text-encoder-path /home/curtis/checkpoints/roberta-base \
  --output-root "$output_root" \
  --snapshot-dir "$snapshot_dir" \
  --gpu-id 0 \
  --object-threshold 0.6 \
  --referring-threshold 0.4 \
  --area-threshold 100 \
  "$@" 2>&1 | tee -a "$output_root/inference_v1.log"
