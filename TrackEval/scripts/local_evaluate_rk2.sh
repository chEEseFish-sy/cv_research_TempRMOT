#!/usr/bin/env bash
set -euo pipefail

source /home/curtis/miniforge3/etc/profile.d/conda.sh
conda activate temprmot
cd /home/curtis/projects/TempRMOT/TrackEval/scripts

project_root=/home/curtis/projects/TempRMOT
data_root=/home/curtis/datasets/refer-kitti-v2
results_root=/home/curtis/experiments/TempRMOT/rk2_official/results_checkpoint_rk2

python run_mot_challenge.py \
  --METRICS HOTA \
  --SEQMAP_FILE "$project_root/datasets/data_path/seqmap.txt" \
  --SKIP_SPLIT_FOL True \
  --GT_FOLDER "$data_root/KITTI/training/image_02" \
  --IMAGES_FOLDER "$data_root/KITTI/training/image_02" \
  --TRACKERS_FOLDER "$results_root" \
  --GT_LOC_FORMAT '{gt_folder}/{video_id}/{expression_id}/gt.txt' \
  --TRACKERS_TO_EVAL "$results_root" \
  --USE_PARALLEL True \
  --NUM_PARALLEL_CORES 2 \
  --PLOT_CURVES False \
  "$@" 2>&1 | tee /home/curtis/experiments/TempRMOT/rk2_official/trackeval.log
