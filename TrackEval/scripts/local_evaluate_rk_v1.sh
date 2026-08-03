#!/usr/bin/env bash
set -euo pipefail

source /home/curtis/miniforge3/etc/profile.d/conda.sh
conda activate temprmot
project_root=/home/curtis/projects/TempRMOT
cd "$project_root"

snapshot_dir="$project_root/.planning/2026-07-31-refer-kitti-v1-official-reproduction/phase2_runtime_snapshot"
data_root=/home/curtis/datasets/refer-kitti
results_root="${1:-/home/curtis/experiments/TempRMOT/rk_official/oracle_rmot_official}"
log_path="${2:-/home/curtis/experiments/TempRMOT/rk_official/trackeval_v1.log}"

python tools/freeze_refer_kitti_v1_runtime.py --verify --snapshot-dir "$snapshot_dir"
python tools/validate_refer_kitti_v1_results.py \
  --results-root "$results_root" \
  --seqmap "$data_root/seqmap.txt" \
  --images-root "$data_root/KITTI/training/image_02" \
  --report /home/curtis/experiments/TempRMOT/rk_official/phase2_input_integrity.json

cd "$project_root/TrackEval/scripts"
python run_mot_challenge.py \
  --METRICS HOTA \
  --SEQMAP_FILE "$data_root/seqmap.txt" \
  --SKIP_SPLIT_FOL True \
  --GT_FOLDER "$data_root/KITTI/training/image_02" \
  --IMAGES_FOLDER "$data_root/KITTI/training/image_02" \
  --TRACKERS_FOLDER "$results_root" \
  --GT_LOC_FORMAT '{gt_folder}/{video_id}/{expression_id}/gt.txt' \
  --TRACKERS_TO_EVAL "$results_root" \
  --USE_PARALLEL True \
  --NUM_PARALLEL_CORES 2 \
  --PLOT_CURVES False \
  2>&1 | tee "$log_path"

cd "$project_root"
python tools/check_refer_kitti_v1_oracle_metrics.py \
  --summary "$results_root/pedestrian_summary.txt" \
  --report /home/curtis/experiments/TempRMOT/rk_official/phase2_oracle_metrics.json
python tools/freeze_refer_kitti_v1_runtime.py --verify --snapshot-dir "$snapshot_dir"
