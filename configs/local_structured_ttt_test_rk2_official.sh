#!/usr/bin/env bash
set -euo pipefail

source /home/curtis/miniforge3/etc/profile.d/conda.sh
conda activate temprmot
cd /home/curtis/projects/TempRMOT

output_dir=/home/curtis/experiments/TempRMOT/structured_ttt/official_v2_full_step25
checkpoint=/home/curtis/experiments/TempRMOT/structured_ttt/20260910_adaptation_train24_80x3_lr3e4/ttt_step_000025.pth
seqmap=/home/curtis/projects/TempRMOT/datasets/data_path/seqmap.txt
result_tag=structured_ttt_step25

mkdir -p "$output_dir"

PYTHONPATH=. python -u tools/run_ttt_pilot.py eval \
  --output "$output_dir" \
  --checkpoint "$checkpoint" \
  --seqmap "$seqmap" \
  --result-tag "$result_tag" \
  --resume-eval \
  2>&1 | tee -a "$output_dir/full_run.log"

PYTHONPATH=. python -u tools/evaluate_ttt_pilot.py "$output_dir" \
  --results-name "results_$result_tag" \
  2>&1 | tee -a "$output_dir/trackeval.log"
