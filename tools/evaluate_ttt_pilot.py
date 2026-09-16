"""Evaluate pilot predictions with vendored HOTA/CLEAR/Identity, no vendor edits."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'TrackEval'))
import trackeval


def evaluate(root, results_name='results_pilot'):
    root = root.resolve()
    results = root / results_name
    assert results.is_dir()
    # TrackEval's historical Identity implementation uses removed NumPy aliases.
    # Compatibility is local to this evaluator process, never edits vendored code.
    if 'int' not in np.__dict__:
        np.int = int
    evaluator = trackeval.Evaluator(dict(USE_PARALLEL=False, PRINT_CONFIG=False,
        PLOT_CURVES=False, BREAK_ON_ERROR=True, PRINT_ONLY_COMBINED=True))
    dataset = trackeval.datasets.MotChallenge2DBox(dict(
        GT_FOLDER=str(results), TRACKERS_FOLDER=str(results),
        TRACKERS_TO_EVAL=[str(results)], SKIP_SPLIT_FOL=True,
        IMAGES_FOLDER='/home/curtis/datasets/refer-kitti-v2/KITTI/training/image_02',
        GT_LOC_FORMAT='{gt_folder}/{video_id}/{expression_id}/gt.txt',
        SEQMAP_FILE=str(root / 'seqmap.txt'), PRINT_CONFIG=False))
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    result, messages = evaluator.evaluate([dataset], metrics)
    combined = result[dataset.get_name()][str(results)]['COMBINED_SEQ']['pedestrian']
    summary = {}
    for group in combined.values():
        for key, value in group.items():
            if key in ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'Frag'):
                summary[key] = float(np.asarray(value).mean())
    (root / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--results-name', default='results_pilot')
    opts = parser.parse_args()
    evaluate(opts.root, opts.results_name)
