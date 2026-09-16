"""Real-image, zero-residual equivalence check. Does not change original weights."""
import argparse
import json
from pathlib import Path

import torch

from inference import ListImgDataset
from tools.run_ttt_pilot import config, load


def main(output):
    args = config(10, True)
    model, _ = load(args, Path('/home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth'))
    model.eval()
    memory = model.ttt_memory
    with torch.no_grad():
        memory.alpha.zero_()
    root = Path(args.rmot_path)
    expression = sorted((root / 'expression/0002').glob('*.json'))[0]
    sentence = [json.loads(expression.read_text())['sentence']]
    files = sorted((root / 'KITTI/training/image_02/0002').glob('*.png'))[:10]
    dataset = ListImgDataset([str(p) for p in files])
    predictions = []
    for enabled in (False, True):
        model.ttt_memory = memory if enabled else None
        model.track_base.clear()
        tracks = None
        sequence = []
        for i in range(len(dataset)):
            img, original = dataset[i]
            if tracks is not None:
                tracks.remove('boxes')
                tracks.remove('labels')
            result = model.inference_single_image(img.cuda().float(), sentence,
                                                 original.shape[:2], tracks)
            tracks = result['track_instances']
            sequence.append({k: tracks.get(k).detach().cpu().clone()
                             for k in ('obj_idxes', 'scores', 'boxes', 'refers', 'query_pos')})
        predictions.append(sequence)
    maximum = 0.0
    for off, on in zip(*predictions):
        for key in off:
            torch.testing.assert_close(off[key], on[key], rtol=0, atol=0)
            if off[key].numel():
                maximum = max(maximum, float((off[key] - on[key]).abs().max()))
    assert memory.stats['reads'] > 0
    output.write_text(json.dumps(dict(frames=len(files), exact_equal=True,
                                     max_difference=maximum, stats=memory.stats), indent=2))
    print(output.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    torch.set_num_threads(4)
    main(parser.parse_args().output)
