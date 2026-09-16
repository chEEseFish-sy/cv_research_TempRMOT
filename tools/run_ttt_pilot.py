"""Bounded TTT-only training and full-sequence diagnostic evaluation.

Run from repository root with PYTHONPATH=.; original tracking rules are unchanged.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from main import get_args_parser
from models import build_model
from datasets import build_dataset
from datasets.data_prefetcher import data_dict_to_cuda


ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = ('0002', '0006', '0010')


def load_seqmap(path, expression_root):
    """Resolve a frozen seqmap into ordered expression JSON files."""
    selected = []
    seen = set()
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        entry = raw_line.strip()
        if not entry:
            continue
        video, separator, expression = entry.partition('+')
        if not separator or not video or not expression:
            raise ValueError(f'Invalid seqmap row {line_number}: {raw_line!r}')
        key = (video, expression)
        if key in seen:
            raise ValueError(f'Duplicate seqmap row {line_number}: {entry}')
        expression_path = expression_root / video / f'{expression}.json'
        if not expression_path.is_file():
            raise FileNotFoundError(expression_path)
        seen.add(key)
        selected.append(expression_path)
    if not selected:
        raise ValueError(f'No expressions in seqmap: {path}')
    return selected


def is_evaluation_complete(output, result_tag, expression_path):
    result = output / f'results_{result_tag}' / expression_path.parent.name / expression_path.stem
    # predict.txt may legitimately be empty, so completion requires both files.
    return (result / 'predict.txt').is_file() and (result / 'gt.txt').is_file()


def config(length, enabled):
    args = get_args_parser().parse_args([
        '--meta_arch', 'temp_rmot', '--dataset_file', 'e2e_rmot',
        '--with_box_refine', '--batch_size', '1', '--sample_mode', 'fixed_interval',
        '--sample_interval', '1', '--sampler_lengths', str(length),
        '--update_query_pos', '--merger_dropout', '0', '--dropout', '0',
        '--random_drop', '0.1', '--fp_ratio', '0.3', '--query_interaction_layer', 'QIM',
        '--hist_len', '4', '--rmot_path', '/home/curtis/datasets/refer-kitti-v2',
        '--text_encoder_path', '/home/curtis/checkpoints/roberta-base',
        '--data_txt_path_train', str(ROOT / 'datasets/data_path/refer-kitti-v2.train'),
    ] + (['--ttt_memory'] if enabled else []))
    return args


def digest_model(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith('ttt_memory.'):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load(args, baseline, ttt_checkpoint=None):
    model, criterion, _ = build_model(args)
    state = torch.load(baseline, map_location='cpu', weights_only=False)['model']
    # Profiling counters are not learned model parameters.
    state = {k: v for k, v in state.items()
             if not k.endswith(('total_params', 'total_ops'))}
    # Older Transformers serialized this deterministic buffer; newer ones do not.
    position_ids = state.pop('text_encoder.embeddings.position_ids', None)
    if position_ids is not None:
        assert torch.equal(position_ids, torch.arange(position_ids.numel()).view_as(position_ids))
    incompatible = model.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys, incompatible
    assert all(k.startswith('ttt_memory.') for k in incompatible.missing_keys), incompatible
    if ttt_checkpoint:
        checkpoint = torch.load(ttt_checkpoint, map_location='cpu', weights_only=False)
        assert digest_model(model) == checkpoint['baseline_digest']
        model.ttt_memory.load_state_dict(checkpoint['ttt'])
    return model.cuda(), criterion.cuda()


def override_inner_lr(model, value):
    """Set all fast-update rates for a preregistered sensitivity run."""
    if value is None:
        return
    if not np.isfinite(value) or value <= 0:
        raise ValueError('inner-lr override must be finite and positive')
    with torch.no_grad():
        model.ttt_memory.log_lr.fill_(math.log(math.expm1(value)))


def training(opts):
    args = config(opts.frames, True)
    args.ttt_tbptt = opts.tbptt
    args.ttt_disable_feature = opts.disable_feature
    args.ttt_disable_actions = opts.disable_actions
    model, criterion = load(args, opts.baseline, opts.init_checkpoint)
    override_inner_lr(model, opts.inner_lr_override)
    baseline_digest = digest_model(model)
    dataset = build_dataset('train', args)
    candidates = []
    for i in range(len(dataset)):
        paths = [Path(p) for p in dataset.img_files[i:i + opts.frames]]
        if (len(paths) == opts.frames and paths[0].parent.name not in HELD_OUT
                and all(p.parent == paths[0].parent for p in paths)
                and all(int(p.stem) == int(paths[0].stem) + j for j, p in enumerate(paths))):
            candidates.append(i)
    random.Random(opts.seed).shuffle(candidates)
    indices = candidates[:opts.clips]
    assert len(indices) == opts.clips
    manifest = [{'index': i, 'start': dataset.img_files[i],
                 'end': dataset.img_files[i + opts.frames - 1]} for i in indices]
    (opts.output / 'training_manifest.json').write_text(json.dumps(manifest, indent=2))
    groups = [dict(params=[p for n, p in model.ttt_memory.named_parameters() if p.ndim > 1],
                   weight_decay=1e-4),
              dict(params=[p for n, p in model.ttt_memory.named_parameters() if p.ndim <= 1],
                   weight_decay=0.0)]
    optimizer = torch.optim.AdamW(groups, lr=opts.outer_lr)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith('ttt_memory.') for n in trainable)
    initial = {n: p.detach().cpu().clone() for n, p in model.ttt_memory.named_parameters()}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def save_checkpoint(path, epoch, step):
        assert digest_model(model) == baseline_digest, 'Frozen baseline changed'
        torch.save(dict(ttt=model.ttt_memory.state_dict(), optimizer=optimizer.state_dict(),
                        baseline_digest=baseline_digest, baseline=str(opts.baseline),
                        init_checkpoint=str(opts.init_checkpoint) if opts.init_checkpoint else None,
                        epoch=epoch, step=step, seed=opts.seed, config=vars(args)), path)

    for epoch in range(opts.epochs):
        model.train()
        criterion.train()
        dataset.set_epoch(epoch)
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        for step, index in enumerate(indices):
            if time.monotonic() - started > opts.max_seconds:
                raise TimeoutError('Pilot wall-clock budget exceeded')
            # Reproducible augmentation/expression sampling, recorded per clip.
            clip_seed = opts.seed + epoch * opts.clips + step
            random.seed(clip_seed)
            np.random.seed(clip_seed)
            torch.manual_seed(clip_seed)
            data = dataset[index]
            sentence = data.pop('sentences')
            name = data.pop('dataset_name')
            data = data_dict_to_cuda(data, torch.device('cuda'))
            data.update(sentences=sentence, dataset_name=name)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(data)
            losses = criterion(outputs, data)
            ttt_losses = {key: float(value.detach()) for key, value in losses.items()
                          if '_ttt_' in key}
            selected_losses = {
                k: v for k, v in losses.items()
                if k in criterion.weight_dict
                and (opts.objective == 'all' or '_ttt_' in k)}
            if selected_losses:
                loss = sum(v * criterion.weight_dict[k]
                           for k, v in selected_losses.items())
            else:
                # A clip can contain no live history and therefore no TTT
                # supervision. Keep logging deterministic and skip its update.
                loss = next(iter(losses.values())).sum() * 0
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite tracking loss')
            # Clips with no usable history have no path to the only trainable module.
            has_task_gradient = loss.requires_grad
            if has_task_gradient:
                loss.backward()
            grads = {n: float(p.grad.norm()) if p.grad is not None else None
                     for n, p in model.ttt_memory.named_parameters()}
            assert all(p.grad is None for n, p in model.named_parameters()
                       if not n.startswith('ttt_memory.'))
            if any(v is not None and not np.isfinite(v) for v in grads.values()):
                raise FloatingPointError('Non-finite TTT gradient')
            norm = torch.nn.utils.clip_grad_norm_(model.ttt_memory.parameters(), 1.0)
            if has_task_gradient:
                optimizer.step()
            row = dict(epoch=epoch, step=step, index=index, seed=clip_seed,
                       sentence=sentence, loss=float(loss.detach()), grad_norm=float(norm),
                       objective=opts.objective, outer_lr=opts.outer_lr,
                       has_task_gradient=has_task_gradient,
                       grads=grads, ttt_losses=ttt_losses,
                       ttt_loss_sum=sum(ttt_losses.values()),
                       stats=model.ttt_memory.stats.copy(),
                       inner_lr=torch.nn.functional.softplus(
                           model.ttt_memory.log_lr).detach().cpu().tolist(),
                       alpha_mean=float(model.ttt_memory.alpha.detach().mean()),
                       peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                       elapsed=time.monotonic() - started)
            with (opts.output / 'train.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps({k: v for k, v in row.items()
                              if k not in ('grads', 'ttt_losses')}), flush=True)
            model.ttt_memory.reset()
            del outputs, losses, loss, data
            global_step = epoch * opts.clips + step + 1
            if opts.save_every and global_step % opts.save_every == 0:
                save_checkpoint(opts.output / ('ttt_step_%06d.pth' % global_step),
                                epoch, step)
        save_checkpoint(opts.output / ('ttt_epoch_%d.pth' % epoch),
                        epoch, len(indices) - 1)
    changed = {n: float((p.detach().cpu() - initial[n]).abs().max())
               for n, p in model.ttt_memory.named_parameters()}
    (opts.output / 'summary.json').write_text(json.dumps(dict(
        baseline_unchanged=True, baseline_digest=baseline_digest,
        trainable_parameters=sum(p.numel() for p in model.ttt_memory.parameters()),
        changed=changed, elapsed=time.monotonic() - started,
        peak_mib=torch.cuda.max_memory_allocated() / 2**20), indent=2))


def evaluation(opts):
    from inference import Detector
    args = config(opts.frames, bool(opts.checkpoint))
    args.ttt_mode = opts.mode
    args.ttt_disable_feature = opts.disable_feature
    args.ttt_disable_actions = opts.disable_actions
    args.output_dir = str(opts.output)
    args.max_frames = opts.max_frames
    model, _ = load(args, opts.baseline, opts.checkpoint)
    override_inner_lr(model, opts.inner_lr_override)
    model.eval()
    expression_root = Path(args.rmot_path) / 'expression'
    if opts.seqmap:
        selected = load_seqmap(opts.seqmap, expression_root)
    else:
        # Round-robin across fixed held-out videos, names selected without results.
        lists = [sorted((expression_root / v).glob('*.json')) for v in HELD_OUT]
        selected = [p for group in zip(*lists) for p in group]
    if opts.start_expression >= len(selected):
        raise ValueError('start-expression is outside the selected expression list')
    selected = selected[opts.start_expression:]
    expression_count = opts.expressions if opts.expressions is not None else (
        len(selected) if opts.seqmap else 20)
    selected = selected[:expression_count]
    if len(selected) != expression_count:
        raise ValueError(f'Requested {expression_count} expressions, found {len(selected)}')

    rendered_seqmap = ''.join(p.parent.name + '+' + p.stem + '\n' for p in selected)
    output_seqmap = opts.output / 'seqmap.txt'
    if output_seqmap.exists() and output_seqmap.read_text() != rendered_seqmap:
        raise RuntimeError(f'Resume seqmap differs from existing file: {output_seqmap}')
    output_seqmap.write_text(rendered_seqmap)
    manifest = dict(
        source=str(opts.seqmap) if opts.seqmap else None,
        source_sha256=hashlib.sha256(opts.seqmap.read_bytes()).hexdigest() if opts.seqmap else None,
        rendered_sha256=hashlib.sha256(rendered_seqmap.encode()).hexdigest(),
        count=len(selected),
        start_expression=opts.start_expression,
        result_tag=opts.result_tag,
    )
    manifest_path = opts.output / 'evaluation_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(f'Resume manifest differs from existing file: {manifest_path}')
    manifest_path.write_text(json.dumps(manifest, indent=2))

    completed_before_start = sum(
        is_evaluation_complete(opts.output, opts.result_tag, p) for p in selected)
    print(f'Official evaluation: {len(selected)} expressions; '
          f'{completed_before_start} already complete; '
          f'seqmap_sha256={manifest["source_sha256"]}', flush=True)
    run_started = time.monotonic()
    completed_this_run = 0
    for index, p in enumerate(selected, 1):
        if is_evaluation_complete(opts.output, opts.result_tag, p):
            print(f'[{index}/{len(selected)}] SKIP {p.parent.name}+{p.stem}', flush=True)
            continue
        started = time.monotonic()
        print(f'[{index}/{len(selected)}] RUN  {p.parent.name}+{p.stem}', flush=True)
        detector = Detector(args, opts.result_tag, model, (p.parent.name, p.name))
        detector.detect()
        completed_this_run += 1
        stats = model.ttt_memory.stats.copy() if opts.checkpoint else {}
        elapsed = time.monotonic() - started
        finished = completed_before_start + completed_this_run
        mean_elapsed = (time.monotonic() - run_started) / completed_this_run
        eta_hours = mean_elapsed * (len(selected) - finished) / 3600
        with (opts.output / 'evaluation.jsonl').open('a') as f:
            f.write(json.dumps(dict(index=index, total=len(selected),
                                   video=p.parent.name, expression=p.stem,
                                   elapsed=elapsed, eta_hours=eta_hours,
                                   stats=stats)) + '\n')
        print(f'[{index}/{len(selected)}] DONE {p.parent.name}+{p.stem} '
              f'in {elapsed:.1f}s; estimated remaining {eta_hours:.1f}h', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'eval'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline', type=Path, default=Path('/home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--mode', default='normal', choices=['normal', 'reset', 'no_update'])
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--clips', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expressions', type=int,
                        help='Number of expressions to run; defaults to all seqmap rows or 20 pilot rows.')
    parser.add_argument('--seqmap', type=Path,
                        help='Frozen video+expression list to evaluate in exact file order.')
    parser.add_argument('--start-expression', type=int, default=0,
                        help='Zero-based start row after loading the expression list.')
    parser.add_argument('--result-tag', default='pilot',
                        help='Safe directory suffix used under results_<tag>.')
    parser.add_argument('--resume-eval', action='store_true',
                        help='Reuse an evaluation directory and skip expressions with predict.txt and gt.txt.')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--max-seconds', type=int, default=14400)
    parser.add_argument('--tbptt', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=25,
                        help='Save a resumable diagnostic checkpoint every N clip updates; 0 disables it.')
    parser.add_argument('--outer-lr', type=float, default=1e-4,
                        help='AdamW learning rate for the trainable structured-TTT module.')
    parser.add_argument('--inner-lr-override', type=float,
                        help='Override all learned fast-update rates for a recorded sensitivity run.')
    parser.add_argument('--objective', choices=['all', 'structured'], default='all',
                        help='Use all tracking losses, or pretrain only the structured TTT auxiliary losses.')
    parser.add_argument('--disable-feature', action='store_true',
                        help='Ablate the TTT feature residual while retaining action prediction/execution.')
    parser.add_argument('--disable-actions', action='store_true',
                        help='Ablate candidate/lifecycle actions while retaining the TTT feature residual.')
    opts = parser.parse_args()
    if opts.start_expression < 0:
        parser.error('--start-expression must be non-negative')
    if opts.expressions is not None and opts.expressions < 1:
        parser.error('--expressions must be positive')
    if not opts.result_tag.replace('_', '').replace('-', '').isalnum():
        parser.error('--result-tag may contain only letters, numbers, underscores, and hyphens')
    if opts.action == 'train' and opts.resume_eval:
        parser.error('--resume-eval is only valid for evaluation')
    if opts.output.exists() and not opts.resume_eval:
        parser.error(f'output directory already exists: {opts.output}')
    opts.output.mkdir(parents=True, exist_ok=opts.resume_eval)
    invocation_path = opts.output / 'invocation.json'
    invocation = json.dumps(vars(opts), default=str, indent=2)
    if invocation_path.exists() and invocation_path.read_text() != invocation:
        parser.error(f'resume invocation differs from {invocation_path}')
    invocation_path.write_text(invocation)
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.set_num_threads(4)
    (training if opts.action == 'train' else evaluation)(opts)
