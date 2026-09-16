# TTT-only memory pilot

## Scope

Two-layer per-track fast MLP (256 -> 512 -> 256, GELU), inserted after STReasoner
temporal history reasoning and before its current-frame score/box/refer refinement
heads in both training and inference. Only `output_embedding` is replaced by a
residual-enhanced feature. The original heads consume that feature; frame
summarization, lifecycle, QIM, reference rules and output filters stay unchanged.
All baseline parameters are frozen when `--ttt_memory` is enabled.

The residual now influences current-frame refined predictions and lifecycle as well
as later frames through QIM. QIM still writes query content only for refined scores
> 0.5, and this module still cannot revive an already deleted ID.

## Algorithm

1. Project the history-reasoned STReasoner feature into q, k, v before the
   refinement heads.
2. Compute mean feature reconstruction loss `mean((MLP_old(k) - v)^2)` per track.
3. Take one gradient step on all four MLP tensors. The effective inner learning
   rate is `softplus(log_lr)`, initialized to 0.01 and trained by the outer loss.
4. Read `MLP_new(q)` and add `tanh(alpha) * readout` to the original feature.
   Alpha is a shared 256-vector initialized to 0.001. No extra output projection.
5. Permit writing for valid IDs and cache_scores > 0.5. No quality head or GT
   write gate. First write has no read residual; subsequent frames can use it.

States are stored sparsely by ID in the module, avoiding ~300 MiB of unused
weights for fresh Detect Queries. They are differentiable within a training clip,
detached in inference, pruned after QIM, and reset for each independent clip or
video-expression stream. The implementation supports one sequential stream per
model instance; interleaving streams through one instance is not supported.

Shared trainable parameters: **460,545**. Each live track has 262,912 dynamic
float32 state elements (~1.003 MiB), separate from the shared parameter count.

## Training

Use `tools/run_ttt_pilot.py`, not an unmodified generic full-model training recipe.
It strictly loads the baseline, optimizes only TTT with AdamW (lr=1e-4), clips
gradients at 1, and verifies all original state-dict tensors remain byte-identical
after every epoch. Matrix decay=1e-4; bias/gate/lr parameters have no weight decay.

The runner uses original tracking losses and original training transforms. It
samples continuous same-video starts only and records clip indices, expression,
random seed, losses, gradients, memory exposure and resource use. No training-data
files or original dataset code are rewritten.

The first pilot is **100 clips x 10 frames x 3 epochs**, not full-dataset training.
Different epochs may sample different expressions/augmentations at the same
starts. The three development videos 0002/0006/0010 are excluded from new TTT
training, but the pretrained baseline may have seen them. This is not blind
generalization evaluation. Official test videos are not used to tune this run.

Full differentiable unrolling is used inside each 10-frame clip; states are reset
between independent clips. Cross-clip TBPTT is **not implemented** in this pilot.
Clips with no task gradient are logged and skipped for optimizer updates, rather
than silently treating constant baseline losses as TTT supervision.

```bash
PYTHONPATH=. python tools/run_ttt_pilot.py train \
  --output /home/curtis/experiments/TempRMOT/ttt_memory/NEW_TRAIN_RUN \
  --clips 100 --epochs 3 --frames 10
```

Output directories must be new; existing experiment artifacts are not overwritten.
Checkpoints contain TTT shared weights, optimizer state, baseline digest and
configuration. Dynamic per-video memories are not saved as a global initializer.

## Evaluation and mechanism checks

```bash
PYTHONPATH=. python tools/check_ttt_integration.py --output /absolute/new/check.json
PYTHONPATH=. python tools/run_ttt_pilot.py eval \
  --output /absolute/new/baseline --expressions 20
PYTHONPATH=. python tools/run_ttt_pilot.py eval \
  --checkpoint /absolute/train/ttt_epoch_2.pth \
  --output /absolute/new/ttt --expressions 20
PYTHONPATH=. python tools/evaluate_ttt_pilot.py /absolute/new/ttt
```

Evaluation selects lexicographic expressions round-robin across held-out videos,
then runs the entire video for each expression (unless explicitly capped for a
smoke check). `--mode reset` removes accumulated fast weights each frame while
retaining the same history-valid exposure; `--mode no_update` reads the trained
initial MLP without inner updates. These are inference-time mechanism diagnostics,
not independently retrained equal-budget baselines.

Each run also records `history_delta_sum`, `history_delta_count`, and
`history_delta_max`. Before the current frame updates its fast weights, the module
compares the answer from the saved per-track weights with the answer from the
trained initial MLP on the same query. The statistic is detached, never enters the
residual or loss, and is therefore output-neutral. Its mean is
`history_delta_sum / history_delta_count`; only tracks with an existing saved state
are counted.

The metric wrapper uses vendored HOTA/CLEAR/Identity. The obsolete `np.int` alias
is supplied only inside the evaluator process; no vendored evaluator files change.
Metrics JSON stores fractional HOTA/DetA/AssA/IDF1/MOTA, not percentage points.

## Initial verification

- Module tests: first-write identity, nonzero subsequent residual, all shared
  parameter/QKV gradients, ID reorder/isolation, prune/reset, duplicate-ID
  rejection, inference no_grad and zero-gate exact identity.
- Real 2-clip, 10-frame training smoke: baseline digest unchanged, all 8 TTT
  parameter tensors updated; peak allocation 2987.87 MiB, ~14.5 s training.
- The original post-head insertion pilot did not improve tracking metrics; see
  [post-head results](TTT_MEMORY_PILOT_RESULTS_20260910.md).
- The current pre-head insertion was retrained with the same 100 x 3 x 10 budget.
  It affected current predictions but degraded HOTA/AssA/IDF1 on the fixed six
  expressions; see [pre-head results](TTT_MEMORY_BEFORE_HEADS_RESULTS_20260910.md).

No claim of improved tracking performance follows from successful training alone.
