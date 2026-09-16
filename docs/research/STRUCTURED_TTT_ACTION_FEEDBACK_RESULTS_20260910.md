# Structured TTT Track Memory: implementation and pilot results

Date: 2026-09-10

## Scope

This report records the implementation and bounded Refer-KITTI-V2 pilot for a
track-private test-time-training (TTT) memory inserted after STReasoner and
before its refined score, box, and referring heads. The original TempRMOT
parameters are frozen. No dataset or baseline checkpoint was modified.

## Implemented path

```text
Decoder
-> STReasoner history reasoning
-> temporary score / box / refer evidence
-> Identity, Evidence, previous Action-Feedback tokens
-> per-track 256 -> 512 -> 256 fast-model update
-> updated-memory candidate retrieval
-> bounded feature residual and Action head
-> original refined heads, frame summarization, lifecycle, and QIM
```

The outer module has 2,118,538 trainable parameters. Each active track owns
262,912 fast-state values, about 1 MiB in FP32. Runtime actions are TRUST,
REASSOCIATE, HOLD, and TERMINATE. Detect-candidate features or boxes are changed
only when REASSOCIATE is the hard inference action.

## Training

- Baseline: `/home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth`
- Data: existing Refer-KITTI-V2 training split; held-out videos 0002/0006/0010
- Warm-up: 60 updates on 16-frame clips, structured losses only
- Selected continuation: 25 updates on 24-frame clips, outer LR `3e-4`, TBPTT 8
- Selected checkpoint SHA-256:
  `f76cbf98b2e966e38dc5bed88f662b44e53da241342add5b3fe378be8c927121`
- Selected checkpoint:
  `/home/curtis/experiments/TempRMOT/structured_ttt/20260910_adaptation_train24_80x3_lr3e4/ttt_step_000025.pth`

The candidate objective includes a counterfactual adaptation term: after the
online write, candidate cross-entropy is trained to improve over the detached
history-free fast model. It uses a smooth softplus difference and no manually
chosen quality margin.

## Causal memory comparison: fixed six expressions

| Mode | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|
| `normal` | 85.4364 | 81.1433 | 89.9713 | 89.5920 | 81.4361 | 3 | 18 |
| `reset` | 81.5900 | 80.6989 | 82.5187 | 83.5788 | 80.6188 | 7 | 18 |
| `no_update` | 81.5900 | 80.6989 | 82.5187 | 83.5788 | 80.6188 | 7 | 18 |

`reset == no_update`, while accumulated `normal` improves HOTA by 3.8464,
AssA by 7.4526, and IDF1 by 6.0132 percentage points and reduces IDSW from 7
to 3. This is the current direct evidence for cross-frame fast-weight memory.

## Expanded fixed-20 pilot

The two manifests are SHA-256-identical and contain 20 expressions.

| Model | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen TempRMOT | 81.0189 | 81.1646 | 80.9040 | 83.2354 | 81.4830 | 14 | 23 |
| Structured TTT `normal` | 84.2827 | 83.2573 | 85.3462 | 87.9060 | 84.1342 | 14 | 29 |
| Difference | +3.2637 | +2.0926 | +4.4422 | +4.6707 | +2.6512 | 0 | +6 |

The pilot supports improvements in detection and association, but not every
failure mode improved: IDSW is unchanged and fragmentation is six events worse.
These results are not a complete V2 validation result and must not be reported
as such.

Mean inference time was 50.81 seconds per expression for the baseline and
58.16 seconds for TTT, a 14.5% increase in this local run.

## Resource and correctness checks

- Full repository unit suite: 26/26 passed.
- Real ten-frame zero-gate test: bit-exact, maximum difference 0.
- 24-frame selected training peak: 7,262.95 MiB before the selected checkpoint.
- A 32-frame smoke reached 8,034 MiB and was stopped.
- A later high-density 24-frame clip reached 7,838 MiB and was stopped before
  OOM; its incomplete run was not selected.
- No non-finite losses, cross-ID state reuse, GT/future access at inference, or
  baseline-weight mutation was observed.

## Official Refer-KITTI-V2 evaluation

The selected checkpoint was subsequently evaluated on all 862 expressions from
the frozen official seqmap (`0005`, `0011`, `0013`, and `0019`). All 862
`predict.txt` and 862 `gt.txt` files were produced; 48 prediction files were
legitimately empty after output filtering.

| Model | HOTA | DetA | AssA |
|---|---:|---:|---:|
| Frozen TempRMOT reproduction | 35.249 | 23.412 | 53.191 |
| Structured TTT | 34.711 | 22.928 | 52.669 |
| Difference | -0.538 | -0.484 | -0.522 |

Additional Structured TTT metrics were IDF1 36.132, MOTA -7.493, IDSW 2,199,
and Frag 7,004. The previous baseline report did not run CLEAR and Identity in
the same evaluation call, so these four values must not be presented as deltas
until the existing baseline predictions are reevaluated with the same metrics.

The hard action distribution was 1,471,728 TRUST (26.40%), 527,123 REASSOCIATE
(9.46%), 3,575,704 HOLD (64.14%), and zero TERMINATE. The high HOLD rate and
missing TERMINATE behavior are primary audit targets.

Evidence:

```text
/home/curtis/experiments/TempRMOT/structured_ttt/official_v2_full_step25/
  invocation.json
  evaluation_manifest.json
  evaluation.jsonl
  metrics.json
  results_structured_ttt_step25/
```

## Evidence boundary and next work

The six- and twenty-expression pilots show that accumulated fast weights can
change useful local decisions, but that effect did not generalize to a full-set
improvement. The current implementation is therefore a validated engineering
prototype with a negative benchmark result, not a paper-ready method claim.

Required next work is event-balanced training, aligned baseline CLEAR/Identity
evaluation, per-event failure analysis, at least three seeds, and ablations for
Identity, Evidence, Feedback, Action, candidate retrieval, explicit feature
banks, and a longer fixed STReasoner. TERMINATE had no teacher exposure in the
selected training curriculum and must not yet be claimed as learned.
