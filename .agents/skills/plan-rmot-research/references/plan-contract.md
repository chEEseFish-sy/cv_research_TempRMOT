# Research plan contract

Every plan states:

- research question and falsifiable hypothesis;
- current evidence and unresolved assumptions;
- baseline and comparison conditions;
- independent, dependent, and controlled variables;
- datasets, splits, leakage controls, and annotation needs;
- primary and diagnostic metrics;
- compute, storage, wall-clock, and human-time ceilings;
- phases with one concrete deliverable each;
- Go/Revise/Stop thresholds;
- risks, confounds, fallback route, and prohibited scope expansion;
- the single next action.

Approval is phase-scoped. `ALL` is valid only when the user explicitly approves the whole
plan. Resource ceilings enforced by the executor must be finite numeric values. Approval
requires a current feasibility review containing `Decision: GO` and the exact
`Plan-SHA256` of `task_plan.md`.
