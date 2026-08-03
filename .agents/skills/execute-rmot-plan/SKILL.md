---
name: execute-rmot-plan
description: Verify and execute exactly one user-approved phase of a persistent TempRMOT research plan, then record reproducible progress and evidence. Use when experiment_executor is asked to implement, run, or validate an approved experiment. Do not use without a valid approval hash and scope, or to change research goals and success criteria.
---

# Execute RMOT Plan

Load `planning-with-files` and read [execution-contract.md](references/execution-contract.md).

## Gate

Before any mutation or costly command, run:

```powershell
python tools/research_workflow/plan_gate.py verify --plan-dir <dir> --phase <phase> --resource gpu_hours=<requested>
```

Stop on missing approval, hash mismatch, scope mismatch, unknown resource, or exceeded ceiling.

## Execution

1. Read the approved phase, success/stop conditions, and prohibited changes.
2. Execute only the smallest next atomic action inside scope.
3. Preserve unrelated dirty worktree changes.
4. Record commands, environment, outputs, metrics, errors, and decisions in `progress.md` and `findings.md`.
5. Never edit `task_plan.md` or `approval.json`.
6. Run proportional validation after changes.
7. Do not commit, push, edit Zotero, or write Obsidian.
8. Return to the director when evidence requires any plan change.
9. Follow the shared handoff protocol and stop at an atomic boundary when due.
