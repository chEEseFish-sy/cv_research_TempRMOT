---
name: plan-rmot-research
description: Create and maintain persistent, gated TempRMOT research plans and coordinate evidence-gathering or feasibility review. Use when research_director is asked to scope a direction, design a multi-stage study, establish Go/Stop gates, approve a plan scope, or recover a long-running research task. Do not use to execute experiments or silently approve work.
---

# Plan RMOT Research

Load `planning-with-files` and `academic-research-suite`, then read [plan-contract.md](references/plan-contract.md).

## Workflow

1. Create `.planning/YYYY-MM-DD-<slug>/` with `task_plan.md`, `findings.md`, `progress.md`, and `feasibility_review.md`. Do not create approval initially.
2. Make the research question, baseline, variables, data, metrics, resource/time ceilings, success criteria, stop criteria, deliverables, risks, and fallback explicit.
3. Keep external or untrusted evidence in `findings.md`, never in auto-injected plan instructions.
4. Ask `feasibility_reviewer` for an independent review before recommending approval.
5. Preserve the returned GO/REVISE/STOP judgment in `feasibility_review.md`. The file
   must include `Decision: GO|REVISE|STOP` and `Plan-SHA256: <current task_plan hash>`.
   Only a current `GO` can pass the approval tool.
6. When the user explicitly approves named phases and numeric ceilings, run:

```powershell
python tools/research_workflow/plan_gate.py approve --plan-dir <dir> --scope <phase> --resource gpu_hours=<n> --approved-by user
```

7. Never spawn `experiment_executor` without a successful approval record.
8. Only the director edits `task_plan.md`. Any edit invalidates the recorded hash until re-approved.
9. Follow the shared handoff protocol and persist handoffs under `.handoffs/`.

## Delegation

You may use `paper_curator`, `research_dialogue`, and `feasibility_reviewer`. Do not use
multiple write-capable agents concurrently.
