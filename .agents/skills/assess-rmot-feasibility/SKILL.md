---
name: assess-rmot-feasibility
description: Independently assess a TempRMOT research plan for scientific value, novelty, falsifiability, data adequacy, evaluation validity, resource cost, reproducibility, and publication risk. Use when feasibility_reviewer is asked for a GO, REVISE, or STOP decision. Do not use to edit, rescue, approve, or execute the plan.
---

# Assess RMOT Feasibility

Load `academic-research-suite` and read [review-rubric.md](references/review-rubric.md).

## Review

1. Read the plan, relevant project state, and cited paper notes read-only.
2. Evaluate all ten rubric lanes independently.
3. Identify fatal blockers before optimizations.
4. Specify the smallest experiment that resolves the most important uncertainty.
5. Define observable failure thresholds and a lower-risk fallback.
6. Return exactly one top-level decision and the reviewed plan hash using:
   `Decision: GO|REVISE|STOP` and `Plan-SHA256: <sha256>`.
7. Include evidence paths and distinguish fact from inference.

Do not rewrite the plan, create approval, modify notes, or accommodate the director's
preferred conclusion. Follow the shared handoff protocol and emit handoff content to the
parent when due.
