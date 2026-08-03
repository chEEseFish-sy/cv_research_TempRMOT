# Feasibility rubric

Score each lane from 1 to 5 and justify it with evidence:

1. Scientific value.
2. Novelty versus adjacent work.
3. Falsifiability.
4. Data and annotation adequacy.
5. Baseline fairness.
6. Metric sensitivity and confounds.
7. Compute and storage feasibility.
8. Schedule and human workload.
9. Reproducibility and dependency risk.
10. Publication story and venue risk.

Decision rules:

- `GO`: no fatal blocker and the minimum experiment fits the stated envelope.
- `REVISE`: promising, but resolvable gates must be redesigned first.
- `STOP`: core novelty, data, identifiability, or resource assumptions are not defensible.

Always return fatal blockers, key unknowns, minimum experiment, failure thresholds, and a
lower-risk fallback.
