#!/usr/bin/env python3
"""Enforce the preregistered official-output oracle tolerance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from refer_kitti_v1_common import DEFAULT_OUTPUT_ROOT, atomic_write_json


OFFICIAL = {"HOTA": 38.06, "DetA": 29.28, "AssA": 50.83}
TOLERANCE = 0.01


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT_ROOT / "phase2_oracle_metrics.json")
    args = parser.parse_args()
    lines = [line.split() for line in args.summary.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 2 or len(lines[0]) != len(lines[1]):
        raise RuntimeError(f"unexpected TrackEval summary format: {args.summary}")
    values = {name: float(value) for name, value in zip(lines[0], lines[1])}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("TrackEval summary contains NaN/Inf")
    comparisons = {
        metric: {
            "observed": values[metric],
            "official": official,
            "absolute_difference": abs(values[metric] - official),
            "passed": abs(values[metric] - official) <= TOLERANCE,
        }
        for metric, official in OFFICIAL.items()
    }
    report = {
        "ok": all(item["passed"] for item in comparisons.values()),
        "summary": str(args.summary),
        "tolerance": TOLERANCE,
        "metrics": values,
        "comparisons": comparisons,
    }
    atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise RuntimeError("official-output oracle tolerance failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
