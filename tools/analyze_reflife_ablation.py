#!/usr/bin/env python3
"""Compare RefLife ablations with frame-level GT matching and audit statistics."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def read_mot(path):
    frames = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            frame, track_id = int(row[0]), int(row[1])
            x, y, w, h = map(float, row[2:6])
            frames[frame].append((track_id, np.array([x, y, x + w, y + h])))
    return frames


def pairwise_iou(left, right):
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float64)
    a = np.stack([box for _, box in left])
    b = np.stack([box for _, box in right])
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(axis=1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def matched_gt_events(gt_path, prediction_path, threshold):
    gt = read_mot(gt_path)
    prediction = read_mot(prediction_path)
    matched = set()
    for frame, gt_rows in gt.items():
        pred_rows = prediction.get(frame, [])
        iou = pairwise_iou(gt_rows, pred_rows)
        if not iou.size:
            continue
        gt_indices, pred_indices = linear_sum_assignment(1.0 - iou)
        for gt_index, pred_index in zip(gt_indices, pred_indices):
            if iou[gt_index, pred_index] >= threshold:
                matched.add((frame, gt_rows[gt_index][0]))
    return matched, sum(len(rows) for rows in prediction.values())


def audit_stats(path):
    values = []
    uncertainty = []
    residuals = []
    similarities = []
    track_rows = predicted_rows = predicted_present = effective_present = 0
    dormant = long_dormant = ref_updates = feedback_valid = 0
    feedback_consistent = feedback_uncertain = feedback_contradictory = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["query_id"]) < 0:
                continue
            track_rows += 1
            if int(row["prediction_valid"]):
                predicted_rows += 1
                probability = float(row["presence_probability"])
                values.append(probability)
                uncertainty.append(float(row["prediction_uncertainty"]))
                predicted_present += probability >= 0.6
                effective_present += probability >= 0.6 and int(row["feedback_state"]) != 2
            if int(row["feedback_valid"]):
                feedback_valid += 1
                residuals.append(float(row["box_residual"]))
                similarities.append(float(row["identity_similarity"]))
                state = int(row["feedback_state"])
                feedback_uncertain += state == 0
                feedback_consistent += state == 1
                feedback_contradictory += state == 2
            age = int(row["dormant_age"])
            dormant += age > 0
            long_dormant += age >= 10
            ref_updates += int(row["reference_source"]) != 0
    quantiles = np.quantile(values, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).tolist() if values else []
    uncertainty_quantiles = (
        np.quantile(uncertainty, [0, 0.5, 0.9, 1]).tolist() if uncertainty else []
    )
    return {
        "track_rows": track_rows,
        "prediction_valid": predicted_rows,
        "predicted_present": predicted_present,
        "effective_present": effective_present,
        "feedback_valid": feedback_valid,
        "feedback_consistent": feedback_consistent,
        "feedback_uncertain": feedback_uncertain,
        "feedback_contradictory": feedback_contradictory,
        "dormant_rows": dormant,
        "dormant_age_ge_10": long_dormant,
        "reference_updates": ref_updates,
        "presence_quantiles": quantiles,
        "uncertainty_quantiles": uncertainty_quantiles,
        "presence_values": values,
        "uncertainty_values": uncertainty,
        "residual_values": residuals,
        "similarity_values": similarities,
    }


def discover_expressions(root):
    result_root = root / "results_checkpoint" / "0005"
    return sorted(path.name for path in result_root.iterdir() if (path / "predict.txt").exists())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--baseline", default="A_off")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    variants = {}
    for spec in args.variant:
        name, path = spec.split("=", 1)
        variants[name] = Path(path)
    expressions = discover_expressions(variants[args.baseline])
    all_matches = defaultdict(dict)

    for expression in expressions:
        print(f"\n[{expression}]")
        for name, root in variants.items():
            result = root / "results_checkpoint" / "0005" / expression
            matches, output_rows = matched_gt_events(
                result / "gt.txt", result / "predict.txt", args.iou
            )
            all_matches[name][expression] = matches
            print(f"{name}: outputs={output_rows}, matched_gt={len(matches)}")
        baseline = all_matches[args.baseline][expression]
        for name in variants:
            if name == args.baseline:
                continue
            current = all_matches[name][expression]
            print(
                f"  {name} vs {args.baseline}: "
                f"baseline_only={len(baseline-current)}, variant_only={len(current-baseline)}"
            )

    print("\n[TOTAL]")
    baseline_total = set()
    for expression, events in all_matches[args.baseline].items():
        baseline_total |= {(expression, *event) for event in events}
    totals = {}
    for name in variants:
        current_total = set()
        output_total = 0
        for expression, events in all_matches[name].items():
            current_total |= {(expression, *event) for event in events}
            prediction_path = (
                variants[name] / "results_checkpoint" / "0005" / expression / "predict.txt"
            )
            output_total += sum(len(rows) for rows in read_mot(prediction_path).values())
        totals[name] = current_total
        print(
            f"{name}: outputs={output_total}, matched_gt={len(current_total)}, "
            f"unmatched_outputs={output_total-len(current_total)}, "
            f"baseline_only={len(baseline_total-current_total)}, "
            f"variant_only={len(current_total-baseline_total)}"
        )
    names = list(variants)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if {left, right} <= {args.baseline, "B_observer", "C_lifecycle"}:
                continue
            print(
                f"{left} vs {right}: {left}_only={len(totals[left]-totals[right])}, "
                f"{right}_only={len(totals[right]-totals[left])}"
            )

    print("\n[AUDIT]")
    for name, root in variants.items():
        audit_root = root / "audit" / "0005"
        if not audit_root.exists():
            continue
        combined = defaultdict(int)
        presence = []
        uncertainty = []
        residuals = []
        similarities = []
        for expression in expressions:
            stats = audit_stats(audit_root / f"{expression}.csv")
            for key in (
                "track_rows", "prediction_valid", "predicted_present", "effective_present",
                "feedback_valid", "feedback_consistent", "feedback_uncertain",
                "feedback_contradictory", "dormant_rows", "dormant_age_ge_10",
                "reference_updates",
            ):
                combined[key] += stats[key]
            presence.extend(stats["presence_values"])
            uncertainty.extend(stats["uncertainty_values"])
            residuals.extend(stats["residual_values"])
            similarities.extend(stats["similarity_values"])
        print(name, dict(combined))
        if presence:
            print(
                "  pooled presence quantiles:",
                [round(x, 4) for x in np.quantile(presence, [0, .1, .25, .5, .75, .9, 1])],
            )
            print(
                "  pooled uncertainty quantiles:",
                [round(x, 4) for x in np.quantile(uncertainty, [0, .1, .5, .9, 1])],
            )
        if residuals:
            print(
                "  pooled box-residual quantiles:",
                [round(x, 4) for x in np.quantile(residuals, [0, .1, .5, .9, 1])],
            )
            print(
                "  pooled identity-similarity quantiles:",
                [round(x, 4) for x in np.quantile(similarities, [0, .1, .5, .9, 1])],
            )

    if args.checkpoint:
        import torch

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        print("\n[FUTURE PREDICTOR CHECKPOINT]")
        for key in sorted(key for key in state if "future_predictor" in key):
            tensor = state[key].float()
            print(
                f"{key}: shape={tuple(tensor.shape)}, norm={tensor.norm().item():.6f}, "
                f"mean={tensor.mean().item():.6f}"
            )


if __name__ == "__main__":
    main()
