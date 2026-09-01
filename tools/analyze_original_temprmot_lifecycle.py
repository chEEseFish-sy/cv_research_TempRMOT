#!/usr/bin/env python3
"""Diagnose original TempRMOT failures from decision-neutral candidate audits."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def xywh_to_xyxy(row):
    x, y, width, height = map(float, row[2:6])
    return np.array([x, y, x + width, y + height], dtype=np.float64)


def iou(box, boxes):
    if not boxes:
        return np.zeros(0, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=1)
    area = np.clip(box[2:] - box[:2], 0, None).prod()
    other_area = np.clip(boxes[:, 2:] - boxes[:, :2], 0, None).prod(axis=1)
    union = area + other_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def read_mot(path):
    frames = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if row:
                frames[int(row[0])].append({"id": int(row[1]), "box": xywh_to_xyxy(row)})
    return frames


def read_audit(path):
    frames = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["frame_id"] = int(row["frame_id"])
            row["slot_index"] = int(row["slot_index"])
            for key in (
                "is_track_query", "query_id", "query_id_before", "created_this_frame",
                "deleted_this_frame", "qim_selected", "qim_state_updated",
                "pass_all", "pass_emit",
            ):
                row[key] = int(row[key])
            for key in (
                "object_score", "refer_score", "area", "disappear_time",
                "disappear_time_before",
            ):
                row[key] = float(row[key])
            row["box"] = np.array(
                [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                dtype=np.float64,
            )
            frames[row["frame_id"]].append(row)
    return frames


def hungarian_matches(left, right, threshold):
    if not left or not right:
        return {}
    matrix = np.stack([iou(item["box"], [candidate["box"] for candidate in right]) for item in left])
    left_indices, right_indices = linear_sum_assignment(1.0 - matrix)
    return {
        left_index: right_index
        for left_index, right_index in zip(left_indices, right_indices)
        if matrix[left_index, right_index] >= threshold
    }


def best_candidate(gt, candidates):
    if not candidates:
        return None, 0.0
    overlaps = iou(gt["box"], [candidate["box"] for candidate in candidates])
    index = int(overlaps.argmax())
    return candidates[index], float(overlaps[index])


def find_query(candidates, query_id):
    """Find one model query by its zero-based obj_idx, including deletion rows."""
    for row in candidates:
        if row["query_id"] == query_id:
            return row
    for row in candidates:
        if row["query_id_before"] == query_id:
            return row
    return None


def classify_miss(gt, candidates, threshold):
    track = [row for row in candidates if row["is_track_query"]]
    detect = [row for row in candidates if not row["is_track_query"]]
    best_track, track_iou = best_candidate(gt, track)
    best_detect, detect_iou = best_candidate(gt, detect)

    if track_iou >= threshold:
        if best_track["deleted_this_frame"]:
            return "lifecycle_deleted_matching_track", best_track, track_iou
        if best_track["query_id"] < 0:
            return "track_invalid_id", best_track, track_iou
        if best_track["object_score"] <= 0.6:
            return "track_object_score", best_track, track_iou
        if not best_track["pass_emit"]:
            return "track_not_emitted", best_track, track_iou
        if best_track["area"] <= 100.0:
            return "track_area", best_track, track_iou
        if best_track["refer_score"] <= 0.4:
            return "track_refer_score", best_track, track_iou
        return "track_other", best_track, track_iou

    if detect_iou >= threshold:
        if best_detect["created_this_frame"]:
            if best_detect["refer_score"] <= 0.4:
                return "detect_created_refer_score", best_detect, detect_iou
            if best_detect["area"] <= 100.0:
                return "detect_created_area", best_detect, detect_iou
            return "detect_created_other", best_detect, detect_iou
        if best_detect["object_score"] < 0.6:
            return "detect_not_acquired_object", best_detect, detect_iou
        return "detect_not_acquired_other", best_detect, detect_iou

    return "decoder_no_candidate", None, max(track_iou, detect_iou)


def contiguous_episodes(frames):
    if not frames:
        return []
    episodes = [[frames[0]]]
    for frame in frames[1:]:
        if frame == episodes[-1][-1] + 1:
            episodes[-1].append(frame)
        else:
            episodes.append([frame])
    return episodes


def analyze_expression(result_dir, audit_path, threshold):
    gt_frames = read_mot(result_dir / "gt.txt")
    output_frames = read_mot(result_dir / "predict.txt")
    audit_frames = read_audit(audit_path)
    output_matches = {}
    event_category = {}
    event_detail = {}
    gt_presence = defaultdict(list)

    for frame, gt_rows in gt_frames.items():
        for gt in gt_rows:
            gt_presence[gt["id"]].append(frame)
        output_match = hungarian_matches(gt_rows, output_frames.get(frame, []), threshold)
        for gt_index, gt in enumerate(gt_rows):
            key = (frame, gt["id"])
            if gt_index in output_match:
                output_matches[key] = output_frames[frame][output_match[gt_index]]["id"]
                event_category[key] = "output_match"
            else:
                category, candidate, overlap = classify_miss(
                    gt, audit_frames.get(frame, []), threshold
                )
                event_category[key] = category
                event_detail[key] = (candidate, overlap)

    categories = Counter(event_category.values())
    missing_episodes = []
    for gt_id, frames in gt_presence.items():
        ordered_frames = sorted(frames)
        missing_frames = [frame for frame in ordered_frames if (frame, gt_id) not in output_matches]
        for episode in contiguous_episodes(missing_frames):
            previous = [frame for frame in ordered_frames if frame < episode[0] and (frame, gt_id) in output_matches]
            following = [frame for frame in ordered_frames if frame > episode[-1] and (frame, gt_id) in output_matches]
            previous_id = output_matches[(previous[-1], gt_id)] if previous else None
            next_id = output_matches[(following[0], gt_id)] if following else None
            missing_episodes.append(
                {
                    "gt_id": gt_id,
                    "start": episode[0],
                    "end": episode[-1],
                    "length": len(episode),
                    "start_category": event_category[(episode[0], gt_id)],
                    "had_prior_output": previous_id is not None,
                    "recovers": next_id is not None,
                    "previous_output_id": previous_id,
                    "next_output_id": next_id,
                    "recovers_with_new_id": (
                        previous_id is not None
                        and next_id is not None
                        and previous_id != next_id
                    ),
                    "previous_output_frame": previous[-1] if previous else None,
                }
            )

    # An output ID is obj_idx + 1 in inference.py. Inspect whether the same
    # internal query survives and whether QIM updates or freezes it at the
    # first missing frame.
    for episode in missing_episodes:
        episode["same_query_at_start"] = False
        episode["same_query_iou_at_start"] = None
        episode["same_query_qim_selected"] = None
        episode["same_query_qim_state_updated"] = None
        episode["same_query_score"] = None
        episode["same_query_refer"] = None
        if episode["previous_output_id"] is None:
            continue
        model_query_id = episode["previous_output_id"] - 1
        row = find_query(audit_frames.get(episode["start"], []), model_query_id)
        if row is None:
            continue
        target = [
            gt for gt in gt_frames[episode["start"]]
            if gt["id"] == episode["gt_id"]
        ][0]
        episode["same_query_at_start"] = True
        episode["same_query_iou_at_start"] = float(iou(target["box"], [row["box"]])[0])
        episode["same_query_qim_selected"] = bool(row["qim_selected"])
        episode["same_query_qim_state_updated"] = bool(row["qim_state_updated"])
        episode["same_query_score"] = row["object_score"]
        episode["same_query_refer"] = row["refer_score"]

    # Associate model track IDs with GT identities using visible, geometrically matched Track Queries.
    track_to_gt = {}
    last_track_match = {}
    deletion_records = []
    for frame in sorted(audit_frames):
        gt_rows = gt_frames.get(frame, [])
        track_rows = [row for row in audit_frames[frame] if row["is_track_query"]]
        matches = hungarian_matches(gt_rows, track_rows, threshold)
        for gt_index, track_index in matches.items():
            row = track_rows[track_index]
            track_id = row["query_id_before"]
            if track_id >= 0:
                track_to_gt[track_id] = gt_rows[gt_index]["id"]
                last_track_match[track_id] = frame

        for row in track_rows:
            if not row["deleted_this_frame"]:
                continue
            track_id = row["query_id_before"]
            gt_id = track_to_gt.get(track_id)
            future = [] if gt_id is None else [f for f in gt_presence[gt_id] if f > frame]
            current_iou = 0.0
            if gt_id is not None:
                current = [gt for gt in gt_rows if gt["id"] == gt_id]
                if current:
                    current_iou = float(iou(current[0]["box"], [row["box"]])[0])
            deletion_records.append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "gt_id": gt_id,
                    "last_match": last_track_match.get(track_id),
                    "current_iou": current_iou,
                    "future_reappears": bool(future),
                    "reappearance_gap": future[0] - frame if future else None,
                    "score": row["object_score"],
                    "refer": row["refer_score"],
                }
            )

    # Check whether a GT identity is later re-created by a Detect Query after a deletion.
    for deletion in deletion_records:
        deletion["recreated"] = False
        deletion["recreated_gap"] = None
        gt_id = deletion["gt_id"]
        if gt_id is None:
            continue
        for frame in sorted(f for f in gt_frames if f > deletion["frame"]):
            target = [gt for gt in gt_frames[frame] if gt["id"] == gt_id]
            created = [row for row in audit_frames.get(frame, []) if row["created_this_frame"]]
            if not target or not created:
                continue
            _, overlap = best_candidate(target[0], created)
            if overlap >= threshold:
                deletion["recreated"] = True
                deletion["recreated_gap"] = frame - deletion["frame"]
                break

        gt_visible_frames = gt_presence.get(gt_id, [])
        immediate = [frame for frame in gt_visible_frames if deletion["frame"] <= frame <= deletion["frame"] + 2]
        deletion["immediate_missing_output"] = any(
            (frame, gt_id) not in output_matches for frame in immediate
        )

    for episode in missing_episodes:
        previous_query_id = (
            episode["previous_output_id"] - 1
            if episode["previous_output_id"] is not None else None
        )
        relevant = [
            row for row in deletion_records
            if row["track_id"] == previous_query_id
            and episode["start"] - 10 <= row["frame"] <= episode["start"]
        ]
        episode["recent_deletion"] = bool(relevant)
        during = [
            row for row in deletion_records
            if row["track_id"] == previous_query_id
            and episode["start"] < row["frame"] <= episode["end"]
        ]
        episode["deletion_during_episode"] = bool(during)
        episode["deletion_frame"] = during[0]["frame"] if during else None

    return {
        "gt_events": sum(len(rows) for rows in gt_frames.values()),
        "output_rows": sum(len(rows) for rows in output_frames.values()),
        "categories": categories,
        "episodes": missing_episodes,
        "deletions": deletion_records,
        "missing_event_rows": [
            (event_category[key], detail[0])
            for key, detail in event_detail.items()
        ],
        "frames": len(audit_frames),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--video", default="0005")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    result_root = args.root / "results_checkpoint_rk2" / args.video
    expressions = sorted(path.name for path in result_root.iterdir() if path.is_dir())
    total_categories = Counter()
    total_deletions = []
    total_episodes = []
    total_event_details = []
    total_gt = total_outputs = 0

    for expression in expressions:
        result = analyze_expression(
            result_root / expression,
            args.audit_root / args.video / f"{expression}.csv",
            args.iou,
        )
        total_gt += result["gt_events"]
        total_outputs += result["output_rows"]
        total_categories.update(result["categories"])
        total_deletions.extend((expression, row) for row in result["deletions"])
        total_episodes.extend((expression, row) for row in result["episodes"])
        total_event_details.extend(
            (expression, category, row)
            for category, row in result["missing_event_rows"]
        )
        deletion_summary = Counter(
            "future_reappears" if row["future_reappears"] else "no_future_gt"
            for row in result["deletions"] if row["gt_id"] is not None
        )
        print(f"\n[{expression}]")
        print(f"frames={result['frames']} gt_events={result['gt_events']} outputs={result['output_rows']}")
        print("categories=", dict(result["categories"]))
        print(f"missing_episodes={len(result['episodes'])} deletion_summary={dict(deletion_summary)}")

    print("\n[TOTAL]")
    print(f"gt_events={total_gt} outputs={total_outputs}")
    print("categories=", dict(total_categories))
    print(f"missing_episodes={len(total_episodes)}")
    print("episode_starts=", dict(Counter(row["start_category"] for _, row in total_episodes)))
    print(
        "episode_history=",
        {
            "initial_acquisition": sum(not row["had_prior_output"] for _, row in total_episodes),
            "after_prior_tracking": sum(row["had_prior_output"] for _, row in total_episodes),
            "recovers": sum(row["recovers"] for _, row in total_episodes),
            "recovers_with_new_id": sum(row["recovers_with_new_id"] for _, row in total_episodes),
            "previous_query_deleted_before_start": sum(
                row["recent_deletion"] for _, row in total_episodes
            ),
            "previous_query_deleted_during_episode": sum(
                row["deletion_during_episode"] for _, row in total_episodes
            ),
        },
    )
    print(
        "initial_episode_starts=",
        dict(Counter(row["start_category"] for _, row in total_episodes if not row["had_prior_output"])),
    )
    print(
        "tracked_episode_starts=",
        dict(Counter(row["start_category"] for _, row in total_episodes if row["had_prior_output"])),
    )
    tracked_episodes = [row for _, row in total_episodes if row["had_prior_output"]]
    same_query_episodes = [row for row in tracked_episodes if row["same_query_at_start"]]
    print(
        "tracked_episode_query_state=",
        {
            "same_query_present": len(same_query_episodes),
            "same_query_geometry_matches": sum(
                row["same_query_iou_at_start"] >= args.iou
                for row in same_query_episodes
            ),
            "selected_and_updated": sum(
                row["same_query_qim_selected"]
                and row["same_query_qim_state_updated"]
                for row in same_query_episodes
            ),
            "selected_but_frozen": sum(
                row["same_query_qim_selected"]
                and not row["same_query_qim_state_updated"]
                for row in same_query_episodes
            ),
            "not_selected": sum(
                not row["same_query_qim_selected"]
                for row in same_query_episodes
            ),
        },
    )
    track_object_events = [
        row for _, category, row in total_event_details
        if category == "track_object_score" and row is not None
    ]
    track_refer_events = [
        row for _, category, row in total_event_details
        if category == "track_refer_score" and row is not None
    ]
    print(
        "missing_track_query_states=",
        {
            "object_le_0.6_total": len(track_object_events),
            "object_le_0.5_frozen": sum(
                not row["qim_state_updated"] for row in track_object_events
            ),
            "object_0.5_to_0.6_updated": sum(
                row["qim_state_updated"] for row in track_object_events
            ),
            "refer_le_0.4_total": len(track_refer_events),
            "refer_low_but_qim_updated": sum(
                row["qim_state_updated"] for row in track_refer_events
            ),
        },
    )
    associated = [row for _, row in total_deletions if row["gt_id"] is not None]
    print(f"all_track_deletions={len(total_deletions)} associated_deletions={len(associated)}")
    print(
        "associated_deletions=",
        {
            "visible_and_iou_ge_threshold": sum(row["current_iou"] >= args.iou for row in associated),
            "future_reappears": sum(row["future_reappears"] for row in associated),
            "recreated_by_detect": sum(row["recreated"] for row in associated),
            "immediate_missing_output": sum(row["immediate_missing_output"] for row in associated),
        },
    )
    print("new_id_recovery_details=")
    for expression, row in total_episodes:
        if row["recovers_with_new_id"]:
            print(expression, row)
    print("recent_deletion_episode_details=")
    for expression, row in total_episodes:
        if row["recent_deletion"] or row["deletion_during_episode"]:
            print(expression, row)
    if args.details:
        print("\n[MISSING EPISODE DETAILS]")
        for expression, row in total_episodes:
            print(expression, row)
    if args.details:
        print("\n[ASSOCIATED DELETION DETAILS]")
        for expression, row in total_deletions:
            if row["gt_id"] is not None:
                print(expression, row)


if __name__ == "__main__":
    main()
