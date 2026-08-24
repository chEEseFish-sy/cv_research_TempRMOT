#!/usr/bin/env python3
"""Validate Refer-KITTI V1 prediction/GT files before TrackEval."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from refer_kitti_v1_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SEQMAP,
    atomic_write_json,
    query_parts,
    read_seqmap,
)


def image_info(images_root: Path, video_id: str) -> tuple[int, int, int]:
    images = sorted((images_root / video_id).glob("*.png")) + sorted((images_root / video_id).glob("*.jpg"))
    if not images:
        raise RuntimeError(f"no images found for video {video_id}: {images_root / video_id}")
    with Image.open(images[0]) as image:
        width, height = image.size
    return len(images), width, height


def validate_mot_file(
    path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    require_nonempty: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing file: {path}")
    rows = 0
    seen_frame_ids: set[tuple[int, int]] = set()
    out_of_bounds_count = 0
    out_of_bounds_samples = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) < 6:
                raise RuntimeError(f"bad line with fewer than 6 columns: {path}:{line_number}")
            try:
                frame_value = float(row[0])
                track_value = float(row[1])
                values = [float(field) for field in row]
            except ValueError as exc:
                raise RuntimeError(f"non-numeric value: {path}:{line_number}") from exc
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"NaN/Inf value: {path}:{line_number}")
            if not frame_value.is_integer() or not track_value.is_integer():
                raise RuntimeError(f"non-integral frame or track ID: {path}:{line_number}")
            frame_id, track_id = int(frame_value), int(track_value)
            if frame_id < 1 or frame_id > frame_count:
                raise RuntimeError(f"frame ID outside [1,{frame_count}]: {path}:{line_number}")
            if track_id < 0:
                raise RuntimeError(f"negative track ID: {path}:{line_number}")
            frame_track = (frame_id, track_id)
            if frame_track in seen_frame_ids:
                raise RuntimeError(f"duplicate frame/track ID: {path}:{line_number}: {frame_track}")
            seen_frame_ids.add(frame_track)
            x, y, box_width, box_height = values[2:6]
            if box_width <= 0 or box_height <= 0:
                raise RuntimeError(f"illegal box: {path}:{line_number}: {(x, y, box_width, box_height)}")
            tolerance = 1e-3
            if x < -tolerance or y < -tolerance or x + box_width > width + tolerance or y + box_height > height + tolerance:
                out_of_bounds_count += 1
                if len(out_of_bounds_samples) < 10:
                    out_of_bounds_samples.append({
                        "line_number": line_number,
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "box_xywh": [x, y, box_width, box_height],
                    })
            rows += 1
    if require_nonempty and rows == 0:
        raise RuntimeError(f"required nonempty file is empty: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "out_of_bounds_count": out_of_bounds_count,
        "out_of_bounds_samples": out_of_bounds_samples,
    }


def validate_results(
    results_root: Path,
    seqmap: Path,
    images_root: Path,
    *,
    require_nonempty_prediction: bool,
    require_exact_query_dirs: bool = True,
) -> dict[str, Any]:
    queries = read_seqmap(seqmap)
    expected_pairs = {query_parts(query) for query in queries}
    observed_pairs = {
        (video_dir.name, expression_dir.name)
        for video_dir in results_root.iterdir()
        if video_dir.is_dir()
        for expression_dir in video_dir.iterdir()
        if expression_dir.is_dir()
    } if results_root.is_dir() else set()
    if require_exact_query_dirs and observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise RuntimeError(f"query directory identity mismatch: missing={missing[:10]}, extra={extra[:10]}")

    video_info: dict[str, tuple[int, int, int]] = {}
    query_reports = []
    total_prediction_rows = 0
    total_gt_rows = 0
    total_out_of_bounds = 0
    out_of_bounds_samples = []
    for query in queries:
        video_id, expression_id = query_parts(query)
        if video_id not in video_info:
            video_info[video_id] = image_info(images_root, video_id)
        frame_count, width, height = video_info[video_id]
        query_dir = results_root / video_id / expression_id
        prediction = validate_mot_file(
            query_dir / "predict.txt",
            frame_count=frame_count,
            width=width,
            height=height,
            require_nonempty=require_nonempty_prediction,
        )
        ground_truth = validate_mot_file(
            query_dir / "gt.txt",
            frame_count=frame_count,
            width=width,
            height=height,
            require_nonempty=True,
        )
        total_prediction_rows += prediction["rows"]
        total_gt_rows += ground_truth["rows"]
        total_out_of_bounds += prediction["out_of_bounds_count"] + ground_truth["out_of_bounds_count"]
        for source, file_report in (("predict.txt", prediction), ("gt.txt", ground_truth)):
            for sample in file_report["out_of_bounds_samples"]:
                if len(out_of_bounds_samples) < 20:
                    out_of_bounds_samples.append({"query_id": query, "source": source, **sample})
        query_reports.append({
            "query_id": query,
            "frame_count": frame_count,
            "image_width": width,
            "image_height": height,
            "prediction": prediction,
            "ground_truth": ground_truth,
        })
    return {
        "ok": True,
        "results_root": str(results_root),
        "seqmap": str(seqmap),
        "query_count": len(queries),
        "predict_file_count": len(query_reports),
        "gt_file_count": len(query_reports),
        "prediction_rows": total_prediction_rows,
        "gt_rows": total_gt_rows,
        "out_of_bounds_count": total_out_of_bounds,
        "out_of_bounds_samples": out_of_bounds_samples,
        "require_nonempty_prediction": require_nonempty_prediction,
        "queries": query_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seqmap", type=Path, default=DEFAULT_SEQMAP)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_DATA_ROOT / "KITTI/training/image_02")
    parser.add_argument("--require-nonempty-prediction", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_results(
        args.results_root,
        args.seqmap,
        args.images_root,
        require_nonempty_prediction=args.require_nonempty_prediction,
    )
    if args.report:
        atomic_write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "queries"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
