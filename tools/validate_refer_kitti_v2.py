#!/usr/bin/env python3
"""Validate the on-disk Refer-KITTI-V2 data used by TempRMOT."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_SEQUENCES = {f"{index:04d}" for index in range(21)}
TRAIN_SEQUENCES = {
    "0000", "0001", "0002", "0003", "0004", "0006", "0007", "0008",
    "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018",
    "0020",
}
TEST_SEQUENCES = {"0005", "0011", "0013", "0019"}
EXPECTED_TRAIN_IMAGES = 5172
EXPECTED_TEST_QUERIES = 862
EXPECTED_LABEL_FILES = 7162
EXPECTED_EXPRESSION_FILES = 9758


class Report:
    def __init__(self, detail_limit: int) -> None:
        self.counts: Counter[str] = Counter()
        self.details: dict[str, list[str]] = defaultdict(list)
        self.warning_counts: Counter[str] = Counter()
        self.warning_details: dict[str, list[str]] = defaultdict(list)
        self.detail_limit = detail_limit

    def error(self, category: str, message: str) -> None:
        self.counts[category] += 1
        if len(self.details[category]) < self.detail_limit:
            self.details[category].append(message)

    def warning(self, category: str, message: str) -> None:
        self.warning_counts[category] += 1
        if len(self.warning_details[category]) < self.detail_limit:
            self.warning_details[category].append(message)

    @property
    def total_errors(self) -> int:
        return sum(self.counts.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/curtis/datasets/refer-kitti-v2"),
        help="Refer-KITTI-V2 root directory",
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=Path("datasets/data_path/refer-kitti-v2.train"),
    )
    parser.add_argument(
        "--seqmap",
        type=Path,
        default=Path("datasets/data_path/seqmap.txt"),
    )
    parser.add_argument("--detail-limit", type=int, default=50)
    return parser.parse_args()


def numeric_stems(paths: list[Path], report: Report, category: str) -> list[int]:
    stems: list[int] = []
    for path in paths:
        try:
            stems.append(int(path.stem))
        except ValueError:
            report.error(category, f"non-numeric frame name: {path}")
    return sorted(stems)


def check_sequence_directories(root: Path, report: Report) -> tuple[int, int]:
    image_root = root / "KITTI/training/image_02"
    label_root = root / "KITTI/labels_with_ids/image_02"
    image_sequences = {p.name for p in image_root.iterdir() if p.is_dir()} if image_root.is_dir() else set()
    label_sequences = {p.name for p in label_root.iterdir() if p.is_dir()} if label_root.is_dir() else set()

    for sequence in sorted(EXPECTED_SEQUENCES - image_sequences):
        report.error("missing_image_sequence", sequence)
    for sequence in sorted(image_sequences - EXPECTED_SEQUENCES):
        report.error("unexpected_image_sequence", sequence)
    for sequence in sorted(EXPECTED_SEQUENCES - label_sequences):
        report.error("missing_label_sequence", sequence)

    image_count = 0
    label_count = 0
    for sequence in sorted(EXPECTED_SEQUENCES & image_sequences):
        images = sorted((image_root / sequence).glob("*.png"))
        image_count += len(images)
        frames = numeric_stems(images, report, "bad_image_frame_name")
        if frames and frames != list(range(frames[0], frames[-1] + 1)):
            report.error("noncontiguous_image_frames", sequence)

    for sequence in sorted(EXPECTED_SEQUENCES & label_sequences):
        label_count += len(list((label_root / sequence).glob("*.txt")))

    return image_count, label_count


def parse_label_ids(path: Path, report: Report) -> set[int]:
    ids: set[int] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        report.error("unreadable_label", f"{path}: {exc}")
        return ids

    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 6:
            report.error("bad_label_columns", f"{path}:{line_number}: {len(fields)} columns")
            continue
        try:
            values = [float(field) for field in fields]
        except ValueError:
            report.error("nonnumeric_label", f"{path}:{line_number}")
            continue
        if not all(math.isfinite(value) for value in values):
            report.error("nonfinite_label", f"{path}:{line_number}")
            continue
        ids.add(int(values[1]))
    return ids


def check_train_list(root: Path, train_list: Path, report: Report) -> tuple[int, set[str]]:
    try:
        entries = [line.strip() for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        report.error("unreadable_train_list", f"{train_list}: {exc}")
        return 0, set()

    if len(entries) != EXPECTED_TRAIN_IMAGES:
        report.error("train_list_count", f"expected {EXPECTED_TRAIN_IMAGES}, found {len(entries)}")

    videos: set[str] = set()
    for entry in entries:
        relative = Path(entry)
        parts = relative.parts
        if len(parts) < 5:
            report.error("bad_train_path", entry)
            continue
        video = parts[-2]
        videos.add(video)
        image_path = root / relative
        if not image_path.is_file():
            report.error("missing_train_image", entry)
        label_path = root / "KITTI/labels_with_ids/image_02" / video / f"{relative.stem}.txt"
        if not label_path.is_file():
            report.error("missing_train_label", str(label_path))

    if videos != TRAIN_SEQUENCES:
        report.error("train_split", f"expected {sorted(TRAIN_SEQUENCES)}, found {sorted(videos)}")
    if videos & TEST_SEQUENCES:
        report.error("test_video_in_train", str(sorted(videos & TEST_SEQUENCES)))
    return len(entries), videos


def check_labels(root: Path, report: Report) -> tuple[int, dict[Path, set[int]]]:
    label_root = root / "KITTI/labels_with_ids/image_02"
    paths = sorted(label_root.glob("*/*.txt")) if label_root.is_dir() else []
    if len(paths) != EXPECTED_LABEL_FILES:
        report.error("label_file_count", f"expected {EXPECTED_LABEL_FILES}, found {len(paths)}")
    cache = {path: parse_label_ids(path, report) for path in paths}
    return len(paths), cache


def check_expressions(
    root: Path, label_cache: dict[Path, set[int]], report: Report
) -> tuple[int, set[str]]:
    expression_root = root / "expression"
    paths = sorted(expression_root.glob("*/*.json")) if expression_root.is_dir() else []
    if len(paths) != EXPECTED_EXPRESSION_FILES:
        report.error(
            "expression_file_count",
            f"expected {EXPECTED_EXPRESSION_FILES}, found {len(paths)}",
        )
    video_label_ids: dict[str, set[int]] = defaultdict(set)
    for label_path, object_ids in label_cache.items():
        video_label_ids[label_path.parent.name].update(object_ids)
    videos: set[str] = set()
    for path in paths:
        videos.add(path.parent.name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            report.error("bad_json", f"{path}: {exc}")
            continue

        missing_keys = {"label", "video_name", "sentence"} - payload.keys()
        if missing_keys:
            report.error("missing_json_keys", f"{path}: {sorted(missing_keys)}")
            continue
        if not isinstance(payload["label"], dict):
            report.error("bad_json_label", str(path))
            continue
        raw_video_name = str(payload["video_name"])
        if raw_video_name.upper().startswith("KITTI_"):
            raw_video_name = raw_video_name.split("_", 1)[1]
        try:
            video_name = f"{int(raw_video_name):04d}"
        except ValueError:
            video_name = raw_video_name
        if video_name != path.parent.name:
            report.error("json_video_mismatch", f"{path}: video_name={payload['video_name']!r}")

        for frame, object_ids in payload["label"].items():
            try:
                frame_name = f"{int(frame):06d}.txt"
            except (TypeError, ValueError):
                report.error("bad_json_frame", f"{path}: {frame!r}")
                continue
            label_path = root / "KITTI/labels_with_ids/image_02" / path.parent.name / frame_name
            if not label_path.is_file():
                report.error("missing_referenced_label", f"{path}: {frame_name}")
                continue
            label_ids = label_cache.get(label_path)
            if label_ids is None:
                label_ids = parse_label_ids(label_path, report)
                label_cache[label_path] = label_ids
            if not isinstance(object_ids, list):
                report.error("bad_json_object_ids", f"{path}: frame {frame}")
                continue
            for object_id in object_ids:
                try:
                    normalized_id = int(object_id)
                except (TypeError, ValueError):
                    report.error("bad_json_object_id", f"{path}: {object_id!r}")
                    continue
                if normalized_id not in label_ids:
                    message = f"{path}: frame {frame}, object {normalized_id}"
                    if normalized_id not in video_label_ids[path.parent.name]:
                        report.error("unknown_object_id", message)
                    else:
                        report.warning("reference_without_box", message)
    if videos != EXPECTED_SEQUENCES:
        report.error(
            "expression_video_set",
            f"expected {sorted(EXPECTED_SEQUENCES)}, found {sorted(videos)}",
        )
    return len(paths), videos


def check_seqmap(root: Path, seqmap: Path, report: Report) -> tuple[int, set[str]]:
    try:
        entries = [line.strip() for line in seqmap.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        report.error("unreadable_seqmap", f"{seqmap}: {exc}")
        return 0, set()

    if len(entries) != EXPECTED_TEST_QUERIES:
        report.error("seqmap_count", f"expected {EXPECTED_TEST_QUERIES}, found {len(entries)}")
    if len(entries) != len(set(entries)):
        report.error("duplicate_seqmap_entry", f"{len(entries) - len(set(entries))} duplicates")

    videos: set[str] = set()
    for entry in entries:
        if "+" not in entry:
            report.error("bad_seqmap_entry", entry)
            continue
        video, expression_stem = entry.split("+", 1)
        videos.add(video)
        expression_path = root / "expression" / video / f"{expression_stem}.json"
        if not expression_path.is_file():
            report.error("missing_seqmap_expression", str(expression_path))

    if videos != TEST_SEQUENCES:
        report.error("test_split", f"expected {sorted(TEST_SEQUENCES)}, found {sorted(videos)}")
    if videos & TRAIN_SEQUENCES:
        report.error("train_video_in_seqmap", str(sorted(videos & TRAIN_SEQUENCES)))
    return len(entries), videos


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    report = Report(args.detail_limit)

    image_count, directory_label_count = check_sequence_directories(root, report)
    train_count, train_videos = check_train_list(root, args.train_list, report)
    label_count, label_cache = check_labels(root, report)
    expression_count, expression_videos = check_expressions(root, label_cache, report)
    seqmap_count, test_videos = check_seqmap(root, args.seqmap, report)

    print(f"root: {root}")
    print(f"image_sequences: {len(EXPECTED_SEQUENCES)} expected")
    print(f"images: {image_count}")
    print(f"labels: {label_count} (directory scan: {directory_label_count})")
    print(f"expressions: {expression_count} across {len(expression_videos)} videos")
    print(f"train_list: {train_count} images across {len(train_videos)} videos")
    print(f"seqmap: {seqmap_count} queries across {len(test_videos)} videos")
    print(f"errors: {report.total_errors}")
    for category, count in sorted(report.counts.items()):
        print(f"  {category}: {count}")
    if report.details:
        print("error_details:")
        for category in sorted(report.details):
            for detail in report.details[category]:
                print(f"  [{category}] {detail}")
    print(f"warnings: {sum(report.warning_counts.values())}")
    for category, count in sorted(report.warning_counts.items()):
        print(f"  {category}: {count}")
    if report.warning_details:
        print("warning_details:")
        for category in sorted(report.warning_details):
            for detail in report.warning_details[category]:
                print(f"  [{category}] {detail}")

    if report.total_errors:
        print("G validation: FAILED")
        return 1
    print("G validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
