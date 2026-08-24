#!/usr/bin/env python3
"""Seqmap-driven, resumable official Refer-KITTI V1 inference launcher."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import time
from pathlib import Path

import torch

from refer_kitti_v1_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEQMAP,
    DEFAULT_TEXT_ENCODER,
    atomic_write_json,
    build_checkpoint_model,
    canonical_hash,
    load_checkpoint_container,
    query_parts,
    read_seqmap,
    runtime_namespace,
    restore_checkpoint_position_ids_persistence,
    select_queries,
    sha256_file,
    state_dict_diagnostics,
    verify_runtime_snapshot,
)
from validate_refer_kitti_v1_results import image_info, validate_mot_file


def completion_payload(query_dir: Path, query_id: str, checkpoint_hash: str, config_hash: str, frame_count: int) -> dict:
    files = {}
    for filename in ("predict.txt", "gt.txt"):
        path = query_dir / filename
        files[filename] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "query_id": query_id,
        "frame_count": frame_count,
        "output_size_bytes": sum(item["bytes"] for item in files.values()),
        "files": files,
        "exit_status": 0,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def valid_completion(query_dir: Path, expected: dict) -> bool:
    marker = query_dir / ".complete.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        for key in ("checkpoint_sha256", "config_sha256", "query_id", "frame_count", "exit_status"):
            if payload.get(key) != expected[key]:
                return False
        total = 0
        for filename in ("predict.txt", "gt.txt"):
            path = query_dir / filename
            record = payload["files"][filename]
            if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                return False
            total += record["bytes"]
        return total == payload.get("output_size_bytes")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seqmap", type=Path, default=DEFAULT_SEQMAP)
    parser.add_argument("--text-encoder-path", type=Path, default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--object-threshold", type=float, default=0.6)
    parser.add_argument("--referring-threshold", type=float, default=0.4)
    parser.add_argument("--area-threshold", type=float, default=100.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--require-nonempty-prediction", action="store_true")
    parser.add_argument("--audit-candidates", action="store_true")
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    selected = select_queries(read_seqmap(args.seqmap), args.query_start, args.query_limit)
    if args.list_only:
        print("\n".join(selected))
        return 0
    if args.snapshot_dir is None:
        parser.error("--snapshot-dir is required unless --list-only is used")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for inference")
    verify_runtime_snapshot(args.snapshot_dir)

    checkpoint_hash = sha256_file(args.checkpoint)
    checkpoint = load_checkpoint_container(args.checkpoint)
    runtime_args = runtime_namespace(
        checkpoint["args"],
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        text_encoder_path=args.text_encoder_path,
        device="cuda",
        gpu_id=args.gpu_id,
        object_threshold=args.object_threshold,
        referring_threshold=args.referring_threshold,
        area_threshold=args.area_threshold,
        max_frames=args.max_frames,
        audit_candidates=args.audit_candidates,
        audit_dir=args.audit_dir,
    )
    config_hash = canonical_hash({
        "checkpoint_args": vars(checkpoint["args"]),
        "runtime": {
            "data_root": args.data_root,
            "text_encoder_path": args.text_encoder_path,
            "object_threshold": args.object_threshold,
            "referring_threshold": args.referring_threshold,
            "area_threshold": args.area_threshold,
            "max_frames": args.max_frames,
            "backbone_pretrained": False,
        },
    })

    torch.cuda.set_device(args.gpu_id)
    model = build_checkpoint_model(runtime_args)
    restore_checkpoint_position_ids_persistence(model, checkpoint["model"])
    diagnostics = state_dict_diagnostics(model, checkpoint["model"])
    state = diagnostics.pop("normalized_state")
    if diagnostics["missing_keys"] or diagnostics["unexpected_keys"] or diagnostics["shape_mismatches"]:
        raise RuntimeError(f"checkpoint state is not exact: {diagnostics}")
    model.load_state_dict(state, strict=True)
    model.eval().cuda()

    from inference import Detector

    results_root = args.output_root / f"results_{args.checkpoint.stem}"
    invalid_root = args.output_root / "invalid_partial"
    summary = {"selected": selected, "skipped": [], "completed": [], "started_at": time.time()}
    for query_id in selected:
        verify_runtime_snapshot(args.snapshot_dir)
        video_id, expression_id = query_parts(query_id)
        frame_count, width, height = image_info(args.data_root / "KITTI/training/image_02", video_id)
        if args.max_frames is not None:
            frame_count = min(frame_count, args.max_frames)
        query_dir = results_root / video_id / expression_id
        expected = {
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": config_hash,
            "query_id": query_id,
            "frame_count": frame_count,
            "exit_status": 0,
        }
        if query_dir.exists() and valid_completion(query_dir, expected):
            summary["skipped"].append(query_id)
            continue
        if query_dir.exists():
            invalid_root.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            destination = invalid_root / f"{stamp}__{video_id}__{expression_id}"
            shutil.move(str(query_dir), destination)
        detector = Detector(runtime_args, args.checkpoint.stem, model=model, seq_num=[video_id, expression_id + ".json"])
        detector.detect()
        validate_mot_file(query_dir / "predict.txt", frame_count=frame_count, width=width, height=height,
                          require_nonempty=args.require_nonempty_prediction)
        validate_mot_file(query_dir / "gt.txt", frame_count=frame_count, width=width, height=height,
                          require_nonempty=True)
        atomic_write_json(query_dir / ".complete.json",
                          completion_payload(query_dir, query_id, checkpoint_hash, config_hash, frame_count))
        summary["completed"].append(query_id)

    summary["elapsed_seconds"] = time.time() - summary["started_at"]
    atomic_write_json(args.output_root / "v1_inference_summary.json", summary)
    verify_runtime_snapshot(args.snapshot_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
