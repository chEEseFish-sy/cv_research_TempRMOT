#!/usr/bin/env python3
"""Strict structural and numerical preflight for checkpoint_rk.pth."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from refer_kitti_v1_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TEXT_ENCODER,
    atomic_write_json,
    build_checkpoint_model,
    iter_tensors,
    load_checkpoint_container,
    runtime_namespace,
    restore_checkpoint_position_ids_persistence,
    sha256_file,
    state_dict_diagnostics,
    verify_runtime_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--text-encoder-path", type=Path, default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT_ROOT / "phase2_checkpoint_preflight.json")
    args = parser.parse_args()

    started = time.monotonic()
    verify_runtime_snapshot(args.snapshot_dir)
    checkpoint_hash = sha256_file(args.checkpoint)
    checkpoint = load_checkpoint_container(args.checkpoint)
    runtime_args = runtime_namespace(
        checkpoint["args"],
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        output_root=DEFAULT_OUTPUT_ROOT,
        text_encoder_path=args.text_encoder_path,
        device="cpu",
        gpu_id=0,
    )
    model = build_checkpoint_model(runtime_args)
    serialization_compat = restore_checkpoint_position_ids_persistence(model, checkpoint["model"])
    diagnostics = state_dict_diagnostics(model, checkpoint["model"])
    normalized_state = diagnostics.pop("normalized_state")

    structural_ok = (
        diagnostics["state_covers_parameters_and_persistent_buffers"]
        and not diagnostics["missing_keys"]
        and not diagnostics["unexpected_keys"]
        and not diagnostics["shape_mismatches"]
    )
    load_missing: list[str] = []
    load_unexpected: list[str] = []
    strict_load_succeeded = False
    if structural_ok:
        incompatible = model.load_state_dict(normalized_state, strict=False)
        load_missing = list(incompatible.missing_keys)
        load_unexpected = list(incompatible.unexpected_keys)
        structural_ok = not load_missing and not load_unexpected
        if structural_ok:
            model.load_state_dict(normalized_state, strict=True)
            strict_load_succeeded = True

    tensor_count = 0
    tensor_numel = 0
    nonfinite_count = 0
    nonfinite_paths = []
    with torch.no_grad():
        for tensor_path, tensor in iter_tensors(checkpoint):
            tensor_count += 1
            tensor_numel += tensor.numel()
            if tensor.is_floating_point() or tensor.is_complex():
                count = int((~torch.isfinite(tensor)).sum().item())
                if count:
                    nonfinite_count += count
                    nonfinite_paths.append({"path": tensor_path, "count": count})

    report = {
        "ok": structural_ok and nonfinite_count == 0,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "saved_args_field_count": len(vars(checkpoint["args"])),
        "saved_structural_args": {
            "meta_arch": checkpoint["args"].meta_arch,
            "hist_len": checkpoint["args"].hist_len,
            "extra_track_attn": checkpoint["args"].extra_track_attn,
            "sampler_lengths": checkpoint["args"].sampler_lengths,
        },
        "runtime_compatibility_fields": {"backbone_pretrained": False},
        "serialization_compatibility": serialization_compat,
        **diagnostics,
        "load_state_dict_missing_keys": load_missing,
        "load_state_dict_unexpected_keys": load_unexpected,
        "strict_load_succeeded": strict_load_succeeded,
        "checkpoint_tensor_count": tensor_count,
        "checkpoint_tensor_numel": tensor_numel,
        "nonfinite_count": nonfinite_count,
        "nonfinite_paths": nonfinite_paths,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_json(args.report, report)
    verify_runtime_snapshot(args.snapshot_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise RuntimeError("strict checkpoint preflight failed; see report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
