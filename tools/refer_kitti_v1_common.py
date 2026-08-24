#!/usr/bin/env python3
"""Shared, strict helpers for the official Refer-KITTI V1 reproduction."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DATA_ROOT = Path("/home/curtis/datasets/refer-kitti")
DEFAULT_SEQMAP = DEFAULT_DATA_ROOT / "seqmap.txt"
DEFAULT_CHECKPOINT = Path("/home/curtis/checkpoints/TempRMOT/checkpoint_rk.pth")
DEFAULT_TEXT_ENCODER = Path("/home/curtis/checkpoints/roberta-base")
DEFAULT_OUTPUT_ROOT = Path("/home/curtis/experiments/TempRMOT/rk_official")
EXPECTED_PLAN_SHA256 = "16a1469510aead9ad64e5edf2b6accf872c317674d2ca154081806e723cca302"
EXPECTED_SEQMAP_SHA256 = "3b328143c066afe8c45615652636146b69ac2ac4df69fcaa5b2a1f889957e222"
EXPECTED_CHECKPOINT_SHA256 = "55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8"
EXPECTED_QUERY_COUNT = 150
EXPECTED_META_ARCH = "rmot_qim_v2"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_seqmap(path: Path = DEFAULT_SEQMAP, require_full: bool = True) -> list[str]:
    raw = path.read_bytes()
    if require_full and sha256_bytes(raw) != EXPECTED_SEQMAP_SHA256:
        raise RuntimeError(f"seqmap SHA-256 mismatch: {path}")
    entries = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    if require_full and len(entries) != EXPECTED_QUERY_COUNT:
        raise RuntimeError(f"expected {EXPECTED_QUERY_COUNT} seqmap entries, found {len(entries)}")
    if len(entries) != len(set(entries)):
        raise RuntimeError("seqmap contains duplicate queries")
    for entry in entries:
        video_id, separator, expression_id = entry.partition("+")
        if not separator or len(video_id) != 4 or not video_id.isdigit() or not expression_id:
            raise RuntimeError(f"invalid seqmap query: {entry!r}")
    return entries


def select_queries(entries: list[str], query_start: int, query_limit: int | None) -> list[str]:
    if query_start < 0 or query_start >= len(entries):
        raise ValueError(f"query-start must be in [0, {len(entries) - 1}]")
    if query_limit is not None and query_limit < 1:
        raise ValueError("query-limit must be at least 1")
    selected = entries[query_start:] if query_limit is None else entries[query_start:query_start + query_limit]
    if not selected:
        raise ValueError("query slice is empty")
    return selected


def query_parts(query_id: str) -> tuple[str, str]:
    video_id, expression_id = query_id.split("+", 1)
    return video_id, expression_id


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    return repr(value)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(encoded)


def load_checkpoint_container(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "args" not in checkpoint or "model" not in checkpoint:
        raise RuntimeError("checkpoint must contain top-level 'args' and 'model'")
    if not isinstance(checkpoint["args"], Namespace):
        raise RuntimeError(f"checkpoint args is not argparse.Namespace: {type(checkpoint['args'])!r}")
    if not isinstance(checkpoint["model"], dict):
        raise RuntimeError("checkpoint model is not a state-dict mapping")
    return checkpoint


def runtime_namespace(
    saved_args: Namespace,
    *,
    data_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    text_encoder_path: Path,
    device: str,
    gpu_id: int,
    object_threshold: float = 0.6,
    referring_threshold: float = 0.4,
    area_threshold: float = 100.0,
    max_frames: int | None = None,
    audit_candidates: bool = False,
    audit_dir: Path | None = None,
) -> Namespace:
    """Copy every saved field, overriding only runtime/IO controls.

    ``backbone_pretrained=False`` is an initialization-only compatibility field
    introduced in the local source after this checkpoint was saved. It changes
    neither module names nor shapes and prevents external weight downloads.
    """
    args = copy.deepcopy(saved_args)
    structural_expectations = {
        "meta_arch": EXPECTED_META_ARCH,
        "hist_len": 4,
        "extra_track_attn": False,
        "sampler_lengths": [4, 4, 4, 4],
    }
    for field, expected in structural_expectations.items():
        actual = getattr(args, field, None)
        if actual != expected:
            raise RuntimeError(f"checkpoint structural field {field}={actual!r}, expected {expected!r}")

    runtime_values = {
        "rmot_path": str(data_root),
        "data_txt_path_train": str(data_root / "refer-kitti.train"),
        "resume": str(checkpoint_path),
        "output_dir": str(output_root),
        "text_encoder_path": str(text_encoder_path),
        "device": device,
        "gpu": gpu_id,
        "gpu_id": gpu_id,
        "distributed": False,
        "rank": 0,
        "world_size": 1,
        "object_threshold": object_threshold,
        "referring_threshold": referring_threshold,
        "area_threshold": area_threshold,
        "max_frames": max_frames,
        "audit_candidates": audit_candidates,
        "audit_dir": None if audit_dir is None else str(audit_dir),
        "visualization": False,
        "num_inference_workers": 1,
        "video_ids": None,
        "max_expressions": None,
        "backbone_pretrained": False,
    }
    for field, value in runtime_values.items():
        setattr(args, field, value)
    return args


def build_checkpoint_model(args: Namespace):
    if args.meta_arch != EXPECTED_META_ARCH:
        raise RuntimeError(f"unsupported checkpoint meta_arch: {args.meta_arch!r}")
    from models.transrmot_pro import build

    return build(args)[0]


def normalized_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str | None]:
    keys = list(state_dict)
    prefix = "module."
    if keys and all(key.startswith(prefix) for key in keys):
        return {key[len(prefix):]: value for key, value in state_dict.items()}, prefix
    return state_dict, None


def persistent_buffer_names(model: torch.nn.Module, *, remove_duplicate: bool = False) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules(remove_duplicate=remove_duplicate):
        for buffer_name, buffer in module._buffers.items():
            if buffer is None or buffer_name in module._non_persistent_buffers_set:
                continue
            names.add(f"{module_name}.{buffer_name}" if module_name else buffer_name)
    return names


def restore_checkpoint_position_ids_persistence(
    model: torch.nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Restore only the older HF serialization semantics proven by the checkpoint.

    This does not create, replace, resize, cast, or otherwise modify a tensor. It
    only makes an already registered, byte-for-byte equal ``position_ids``
    buffer persistent again. Every guard is mandatory and ``token_type_ids`` is
    deliberately untouched.
    """
    key = "text_encoder.embeddings.position_ids"
    if key not in checkpoint_state:
        raise RuntimeError(f"serialization-compat guard failed: checkpoint lacks {key}")
    try:
        module = model.text_encoder.embeddings
    except AttributeError as exc:
        raise RuntimeError("serialization-compat guard failed: model lacks RobertaEmbeddings") from exc
    if "position_ids" not in module._buffers or module._buffers["position_ids"] is None:
        raise RuntimeError("serialization-compat guard failed: position_ids is not a registered buffer")
    if "position_ids" not in module._non_persistent_buffers_set:
        raise RuntimeError("serialization-compat guard failed: position_ids is not marked non-persistent")
    model_tensor = module._buffers["position_ids"]
    checkpoint_tensor = checkpoint_state[key]
    if tuple(model_tensor.shape) != tuple(checkpoint_tensor.shape):
        raise RuntimeError("serialization-compat guard failed: position_ids shape mismatch")
    if model_tensor.dtype != checkpoint_tensor.dtype:
        raise RuntimeError("serialization-compat guard failed: position_ids dtype mismatch")
    if not torch.equal(model_tensor, checkpoint_tensor):
        raise RuntimeError("serialization-compat guard failed: position_ids values differ")
    module._non_persistent_buffers_set.remove("position_ids")
    if "token_type_ids" in module._buffers and "token_type_ids" not in module._non_persistent_buffers_set:
        raise RuntimeError("serialization-compat guard failed: token_type_ids persistence changed")
    restored_state = model.state_dict()
    if key not in restored_state or not torch.equal(restored_state[key], checkpoint_tensor):
        raise RuntimeError("serialization-compat guard failed: restored state is not exact")
    return {
        "key": key,
        "shape": list(model_tensor.shape),
        "dtype": str(model_tensor.dtype),
        "elementwise_equal": True,
        "position_ids_persistent": True,
        "token_type_ids_unchanged_non_persistent": "token_type_ids" in module._non_persistent_buffers_set,
    }


def state_dict_diagnostics(model: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    normalized, stripped_prefix = normalized_state_dict(checkpoint_state)
    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(normalized)
    shape_mismatches = [
        {
            "key": key,
            "model_shape": list(model_state[key].shape),
            "checkpoint_shape": list(normalized[key].shape),
        }
        for key in sorted(model_keys & checkpoint_keys)
        if tuple(model_state[key].shape) != tuple(normalized[key].shape)
    ]
    deduplicated_parameter_names = set(dict(model.named_parameters()))
    parameter_names = set(dict(model.named_parameters(remove_duplicate=False)))
    buffer_names = persistent_buffer_names(model)
    return {
        "stripped_prefix": stripped_prefix,
        "model_state_count": len(model_state),
        "checkpoint_state_count": len(normalized),
        "learned_parameter_count": len(parameter_names),
        "deduplicated_learned_parameter_count": len(deduplicated_parameter_names),
        "parameter_alias_state_paths": sorted(parameter_names - deduplicated_parameter_names),
        "persistent_buffer_count": len(buffer_names),
        "state_covers_parameters_and_persistent_buffers": model_keys == parameter_names | buffer_names,
        "missing_keys": sorted(model_keys - checkpoint_keys),
        "unexpected_keys": sorted(checkpoint_keys - model_keys),
        "shape_mismatches": shape_mismatches,
        "normalized_state": normalized,
    }


def iter_tensors(value: Any, path: str = "checkpoint") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_tensors(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_tensors(item, f"{path}[{index}]")


def verify_runtime_snapshot(snapshot_dir: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "freeze_refer_kitti_v1_runtime.py"),
        "--verify",
        "--snapshot-dir",
        str(snapshot_dir),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
