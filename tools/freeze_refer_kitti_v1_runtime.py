#!/usr/bin/env python3
"""Freeze or verify the definitive Refer-KITTI V1 execution identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from refer_kitti_v1_common import PROJECT_ROOT, atomic_write_json, sha256_bytes, sha256_file


SOURCE_SUFFIXES = {".py", ".sh", ".so", ".toml", ".yaml", ".yml"}
EXCLUDED_PARTS = {"__pycache__", ".git", ".handoffs", "outputs"}


def command(args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(args, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=check)
    return result.stdout


def manifest(paths: list[Path]) -> tuple[str, str, int, int]:
    lines = []
    total_bytes = 0
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        size = path.stat().st_size
        total_bytes += size
        lines.append(f"{sha256_file(path)}\t{size}\t{relative}\n")
    text = "".join(lines)
    return text, sha256_bytes(text.encode("utf-8")), len(lines), total_bytes


def untracked_sources() -> list[Path]:
    output = command(["git", "ls-files", "--others", "--exclude-standard", "-z"])
    paths = []
    for relative in output.split("\0"):
        if not relative:
            continue
        path = PROJECT_ROOT / relative
        if (path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
                and not any(part in EXCLUDED_PARTS for part in path.parts)
                and ".planning" not in path.parts):
            paths.append(path)
    return paths


def trackeval_sources() -> list[Path]:
    root = PROJECT_ROOT / "TrackEval"
    return [path for path in root.rglob("*") if path.is_file()
            and "__pycache__" not in path.parts and path.suffix != ".pyc"]


def msda_identity() -> dict[str, Any]:
    candidates = sorted((PROJECT_ROOT / "models/ops/build").rglob("MultiScaleDeformableAttention*.so"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one MSDA binary, found {len(candidates)}")
    path = candidates[0]
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def capture() -> tuple[dict[str, Any], dict[str, str]]:
    tracked_diff = subprocess.run(["git", "diff", "--binary", "HEAD", "--"], cwd=PROJECT_ROOT,
                                  stdout=subprocess.PIPE, check=True).stdout
    untracked_text, untracked_hash, untracked_count, untracked_bytes = manifest(untracked_sources())
    trackeval_text, trackeval_hash, trackeval_count, trackeval_bytes = manifest(trackeval_sources())
    pip_freeze = command([sys.executable, "-m", "pip", "freeze", "--all"])
    try:
        nvidia_smi = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                              "--format=csv,noheader"], check=False).strip()
    except FileNotFoundError:
        nvidia_smi = "nvidia-smi unavailable"
    import torch
    import torchvision
    import transformers

    identity = {
        "schema_version": 1,
        "git_head": command(["git", "rev-parse", "HEAD"]).strip(),
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "tracked_diff_bytes": len(tracked_diff),
        "untracked_executable_manifest_sha256": untracked_hash,
        "untracked_executable_count": untracked_count,
        "untracked_executable_bytes": untracked_bytes,
        "trackeval_tree_sha256": trackeval_hash,
        "trackeval_file_count": trackeval_count,
        "trackeval_bytes": trackeval_bytes,
        "pip_freeze_sha256": sha256_bytes(pip_freeze.encode("utf-8")),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_smi": nvidia_smi,
        "msda": msda_identity(),
    }
    artifacts = {
        "git_tracked_dirty.diff": tracked_diff.decode("utf-8", errors="surrogateescape"),
        "untracked_executable_manifest.tsv": untracked_text,
        "trackeval_tree_manifest.tsv": trackeval_text,
        "pip_freeze.txt": pip_freeze,
    }
    return identity, artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    identity, artifacts = capture()
    identity_path = args.snapshot_dir / "identity.json"
    if args.verify:
        expected = json.loads(identity_path.read_text(encoding="utf-8"))
        mismatches = {key: {"expected": expected.get(key), "actual": identity.get(key)}
                      for key in identity if expected.get(key) != identity.get(key)}
        if mismatches:
            raise RuntimeError(f"definitive runtime snapshot mismatch: {json.dumps(mismatches, sort_keys=True)}")
        print(json.dumps({"ok": True, "snapshot_dir": str(args.snapshot_dir)}, sort_keys=True))
        return 0
    if args.snapshot_dir.exists():
        raise RuntimeError(f"refusing to overwrite snapshot directory: {args.snapshot_dir}")
    args.snapshot_dir.mkdir(parents=True)
    for filename, content in artifacts.items():
        (args.snapshot_dir / filename).write_text(content, encoding="utf-8", errors="surrogateescape", newline="\n")
    atomic_write_json(identity_path, identity)
    print(json.dumps({"ok": True, "snapshot_dir": str(args.snapshot_dir), "identity_sha256": sha256_file(identity_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
