#!/usr/bin/env python3
"""Extract only the authoritative 150-query oracle pairs from the official ZIP."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from refer_kitti_v1_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEQMAP,
    atomic_write_json,
    query_parts,
    read_seqmap,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_DATA_ROOT / "official_archives/results_epoch99.zip")
    parser.add_argument("--seqmap", type=Path, default=DEFAULT_SEQMAP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "oracle_rmot_official")
    args = parser.parse_args()

    queries = read_seqmap(args.seqmap)
    if args.output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing oracle root: {args.output_root}")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=args.output_root.name + ".tmp-", dir=args.output_root.parent))
    manifest = []
    try:
        with zipfile.ZipFile(args.archive) as archive:
            members = set(archive.namelist())
            for query in queries:
                video_id, expression_id = query_parts(query)
                query_dir = temp_root / video_id / expression_id
                query_dir.mkdir(parents=True, exist_ok=False)
                record = {"query_id": query, "files": {}}
                for filename in ("predict.txt", "gt.txt"):
                    member = f"results_epoch99/{video_id}/{expression_id}/{filename}"
                    if member not in members:
                        raise RuntimeError(f"official archive missing member: {member}")
                    data = archive.read(member)
                    destination = query_dir / filename
                    destination.write_bytes(data)
                    record["files"][filename] = {
                        "archive_member": member,
                        "bytes": len(data),
                        "sha256": sha256_file(destination),
                    }
                manifest.append(record)
        atomic_write_json(temp_root / "oracle_selection_manifest.json", {
            "archive": str(args.archive),
            "archive_sha256": sha256_file(args.archive),
            "seqmap": str(args.seqmap),
            "query_count": len(queries),
            "selected_file_count": len(queries) * 2,
            "queries": manifest,
        })
        os.replace(temp_root, args.output_root)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    print(json.dumps({"ok": True, "output_root": str(args.output_root), "query_count": len(queries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
