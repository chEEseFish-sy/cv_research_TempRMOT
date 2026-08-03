#!/usr/bin/env python3
"""Create and verify deterministic, phase-scoped research-plan approvals."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


SCHEMA_VERSION = 1
RESOURCE_KEY = re.compile(r"^[a-z][a-z0-9_.-]*$")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
REVIEW_DECISION = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?Decision(?:\*\*)?\s*:\s*(GO|REVISE|STOP)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
REVIEW_PLAN_HASH = re.compile(
    r"^\s*(?:\*\*)?Plan-SHA256(?:\*\*)?\s*:\s*`?([0-9a-fA-F]{64})`?\s*$",
    re.MULTILINE,
)


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_resources(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise GateError(f"Resource must use key=value: {value}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not RESOURCE_KEY.fullmatch(key):
            raise GateError(f"Invalid resource key: {key}")
        try:
            number = float(raw)
        except ValueError as exc:
            raise GateError(f"Resource ceiling must be numeric: {value}") from exc
        if not math.isfinite(number) or number < 0:
            raise GateError(f"Resource value must be finite and non-negative: {value}")
        if key in parsed:
            raise GateError(f"Duplicate resource key: {key}")
        parsed[key] = number
    return parsed


def resolve_plan(plan_dir: Path) -> tuple[Path, Path]:
    directory = plan_dir.resolve()
    plan = directory / "task_plan.md"
    if not directory.is_dir():
        raise GateError(f"Plan directory does not exist: {directory}")
    if not plan.is_file():
        raise GateError(f"task_plan.md does not exist: {plan}")
    return directory, plan


def scope_is_declared(plan: Path, scope: str) -> bool:
    candidate = scope.casefold()
    for raw_heading in HEADING.findall(plan.read_text(encoding="utf-8")):
        heading = raw_heading.strip().casefold()
        if heading == candidate:
            return True
        if heading.startswith(candidate):
            remainder = heading[len(candidate) :]
            if remainder.startswith((" ", "—", "–", "-", ":", "：")):
                return True
    return False


def verify_go_review(directory: Path, plan_hash: str) -> Path:
    review = directory / "feasibility_review.md"
    try:
        content = review.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise GateError(f"Feasibility review is missing: {review}") from exc
    decision = REVIEW_DECISION.search(content)
    if not decision:
        raise GateError("Feasibility review must contain a Decision: GO/REVISE/STOP line")
    if decision.group(1).upper() != "GO":
        raise GateError(f"Feasibility decision is not GO: {decision.group(1).upper()}")
    reviewed_hash = REVIEW_PLAN_HASH.search(content)
    if not reviewed_hash:
        raise GateError("Feasibility review must contain Plan-SHA256")
    if reviewed_hash.group(1).lower() != plan_hash:
        raise GateError("Feasibility review is stale for the current plan hash")
    return review


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def approve(args: argparse.Namespace) -> dict[str, Any]:
    directory, plan = resolve_plan(args.plan_dir)
    scopes = list(dict.fromkeys(scope.strip() for scope in args.scope if scope.strip()))
    if not scopes:
        raise GateError("At least one non-empty --scope is required")
    if "ALL" in scopes and scopes != ["ALL"]:
        raise GateError("ALL cannot be combined with phase-specific scopes")
    resources = parse_resources(args.resource)
    if not resources:
        raise GateError("At least one numeric --resource ceiling is required")
    for scope in scopes:
        if scope != "ALL" and not scope_is_declared(plan, scope):
            raise GateError(f"Approved scope is not declared as a plan heading: {scope}")
    plan_hash = sha256(plan)
    review = verify_go_review(directory, plan_hash)
    approval = {
        "schema_version": SCHEMA_VERSION,
        "status": "approved",
        "approved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "approved_by": args.approved_by.strip(),
        "task_plan": str(plan),
        "plan_sha256": plan_hash,
        "feasibility_review": str(review),
        "approved_scopes": scopes,
        "resource_ceilings": resources,
    }
    if not approval["approved_by"]:
        raise GateError("--approved-by cannot be empty")
    target = directory / "approval.json"
    atomic_json(target, approval)
    return {"ok": True, "approval": str(target), **approval}


def load_approval(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateError(f"Approval is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"Approval is invalid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise GateError("Approval must be a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise GateError("Unsupported approval schema version")
    if data.get("status") != "approved":
        raise GateError("Approval status is not approved")
    return data


def verify(args: argparse.Namespace) -> dict[str, Any]:
    directory, plan = resolve_plan(args.plan_dir)
    approval_path = directory / "approval.json"
    approval = load_approval(approval_path)
    current_hash = sha256(plan)
    if approval.get("plan_sha256") != current_hash:
        raise GateError("Plan hash mismatch: approval is stale")
    verify_go_review(directory, current_hash)

    phase = args.phase.strip()
    if not phase:
        raise GateError("--phase cannot be empty")
    scopes = approval.get("approved_scopes")
    if not isinstance(scopes, list) or (phase not in scopes and "ALL" not in scopes):
        raise GateError(f"Phase is outside approved scope: {phase}")

    ceilings = approval.get("resource_ceilings")
    if not isinstance(ceilings, dict):
        raise GateError("resource_ceilings must be an object")
    requested = parse_resources(args.resource)
    missing = sorted(set(ceilings) - set(requested))
    if missing:
        raise GateError(
            "Every approved resource ceiling must have a requested estimate: "
            + ", ".join(missing)
        )
    unknown = sorted(set(requested) - set(ceilings))
    if unknown:
        raise GateError("Unapproved resource requested: " + ", ".join(unknown))
    exceeded = {
        key: {"requested": requested[key], "ceiling": float(ceilings[key])}
        for key in requested
        if requested[key] > float(ceilings[key])
    }
    if exceeded:
        raise GateError("Resource ceiling exceeded: " + json.dumps(exceeded, sort_keys=True))

    return {
        "ok": True,
        "phase": phase,
        "plan_sha256": current_hash,
        "requested_resources": requested,
        "approval": str(approval_path),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    approve_parser = commands.add_parser("approve", help="Record explicit user approval")
    approve_parser.add_argument("--plan-dir", required=True, type=Path)
    approve_parser.add_argument("--scope", action="append", required=True)
    approve_parser.add_argument("--resource", action="append", default=[])
    approve_parser.add_argument("--approved-by", required=True)

    verify_parser = commands.add_parser("verify", help="Verify approval and resource scope")
    verify_parser.add_argument("--plan-dir", required=True, type=Path)
    verify_parser.add_argument("--phase", required=True)
    verify_parser.add_argument("--resource", action="append", default=[])
    return root


def main() -> int:
    args = parser().parse_args()
    result = approve(args) if args.command == "approve" else verify(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
