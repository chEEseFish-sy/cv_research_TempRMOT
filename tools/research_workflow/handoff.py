#!/usr/bin/env python3
"""Evaluate context handoff thresholds and create executable handoff packages."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


AGENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SLUG_RE = re.compile(r"[^a-z0-9]+")
FORBIDDEN_KEYS = {"chain_of_thought", "hidden_reasoning", "reasoning_trace"}
LIST_FIELDS = (
    "decisions",
    "evidence",
    "modified_files",
    "validation",
    "open_questions",
    "risks",
    "failed_attempts",
    "prohibitions",
)


class HandoffError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_threshold(args: argparse.Namespace) -> dict[str, Any]:
    triggers: list[str] = []
    if args.context_percent is not None and args.context_percent >= 75:
        triggers.append("context>=75%")
    if args.substantive_turns >= 12:
        triggers.append("substantive_turns>=12")
    if args.tool_calls >= 40:
        triggers.append("tool_calls>=40")
    if args.compacted:
        triggers.append("compaction")
    if args.l3_papers >= 3:
        triggers.append("before_fourth_L3_paper")
    return {"handoff_required": bool(triggers), "triggers": triggers}


def load_payload(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffError(f"Cannot read payload {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise HandoffError("Payload must be a JSON object")
    forbidden = FORBIDDEN_KEYS.intersection(data)
    if forbidden:
        raise HandoffError("Forbidden hidden-reasoning fields: " + ", ".join(sorted(forbidden)))
    for key in ("agent", "objective", "current_state", "next_action"):
        if not str(data.get(key, "")).strip():
            raise HandoffError(f"Missing required field: {key}")
    if not AGENT_RE.fullmatch(str(data["agent"])):
        raise HandoffError("agent must use snake_case")
    for key in LIST_FIELDS:
        value = data.get(key, [])
        if not isinstance(value, list):
            raise HandoffError(f"{key} must be an array")
        data[key] = [str(item).strip() for item in value if str(item).strip()]
    return data


def slug(value: str) -> str:
    normalized = SLUG_RE.sub("-", value.lower()).strip("-")
    return (normalized or "task")[:48]


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None recorded."


def plan_details(data: dict[str, Any]) -> tuple[str, str]:
    raw = str(data.get("active_plan", "")).strip()
    if not raw:
        return "None", "None"
    path = Path(raw).resolve()
    if path.is_dir():
        path = path / "task_plan.md"
    if not path.is_file():
        raise HandoffError(f"active_plan does not exist: {path}")
    return str(path), file_sha256(path)


def render(data: dict[str, Any], handoff_path: Path) -> tuple[str, str]:
    plan_path, plan_hash = plan_details(data)
    role = str(data.get("role", "")).strip() or data["agent"]
    handoff = f"""# Research Task Handoff

## Identity

- Agent: `{data['agent']}`
- Role: {role}
- Objective: {data['objective']}
- Scope: {str(data.get('scope', '')).strip() or 'Not separately specified.'}

## Current state

{data['current_state']}

Last completed atomic action: {str(data.get('last_completed_action', '')).strip() or 'Not recorded.'}

## Active plan

- Path: `{plan_path}`
- SHA-256: `{plan_hash}`

## Decisions

{bullets(data['decisions'])}

## Evidence and artifacts

{bullets(data['evidence'])}

## Modified files

{bullets(data['modified_files'])}

## Validation

{bullets(data['validation'])}

## Open questions

{bullets(data['open_questions'])}

## Risks

{bullets(data['risks'])}

## Failed attempts

{bullets(data['failed_attempts'])}

## Prohibitions

{bullets(data['prohibitions'])}

## Exact next action

{data['next_action']}
"""
    resume = f"""请接替 TempRMOT 科研任务。

1. 先完整阅读交接文档：`{handoff_path}`。
2. 使用项目 agent `{data['agent']}`，并加载该 agent 要求的项目 skill。
3. 核对当前文件系统、Git 状态、Obsidian/Zotero 只读状态以及 active plan 的 SHA-256；不要假设交接后状态未变化。
4. 从交接文档的“Exact next action”继续，不要重复已经完成的探索、精读、实验或修改。
5. 严格遵守交接文档中的 scope、资源上限、prohibitions 和人工批准门。
6. 如果 plan hash、批准范围或关键证据不一致，立即停止并报告差异。
"""
    return handoff, resume


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def create_package(args: argparse.Namespace) -> dict[str, Any]:
    data = load_payload(args.payload)
    now = dt.datetime.now()
    directory_name = (
        now.strftime("%Y%m%d-%H%M%S")
        + f"-{data['agent']}-"
        + slug(str(data.get("slug") or data["objective"]))
    )
    target = args.root.resolve() / directory_name
    handoff_path = target / "HANDOFF.md"
    resume_path = target / "RESUME_PROMPT.md"
    handoff, resume = render(data, handoff_path)
    result = {
        "ok": True,
        "mode": "apply" if args.apply else "dry-run",
        "directory": str(target),
        "handoff": str(handoff_path),
        "resume_prompt": str(resume_path),
    }
    if args.apply:
        if target.exists():
            raise HandoffError(f"Handoff target already exists: {target}")
        target.mkdir(parents=True)
        atomic_write(handoff_path, handoff)
        atomic_write(resume_path, resume)
    else:
        result["handoff_preview"] = handoff
        result["resume_prompt_preview"] = resume
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Evaluate handoff thresholds")
    check.add_argument("--context-percent", type=float)
    check.add_argument("--substantive-turns", type=int, default=0)
    check.add_argument("--tool-calls", type=int, default=0)
    check.add_argument("--l3-papers", type=int, default=0)
    check.add_argument("--compacted", action="store_true")

    create = commands.add_parser("create", help="Create a handoff package")
    create.add_argument("--payload", required=True, type=Path)
    create.add_argument("--root", type=Path, default=Path(".handoffs"))
    create.add_argument("--apply", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    result = check_threshold(args) if args.command == "check" else create_package(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HandoffError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
