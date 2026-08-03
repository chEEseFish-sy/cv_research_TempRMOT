# Research Agent Handoff Protocol

All five project agents use this protocol to prevent context loss.

## Trigger

Create a handoff when any condition is met:

- exposed context usage is at least 75%;
- 12 substantive user-agent exchanges occurred;
- 40 tool calls occurred;
- the task is about to compact or has compacted once.

Paper curation also hands off before starting a fourth L3 paper in the same task.

## Behavior

Finish the current atomic operation, do not start a new phase, and create an executable
handoff package. Write-capable agents use `tools/research_workflow/handoff.py`.
Read-only agents emit the same payload and resume prompt in their response; the parent task
persists it.

Store packages under:

```text
.handoffs/YYYYMMDD-HHMMSS-<agent>-<slug>/
  HANDOFF.md
  RESUME_PROMPT.md
```

## Required content

- agent and role;
- objective and explicit scope;
- current state and last completed atomic action;
- decisions and their evidence;
- active plan path and SHA-256 when applicable;
- source notes, papers, files, and artifacts;
- modified files and validation results;
- open questions, risks, failed attempts, and prohibitions;
- exact next action.

Do not include hidden chain-of-thought. Include concise decision rationale and verifiable
evidence only.

The resume prompt must tell the new task to read `HANDOFF.md`, verify current filesystem
and plan state, load the same project agent and required skills, continue from the exact
next action, and avoid repeating completed work.
