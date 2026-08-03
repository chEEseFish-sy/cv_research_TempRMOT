# TempRMOT Research Agent Workflow

## Roles

- `paper_curator`: turn named Zotero papers into L3 Obsidian notes.
- `research_dialogue`: discuss papers and project questions without modifying content.
- `research_director`: create persistent plans and coordinate reviews.
- `feasibility_reviewer`: independently return GO, REVISE, or STOP.
- `experiment_executor`: execute one approved phase.

## Typical use

```text
让 paper_curator 整理：<paper title>
让 paper_curator 批量整理：
<title 1>
<title 2>

让 research_dialogue 基于 TempRMOT 和 DKGTrack 回答：模型是否理解事件顺序？
让 research_director 为 EventRMOT 两周可行性实验建立计划。
让 feasibility_reviewer 独立审查当前 active plan。
我批准当前计划的 Phase 1，gpu_hours 上限为 8。
让 experiment_executor 执行已批准的 Phase 1。
```

## Gate

The director owns the plan. The reviewer never edits it. The executor starts only after
an independent `GO` review for the same plan hash and explicit user approval are recorded
with the phase scope and finite numeric resource ceilings. Editing the plan invalidates
both the review attestation and approval. The command-line record does not authenticate
the human by itself; the director may create it only in direct response to the user's
approval message.

## Handoff

When context reaches the shared threshold, the current agent finishes its atomic action,
creates a package under `.handoffs/`, and stops. Paste `RESUME_PROMPT.md` into a new task
to continue without rebuilding context.
