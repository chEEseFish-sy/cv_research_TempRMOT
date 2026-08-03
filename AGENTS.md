# TempRMOT research agents

Use the narrowest project agent for the task:

- `paper_curator`: Zotero-to-Obsidian paper curation. It may write only managed research notes in `D:\ObsidianVault\Research`.
- `research_dialogue`: evidence-grounded research discussion. It is strictly read-only.
- `research_director`: owns research plans and may write only `.planning/` and `.handoffs/`.
- `feasibility_reviewer`: independent, read-only Go/Revise/Stop review.
- `experiment_executor`: executes only a phase authorized by a valid `approval.json`.

Never let the executor change the research question, success criteria, approval scope, or plan. A changed `task_plan.md` invalidates approval. Never let any agent edit Zotero records, `.obsidian`, or researcher-authored text outside managed markers.

All agents follow `docs/research/RESEARCH_AGENT_HANDOFF_PROTOCOL.md`. At a handoff threshold, finish the current atomic action, generate a handoff package, and stop before starting another phase.
