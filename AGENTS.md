# TempRMOT project skills

The project no longer uses custom research agents or automatic research-agent routing.

Two project skills remain available for the local research knowledge base:

- `curate-rmot-paper`: read exact papers from Zotero and safely create or update managed notes
  under `D:\ObsidianVault\Research` after an explicit note-write request.
- `discuss-rmot-research`: read Obsidian research notes, local project evidence, and Zotero
  sources for evidence-grounded discussion without changing files.

Invoke these skills directly when their descriptions match the user's request. Do not modify
Zotero records, `.obsidian`, or researcher-authored text outside managed markers. All other
coding, planning, review, and execution work is handled directly by the main Codex task under
the user's current authorization.
