---
name: discuss-rmot-research
description: Conduct read-only, evidence-grounded discussion about RMOT papers, Obsidian research notes, and the TempRMOT project. Use when the user asks research_dialogue to explain, compare, challenge, connect, or answer a question from Zotero papers and existing notes. Do not use when any note, Zotero item, plan, or project file should be changed.
---

# Discuss RMOT Research

## Sources

1. Read `D:\ObsidianVault\Research` first.
2. Read local project documentation when the question concerns TempRMOT state.
3. If note evidence is insufficient, use the installed Zotero skills read-only to inspect the exact paper, annotation, figure, table, or section.
4. Do not browse the web unless the user explicitly requests new literature.
5. Read [answer-contract.md](references/answer-contract.md).

## Answer

Return:

1. Direct answer.
2. Evidence and locations, with confidence.
3. Counterargument, missing evidence, or weakest assumption.
4. Meaning for TempRMOT or the active research direction.
5. One question that advances the research.

Distinguish author claims, paper evidence, user annotations, and inference. If the knowledge
base is incomplete, provide one prompt for `paper_curator`; do not repair the note.

## Boundary

Remain strictly read-only. Do not modify files, notes, Zotero, plans, metadata, or tags.
Follow the shared handoff protocol; when due, emit a handoff payload and resume prompt for
the parent task to persist.
