---
name: curate-rmot-paper
description: Deep-read one or more exact papers from the local Zotero library and safely create or update evidence-grounded RMOT paper notes in Obsidian. Use when the user asks to organize, read, summarize, compare, or archive named Zotero papers into the TempRMOT research knowledge base. Do not use for open-ended web discovery, research chat without note writes, or Zotero metadata changes.
---

# Curate RMOT Paper

Create durable research notes rather than generic summaries. Use Chinese prose while preserving English titles, terms, variables, formulas, and necessary excerpts.

## Required sources

1. Load and follow `zotero-research-notes`.
2. Load and follow `Zotero` for local API readiness and exact metadata retrieval.
3. Read [note-contract.md](references/note-contract.md) and use [paper-note-sections.md](assets/paper-note-sections.md).

## Workflow

1. Run the curated Zotero helper `status --json`. Never read `zotero.sqlite`.
2. Treat each non-empty input line as one title. Search narrowly, merge by Item Key, and show matched items before reading.
3. If a title is ambiguous, stop on that title and request selection. Do not guess.
4. Process no more than three matched papers per turn, sequentially. Return remaining titles as a queue.
5. Default to L3. Read the full paper, figures, tables, formulas, appendix, Zotero notes, and annotations. Record page or section locations.
6. If full text is missing, create or update only an L1 `triaged` note with `L3 blocked: full text unavailable`.
7. Separate author claims, paper evidence, user annotations, Codex inference, and open questions.
8. Build the managed-note payload. Run the installed `upsert_paper_note.py` without `--apply`, inspect the complete diff, then apply.
9. Never create a new domain, tag, concept, method, dataset, or topic note without approval.
10. Run the installed vault audit and report items used, evidence ceiling, files changed, and queued titles.

## Safety

- Write only under `D:\ObsidianVault\Research`.
- Never modify `.obsidian`, Zotero records, attachments, project code, or text outside managed markers.
- Do not infer unsupported details merely to fill the template.
