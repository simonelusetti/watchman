# watchman

A personal scientific literature bank. Flat structure — one summary file per paper, organisation lives in the metadata.

---

## What it is

A curated, queryable collection of scientific papers with math-forward summaries, structured frontmatter, and a local PDF archive. Designed to be managed through the `papers-bank-librarian` AI skill, which handles acquisition, cataloguing, search, and maintenance.

---

## Structure

```
papers_bank/
├── README.md
├── map.md                  ← full frontmatter spec and field reference
├── bank_index.md           ← auto-generated frontmatter snapshot of all papers
├── search_queue.json       ← input queue for the acquisition pipeline
├── papers/                 ← one {id}_summary.md per paper
├── pdfs/                   ← one {id}.pdf per paper
├── topics/                 ← concept-organised synthesis documents across papers
├── tmp/                    ← staging area for freshly downloaded PDFs
├── scripts/
│   ├── acquire.py          ← downloads PDFs from open-access sources
│   ├── zotero_sync.py      ← syncs a Zotero collection into the pipeline
│   └── check.py            ← validates consistency + regenerates bank_index.md
└── reports/
    ├── acquire_report.json ← result of the last acquisition run
    └── check_report.json   ← result of the last consistency check
```

---

## Workflows

The bank is managed through the `papers-bank-librarian` skill, which exposes six actions.

### Add papers

Tell the skill what to add — by title, arXiv ID, DOI, or Zotero collection name. It checks for duplicates, rewrites `search_queue.json` with the new entries, and tells you what to run:

```bash
python scripts/acquire.py
# or, for a Zotero collection:
python scripts/zotero_sync.py "Collection Name"
```

`acquire.py` tries five sources in order: **arXiv → ACL Anthology → Semantic Scholar → Unpaywall → DuckDuckGo**. Downloaded PDFs land in `tmp/`. Results are written to `reports/acquire_report.json`.

### Catalogue

Once `acquire.py` finishes, ask the skill to catalogue the results. It reads the report, moves PDFs from `tmp/` to `pdfs/`, writes summary files in `papers/`, and runs `check.py` to confirm integrity.

### Search & browse

Ask the skill to find papers by topic, tag, author, venue, or prominence level. It queries `bank_index.md` and returns a compact list. Full summaries available on request. Also surfaces `related_ids` entries not yet in the bank as acquisition candidates.

### Topics

Ask the skill to create a topic file for any area covered by the bank. It identifies relevant papers, reads their full summaries, and writes a math-forward synthesis document to `topics/{slug}.md`. Sections are organised by concept — not paper-by-paper — and close with a summary table and a design-axes comparison. Topic files are not tracked by git; only the directory is preserved via `.gitkeep`.

### Audit

Ask the skill to audit the bank. Produces a report in the conversation — no edits made:

1. **Structural** — runs `check.py`, checks for missing PDFs, orphan files, broken `related_ids`, incomplete frontmatter
2. **Metadata** — scans `bank_index.md` for inconsistencies, thin entries, near-duplicates
3. **Quality** — samples summary prose for math content, length, and objectivity

### Fix

Ask the skill to fix a specific issue flagged by an audit. Changes are targeted and confirmed one at a time.

---

## Scripts (standalone)

```bash
# Validate the bank and regenerate bank_index.md
python scripts/check.py

# Download papers listed in search_queue.json
python scripts/acquire.py

# Sync a Zotero collection (deduplicates, then calls acquire.py)
python scripts/zotero_sync.py "My Collection"
```

`check.py` requires no network and is safe to run anytime. `acquire.py` and `zotero_sync.py` require network access and Zotero credentials in `config.json` (not committed).

---

## Paper IDs

Each paper has a unique slug derived deterministically from its title: lowercase, strip punctuation, remove stop words, take first 6 content words, join with underscores. Example: `attention_all_you_need`. IDs are stable and used as filenames for both summaries and PDFs.

---

## Frontmatter fields

See `map.md` for the full field reference. Key fields:

| Field | Description |
|---|---|
| `id` | Unique slug |
| `title` | Full paper title |
| `tags` | Primary organisational axis — use for search and filtering |
| `prominence` | `foundational`, `notable`, or `peripheral` — literature standing |
| `one_liner` | One sentence describing the paper's contribution |
| `related_ids` | Other papers in this bank that are closely related |
