# watchman

A personal scientific literature bank. Flat structure — one summary file per paper, organisation lives in the metadata.

---

## What it is

A curated, queryable collection of scientific papers with math-forward summaries, structured frontmatter, and a local PDF archive. Managed through the `papers-bank-librarian` AI skill, which handles acquisition, cataloguing, search, topic synthesis, and maintenance.

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

`papers/`, `pdfs/`, `topics/`, and `tmp/` are gitignored — only `.gitkeep` files are tracked to preserve directory structure.

---

## Paper IDs

Each paper has a unique slug derived deterministically from its title: lowercase, strip punctuation, remove stop words, take first 6 content words, join with underscores.

Example: `attention_all_you_need`

IDs are stable and used as filenames for both summaries and PDFs.

---

## Frontmatter fields

See `map.md` for the full field reference. Key fields:

| Field | Description |
|---|---|
| `id` | Unique slug |
| `title` | Full paper title |
| `authors` | List of authors |
| `year` | Publication year |
| `arxiv_id` | arXiv ID, e.g. `"2005.00928"` *(optional)* |
| `venue` | Conference or journal |
| `tags` | Primary organisational axis — use for search and filtering |
| `one_liner` | One sentence describing the paper's contribution |
| `prominence` | `foundational`, `notable`, or `peripheral` — literature standing |
| `related_ids` | Other papers in this bank that are closely related |
| `pdf` | Vault-relative path to the PDF |

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

`check.py` requires no network and is safe to run anytime. `acquire.py` and `zotero_sync.py` require network access; Zotero credentials go in `config.json` (not committed — see `config.json.example`).

`acquire.py` tries five sources in order: **arXiv → ACL Anthology → Semantic Scholar → Unpaywall → DuckDuckGo**. PDFs land in `tmp/`; results written to `reports/acquire_report.json`.

---

## Skill actions

All interactions go through the `papers-bank-librarian` skill. It always reads `bank_index.md` first for awareness before taking any action.

### Add papers

Triggered by: titles, arXiv IDs, DOIs, a Zotero collection name, or a topic description.

The skill identifies specific papers, checks for duplicates against `bank_index.md`, and **rewrites** `search_queue.json` with the new entries (if there are existing queued-but-unrun entries, it asks whether to overwrite or merge). It then tells you to run `acquire.py` (or `zotero_sync.py` for Zotero collections). After delivering those instructions, it also suggests a topic file if the queue has a clear coherent theme.

### Catalogue

Triggered by: "the download is done", "catalogue the results", or similar.

The skill reads `reports/acquire_report.json`, moves PDFs from `tmp/` to `pdfs/`, and writes a `papers/{id}_summary.md` for each successfully downloaded paper. Summaries are math-forward — 3–4 prose paragraphs with LaTeX equations — written in a neutral, project-agnostic style. Finishes by running `check.py` and reporting any issues.

### Search & browse

Triggered by: questions about what's in the bank — by topic, tag, author, venue, year, or prominence level.

The skill queries `bank_index.md` and returns a compact list (`id`, year, title, one-liner). Full summaries available on request. If any `related_ids` in the results point to papers not yet in the bank, it flags them as acquisition candidates.

### Topics

Triggered by: "create a topic file on X", "write a topic summary for Y", or the suggestion offered after queue-filling.

The skill identifies all relevant papers in the bank, asks for confirmation if the topic boundary is ambiguous, reads the full summaries of selected papers, and writes `topics/{slug}.md`. The file is concept-organised — sections cover problems and methods, not individual papers — and closes with a summary table and a design-axes comparison. If a topic file for that area already exists, it asks whether to overwrite or create a new one.

Topic files are not tracked by git.

### Audit

Triggered by: "audit the bank", "check the bank", "run maintenance".

Produces a read-only report in three passes — no files are changed:

1. **Structural** — runs `check.py`; flags missing PDFs, orphan files, broken `related_ids`, incomplete frontmatter
2. **Metadata** — scans `bank_index.md` for ID/filename mismatches, suspicious years, thin entries, near-duplicates
3. **Quality** — samples 5–8 summary files for math content, prose length, and objectivity

Ends with a ranked action list. To act on any finding, ask the skill to fix a specific item.

### Fix

Triggered by: an explicit request to fix something, usually after an audit.

Makes one targeted change at a time — reads the file, edits the specific field, runs `check.py` if structural, shows a before/after. Does not batch-fix without per-issue confirmation.
