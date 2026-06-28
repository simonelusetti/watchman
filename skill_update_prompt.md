# papers-bank-librarian skill update

This document describes all changes made to the `papers_bank` codebase since the skill
was last written. Read it fully, then update `SKILL.md` accordingly.

---

## 1. Scripts renamed and added

| Old name | New name | Notes |
|---|---|---|
| `zotero_sync.py` | `zotero_import.py` | Renamed throughout |
| _(new)_ | `ingest.py` | Reconciles tmp/ after acquire — runs automatically |
| _(new)_ | `zotero_export.py` | Pushes papers from a JSON file into Zotero |
| _(new)_ | `utils.py` | Shared internal library — not user-facing |

Update every mention of `zotero_sync.py` in the skill to `zotero_import.py`.

---

## 2. Updated bank layout

```
papers_bank/
├── map.md
├── bank_index.md
├── search_queue.json
├── zotero_export_example.json    ← format template for zotero_export.py
├── papers/{id}_summary.md
├── pdfs/{id}.pdf
├── tmp/                          ← staging area; acquire.py downloads here
├── scripts/
│   ├── acquire.py
│   ├── ingest.py                 ← NEW: reconciles tmp/ with acquire report
│   ├── check.py
│   ├── zotero_import.py          ← renamed from zotero_sync.py
│   ├── zotero_export.py          ← NEW: exports papers to Zotero
│   └── utils.py                  ← internal shared utilities
└── reports/
    ├── acquire_report.json
    ├── ingest_report.json        ← NEW: written by ingest.py
    ├── zotero_import_report.json ← NEW: written by zotero_import.py
    └── check_report.json
```

---

## 3. Acquisition pipeline — corrected source chain

The skill currently lists the wrong sources. The actual source chain is:

1. **arXiv** — direct ID lookup or title search
2. **DBLP** — title search; covers ACL/EMNLP/NAACL/IEEE and links to ACL Anthology PDFs
3. **OpenAlex** — broad open-access index (250M+ works); DOI or title search
4. **Europe PMC** — biomedical / life-sciences open-access repository
5. **Web search** — DuckDuckGo fallback; scans first results for PDF links

The old sources (ACL Anthology scraper, Semantic Scholar, Unpaywall) are gone.
Replace the source list everywhere in the skill.

---

## 4. Updated acquire_report.json statuses

The `status` field values are:

- `"found"` — PDF downloaded to `tmp/` and content-verified
- `"already_in_bank"` — paper already catalogued; skipped
- `"paywalled"` — metadata found but all PDF download attempts returned 4xx errors
- `"not_found"` — no source returned metadata or a downloadable PDF

Remove `"download_failed"` and `"content_mismatch"` — these no longer appear.
Add `"paywalled"` — this is the new distinct status for papers that exist but are
behind a paywall with no open-access version.

**When reporting to the user**: treat `paywalled` as a separate category from
`not_found`. For paywalled papers, tell the user they can manually download the PDF,
drop it in `tmp/`, and run `python scripts/ingest.py` to resolve it.

---

## 5. ingest.py — new script, runs automatically

`ingest.py` is called automatically by `acquire.py` after each run. The user never
needs to invoke it manually unless resolving paywalled papers.

**What it does**: reads `acquire_report.json`, scans `tmp/` for PDFs, and classifies
each paper into one of four states:

| Status | Meaning |
|---|---|
| `ok` | acquire found it; file is in tmp/ |
| `resolved` | was paywalled/not_found; a manually placed file now covers it |
| `unrecognized` | a file in tmp/ not linked to any acquire report entry |
| `missing` | acquire couldn't get it and no manual file appeared |

It also auto-renames any non-canonically-named files to their slug, and deduplicates
files with the same slug (keeps largest).

**Paywalled paper workflow** (update the skill's "Step 4 — Tell the user what to run"):
1. User downloads the PDF manually (institutional access, etc.)
2. Drops it in `tmp/` under any filename
3. Runs `python scripts/ingest.py`
4. ingest matches it by content, renames to canonical slug, reports `resolved`

**`ingest_report.json`** — written by ingest.py:
```json
{
  "ok":           ["Paper Title", ...],
  "resolved":     ["Paper Title", ...],
  "unrecognized": ["slug_name", ...],
  "missing":      [{"title": "Paper Title", "status": "paywalled"}, ...]
}
```

---

## 6. zotero_import.py — updated behaviour

Renamed from `zotero_sync.py`. Now also attempts to download PDFs directly from Zotero
cloud storage before queuing papers for acquire:

1. Fetches items from the named collection
2. Checks each against the bank index
3. For new items, tries `zot.file()` on any PDF attachment
4. Verified PDFs → saved to `tmp/`, recorded in `zotero_import_report.json`, skipped from queue
5. Unverifiable PDFs → deleted, paper added to queue normally
6. Calls `acquire.py --zotero-report reports/zotero_import_report.json`

Flags:
- `--dry-run` — print what would happen, write nothing
- `--no-acquire` — write queue and report but do not call acquire.py

Command syntax unchanged: `python scripts/zotero_import.py "Collection Name"`

---

## 7. zotero_export.py — new script

Pushes a Claude-written JSON file of papers into a Zotero collection.

**When to use**: user asks to export papers to Zotero, populate a Zotero collection
from the bank, or share a reading list via Zotero.

**Workflow**:
1. Claude writes a `zotero_export.json` file in the bank root (see format below)
2. User runs `python scripts/zotero_export.py zotero_export.json`
3. Script finds or creates collections and subcollections in Zotero
4. Creates one Zotero item per paper; skips papers already in the target collection
5. Prints a table: Collection | Paper | Status (created / skipped)

**Export file format** (see `zotero_export_example.json` for a full example):
```json
{
  "collection": "Top-level collection name",
  "subcollections": [
    {
      "name": "Subcollection name",
      "papers": [
        {
          "title": "Full exact paper title",
          "authors": ["First Last", "First Last"],
          "year": 2023,
          "venue": "NeurIPS",
          "arxiv_id": "1706.03762",
          "doi": "10.18653/v1/...",
          "abstract": "...",
          "tags": ["tag1", "tag2"]
        }
      ]
    }
  ],
  "papers": []
}
```

- `collection` is required; everything else is optional
- `subcollections` is optional — papers can go directly under the root collection
- Each paper needs at minimum a `title`
- Papers with `arxiv_id` and no `doi` are created as `preprint`; others as `conferencePaper`
- Authors are full name strings; the script splits them into first/last automatically

**Add a new action to the skill: "Export to Zotero"**

When the user asks to export papers to Zotero, populate a Zotero collection, or
create a reading list in Zotero:
1. Ask what papers and what collection structure they want (or infer from context)
2. Check bank_index.md for the papers if they should come from the bank
3. Write `zotero_export.json` in the bank root using the format above
4. Tell the user to run: `python scripts/zotero_export.py zotero_export.json`
5. Optionally `--dry-run` first to preview

---

## 8. acquire.py — new flags

- `--clean` — delete all files in `tmp/` before running (unchanged)
- `--zotero-report [PATH]` — merge a zotero_import pre-verified PDF report. Flag absent
  = ignored. Flag alone (`--zotero-report`) = reads from
  `reports/zotero_import_report.json`. Flag with path = reads from that path.
- `--no-polite` — disable the OpenAlex polite-pool email for this run

The user never needs to pass `--zotero-report` manually — it is set automatically by
`zotero_import.py`. Mention it only if the user asks about the flag.

---

## 9. Paper ID stop words

The skill currently lists a small hardcoded stop word set. In reality, the stop words
are sklearn's full English stop word list (~318 words) plus `based` and `using`.
The practical impact on slug derivation is minimal — the key words the user needs to
know are removed (articles, prepositions, common verbs). Don't enumerate all 318 words
in the skill; just note that the list is comprehensive and includes common English
function words plus domain-specific fillers like `based` and `using`.

---

## 10. Script execution policy — update

Replace the old policy block with:

```
check.py        — can run in the sandbox (no network). Always re-run fresh.
acquire.py      — needs the user's machine (network). Tell user the command.
ingest.py       — runs automatically after acquire.py. User only needs it manually
                  to resolve paywalled papers after dropping a PDF in tmp/.
zotero_import.py — needs the user's machine (network + Zotero credentials).
zotero_export.py — needs the user's machine (Zotero credentials).
```

---

## 11. bank_index.md size note

The skill says "~13 KB for 68 papers". The bank is now at ~195 papers. Remove the
specific size claim or replace with a note that it scales with the bank.

---

## Summary of skill sections to update

| Section | Change |
|---|---|
| Bank layout | Add ingest.py, zotero_export.py, utils.py, new report files, zotero_export_example.json |
| Report files | Update statuses, add ingest_report.json |
| Script execution policy | Add ingest.py, rename zotero_sync → zotero_import, add zotero_export.py |
| Action: Add papers → Step 4 | Fix source chain, update command name, describe paywalled workflow |
| Action: Catalogue → Step 1 | Update status list, mention ingest.py runs automatically |
| Paper IDs | Update stop word description |
| New action | Export to Zotero (zotero_export.py) |
| bank_index.md size | Remove stale paper count |
