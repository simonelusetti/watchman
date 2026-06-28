#!/usr/bin/env python3
"""
ingest.py — reconcile tmp/ PDFs with the acquire report.

Reads reports/acquire_report.json to understand what acquire did, then
classifies every paper into one of four states:

  ok           — acquire found it and the file is in tmp/
  resolved     — acquire couldn't get it (paywalled/not_found) but a
                 manually placed file now covers it
  unrecognized — a file in tmp/ that has no matching acquire report entry
  missing      — acquire couldn't get it and no manual file has appeared

Before classifying, non-canonical filenames are renamed to their slug
(matched by content against the missing entries) and duplicate files
(same slug) are deduplicated, keeping the largest.

Usage:
    python scripts/ingest.py
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import print_table

# ---------------------------------------------------------------------------
# Borrow helpers from acquire.py
# ---------------------------------------------------------------------------

_spec = importlib.util.spec_from_file_location("acquire", Path(__file__).parent / "acquire.py")
_acq  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acq)

slugify            = _acq.slugify
verify_pdf_content = _acq.verify_pdf_content
THRESHOLD          = _acq.TITLE_SIMILARITY_THRESHOLD

ROOT       = Path(__file__).parent.parent
TMP        = ROOT / "tmp"
REPORTS    = ROOT / "reports"
ACQ_REPORT = REPORTS / "acquire_report.json"
ING_REPORT = REPORTS / "ingest_report.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_content_match(pdf: Path, candidates: list[dict]) -> tuple[dict | None, float]:
    best, best_score = None, 0.0
    for entry in candidates:
        _, score = verify_pdf_content(pdf, entry["query"])
        if score > best_score:
            best, best_score = entry, score
    return (best, best_score) if best_score >= THRESHOLD else (None, best_score)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    acq_results = json.loads(ACQ_REPORT.read_text())["results"]

    # Split report into found vs missing (skip already_in_bank — not our concern)
    found_slugs:    dict[str, dict] = {}
    missing_entries: list[dict]      = []
    for r in acq_results:
        if r["status"] == "already_in_bank":
            continue
        slug = slugify(r["query"])
        if r["status"] == "found":
            found_slugs[slug] = r
        else:
            missing_entries.append(r)

    TMP.mkdir(exist_ok=True)

    # Step 1: rename non-canonical files by content-matching against missing entries
    for pdf in sorted(TMP.glob("*.pdf")):
        if pdf.stem in found_slugs:
            continue  # acquire placed this, already canonical
        entry, _ = _best_content_match(pdf, missing_entries)
        if entry:
            canonical = slugify(entry["query"]) + ".pdf"
            if pdf.name != canonical:
                pdf.rename(TMP / canonical)

    # Step 2: dedup — keep largest file per slug
    by_stem: dict[str, list[Path]] = {}
    for pdf in sorted(TMP.glob("*.pdf")):
        by_stem.setdefault(pdf.stem, []).append(pdf)
    for files in by_stem.values():
        if len(files) > 1:
            files.sort(key=lambda p: p.stat().st_size, reverse=True)
            for dup in files[1:]:
                dup.unlink()

    present = {p.stem for p in TMP.glob("*.pdf")}
    all_report_slugs = set(found_slugs) | {slugify(r["query"]) for r in missing_entries}

    # Step 3: classify
    cat_ok           = [r for r in found_slugs.values()   if slugify(r["query"]) in present]
    cat_resolved     = [r for r in missing_entries        if slugify(r["query"]) in present]
    cat_missing      = [r for r in missing_entries        if slugify(r["query"]) not in present]
    cat_unrecognized = [s for s in present                if s not in all_report_slugs]

    # Step 4: report (skip if nothing to say)
    rows = []
    for r in cat_ok:
        rows.append((r["query"], "ok", ""))
    for r in cat_resolved:
        rows.append((r["query"], "resolved", "manually provided"))
    for s in cat_unrecognized:
        rows.append((s, "unrecognized", "not linked to any queue entry, if you purposfully manually provided this isn't a problem"))
    for r in cat_missing:
        rows.append((r["query"], "missing", "failed to find the file, manually provide it in tmp/ and rerun ingest.py resolve"))

    if rows:
        print()
        print(f"Ingest summary:")
        print_table(["Paper", "Status", "Description"], rows)

    REPORTS.mkdir(exist_ok=True)
    ING_REPORT.write_text(json.dumps({
        "ok":           [r["query"] for r in cat_ok],
        "resolved":     [r["query"] for r in cat_resolved],
        "unrecognized": cat_unrecognized,
        "missing":      [{"title": r["query"], "status": r["status"]} for r in cat_missing],
    }, indent=2))


if __name__ == "__main__":
    main()
