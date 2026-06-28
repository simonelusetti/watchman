#!/usr/bin/env python3
"""
zotero_import.py — pull a Zotero collection into the acquisition pipeline.

Usage:
    python scripts/zotero_import.py <collection_name> [--dry-run] [--no-acquire]

    <collection_name>   Exact name of a top-level or nested Zotero collection.
                        Case-insensitive. Use quotes if the name contains spaces.
    --dry-run           Print what would be queued without writing anything or
                        calling acquire.py.
    --no-acquire        Write search_queue.json and zotero_import_report.json but
                        do not call acquire.py.

What it does:
    1. Fetches all items in the named collection from Zotero.
    2. Checks each against the bank index (via acquire.already_in_bank).
    3. For new items, attempts to download the PDF directly from Zotero storage.
       - Verified PDFs are saved to tmp/ and recorded in zotero_import_report.json.
       - Malformed or unverifiable PDFs are deleted and the paper is queued normally.
    4. Writes remaining papers (no Zotero PDF) to search_queue.json.
    5. Calls acquire.py --zotero-report so it merges the pre-verified PDFs.

Config:
    Reads Zotero credentials from config.json at the bank root:
    {
      "zotero": {
        "api_key": "...",
        "library_id": "...",
        "library_type": "user",   // or "group"
        "email": "..."            // used by Unpaywall in acquire.py
      }
    }
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import print_table

from pyzotero import zotero as pyzotero

# ---------------------------------------------------------------------------
# Borrow bank + PDF helpers from acquire.py
# ---------------------------------------------------------------------------

_spec = importlib.util.spec_from_file_location("acquire", Path(__file__).parent / "acquire.py")
_acq  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acq)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).parent.parent
CONFIG        = ROOT / "config.json"
QUEUE         = ROOT / "search_queue.json"
REPORTS       = ROOT / "reports"
ZOTERO_REPORT = REPORTS / "zotero_import_report.json"
ACQUIRE       = ROOT / "scripts" / "acquire.py"
TMP           = ROOT / "tmp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_arxiv_id(item_data: dict) -> str | None:
    patterns = [
        r"arxiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
    ]
    for field in ("extra", "url", "DOI"):
        value = item_data.get(field, "") or ""
        for pat in patterns:
            m = re.search(pat, value, re.IGNORECASE)
            if m:
                return m.group(1).split("v")[0]
    return None


def extract_doi(item_data: dict) -> str | None:
    doi = item_data.get("DOI", "").strip()
    return doi if doi else None


def find_collection_key(zot, name: str) -> str | None:
    name_norm = name.strip().lower()
    for col in zot.everything(zot.collections()):
        if col["data"]["name"].strip().lower() == name_norm:
            return col["key"]
    return None


def try_zotero_pdf(zot, item_key: str, title: str) -> tuple[bool, float]:
    """Try to download a verified PDF from Zotero storage for this item.

    Returns (success, content_score). On success the file is saved to
    tmp/{slug}.pdf. On failure any partial file is removed.
    """
    try:
        children = zot.children(item_key, itemType="attachment")
    except Exception:
        return False, 0.0

    for att in children:
        if att["data"].get("contentType") != "application/pdf":
            continue
        att_key = att["key"]
        slug     = _acq.slugify(title)
        pdf_path = TMP / f"{slug}.pdf"
        try:
            pdf_bytes = zot.file(att_key)
            if not isinstance(pdf_bytes, bytes) or pdf_bytes[:4] != b"%PDF":
                continue
            pdf_path.write_bytes(pdf_bytes)
            passed, score = _acq.verify_pdf_content(pdf_path, title)
            if passed:
                return True, score
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pdf_path.unlink(missing_ok=True)

    return False, 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sync a Zotero collection into the papers_bank acquisition pipeline."
    )
    parser.add_argument("collection", help="Zotero collection name (case-insensitive)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be queued without writing files or running acquire.py")
    parser.add_argument("--no-acquire", action="store_true",
                        help="Write queue and report but do not call acquire.py")
    args = parser.parse_args()

    cfg = json.loads(CONFIG.read_text())["zotero"]
    zot = pyzotero.Zotero(cfg["library_id"], cfg["library_type"], cfg["api_key"])

    col_key = find_collection_key(zot, args.collection)
    if not col_key:
        collections = zot.everything(zot.collections())
        print(f"Collection '{args.collection}' not found. Available:")
        print_table(["Collection"], [[c["data"]["name"]] for c in collections])
        sys.exit(1)

    items = zot.everything(zot.collection_items(col_key, itemType="-attachment"))
    bank  = _acq.load_bank_index()
    TMP.mkdir(exist_ok=True)

    rows             = []   # for the summary table
    queue_entries    = []   # papers needing acquire
    zotero_verified  = []   # papers with confirmed Zotero PDFs

    for item in items:
        data     = item["data"]
        title    = data.get("title", "").strip()
        if not title or title.lower() == "untitled":
            continue

        arxiv_id = extract_arxiv_id(data)
        doi      = extract_doi(data)
        source   = arxiv_id or (f"doi:{doi}" if doi else "title search")

        entry: dict = {"title": title}
        if arxiv_id:
            entry["arxiv_id"] = arxiv_id
        if doi:
            entry["doi"] = doi

        if _acq.already_in_bank(entry, bank):
            rows.append((title, source, "in bank", ""))
            continue

        if args.dry_run:
            rows.append((title, source, "new", ""))
            queue_entries.append(entry)
            continue

        ok, score = try_zotero_pdf(zot, item["key"], title)
        if ok:
            zotero_verified.append({"query": title, "file": f"tmp/{_acq.slugify(title)}.pdf",
                                    "content_score": score})
            rows.append((title, source, "zotero pdf", f"{score:.2f}"))
        else:
            queue_entries.append(entry)
            rows.append((title, source, "queued", ""))

    print()
    print_table(["Paper", "Source", "Status", "Score"], rows)

    if args.dry_run:
        print("\n[dry-run] Nothing written. acquire.py NOT called.")
        return

    REPORTS.mkdir(exist_ok=True)
    ZOTERO_REPORT.write_text(json.dumps({"results": zotero_verified}, indent=2))
    QUEUE.write_text(json.dumps(queue_entries, indent=2, ensure_ascii=False))

    if args.no_acquire:
        return

    result = subprocess.run(
        [sys.executable, str(ACQUIRE), "--zotero-report", str(ZOTERO_REPORT)],
        cwd=str(ROOT),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
