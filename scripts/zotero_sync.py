#!/usr/bin/env python3
"""
zotero_sync.py — pull a Zotero collection into the acquisition pipeline.

Usage:
    python scripts/zotero_sync.py <collection_name> [--dry-run]

    <collection_name>   Exact name of a top-level or nested Zotero collection.
                        Case-insensitive. Use quotes if the name contains spaces.
    --dry-run           Print what would be queued without writing anything or
                        calling acquire.py.

What it does:
    1. Fetches all items in the named collection from Zotero.
    2. Checks each against the bank index (via acquire.already_in_bank).
    3. Writes only new papers to search_queue.json.
    4. Calls acquire.py to download them.

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
# Borrow bank helpers from acquire.py
# ---------------------------------------------------------------------------

_spec = importlib.util.spec_from_file_location("acquire", Path(__file__).parent / "acquire.py")
_acq  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acq)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).parent.parent
CONFIG  = ROOT / "config.json"
QUEUE   = ROOT / "search_queue.json"
ACQUIRE = ROOT / "scripts" / "acquire.py"


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

    rows        = []
    new_entries = []

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
            rows.append((title, source, "in bank"))
        else:
            new_entries.append(entry)
            rows.append((title, source, "new"))

    print()
    print_table(["Paper", "Source", "Status"], rows)

    if not new_entries:
        print("\nBank is up to date with this collection.")
        return

    if args.dry_run:
        print("\n[dry-run] search_queue.json NOT written. acquire.py NOT called.")
        return

    QUEUE.write_text(json.dumps(new_entries, indent=2, ensure_ascii=False))
    result = subprocess.run([sys.executable, str(ACQUIRE)], cwd=str(ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
