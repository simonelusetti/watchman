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
    2. Loads existing papers in the bank (papers/*_summary.md) to build a
       known-titles / known-arxiv-ids set.
    3. Writes only *new* papers (not already in the bank) to search_queue.json.
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
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

try:
    from pyzotero import zotero as pyzotero
except ImportError:
    print("Error: pyzotero not installed. Run: pip install pyzotero")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).parent.parent
CONFIG  = ROOT / "config.json"
PAPERS  = ROOT / "papers"
QUEUE   = ROOT / "search_queue.json"
ACQUIRE = ROOT / "scripts" / "acquire.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if not CONFIG.exists():
        print(f"Error: config.json not found at {CONFIG}")
        print("Create it with your Zotero credentials (see script docstring).")
        sys.exit(1)
    return json.loads(CONFIG.read_text())


def normalise_title(title: str) -> str:
    """Lowercase, strip diacritics and punctuation, collapse whitespace."""
    t = unicodedata.normalize("NFKD", title.lower().strip())
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_arxiv_id(item_data: dict) -> str | None:
    """
    Try to find an arXiv ID in the Zotero item.
    Zotero stores it in various places depending on how the item was imported:
      - extra field: "arXiv:2005.00928" or "arXiv ID: 2005.00928"
      - url field: "https://arxiv.org/abs/2005.00928"
    """
    patterns = [
        r"arxiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)",   # extra field
        r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",  # URL
    ]
    for field in ("extra", "url", "DOI"):
        value = item_data.get(field, "") or ""
        for pat in patterns:
            m = re.search(pat, value, re.IGNORECASE)
            if m:
                return m.group(1).split("v")[0]   # strip version suffix
    return None


def extract_doi(item_data: dict) -> str | None:
    doi = item_data.get("DOI", "").strip()
    return doi if doi else None


# ---------------------------------------------------------------------------
# Existing bank index
# ---------------------------------------------------------------------------

def load_existing_papers() -> tuple[set[str], set[str]]:
    """
    Returns (known_titles, known_arxiv_ids) — normalised sets built from
    the frontmatter of every *_summary.md in papers/.
    """
    known_titles: set[str] = set()
    known_arxiv: set[str] = set()

    if not PAPERS.exists():
        return known_titles, known_arxiv

    for p in PAPERS.glob("*_summary.md"):
        text = p.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        end = text.find("\n---", 3)
        if end == -1:
            continue
        try:
            fm = yaml.safe_load(text[3:end]) or {}
        except yaml.YAMLError:
            continue

        title = fm.get("title", "")
        if title:
            known_titles.add(normalise_title(title))

        arxiv_id = fm.get("arxiv_id")
        if arxiv_id:
            known_arxiv.add(str(arxiv_id).split("v")[0])

    return known_titles, known_arxiv


# ---------------------------------------------------------------------------
# Zotero helpers
# ---------------------------------------------------------------------------

def find_collection_key(zot, name: str) -> str | None:
    """
    Return the Zotero key for the first collection whose name matches
    `name` (case-insensitive), or None if not found.
    """
    collections = zot.everything(zot.collections())
    name_norm = name.strip().lower()
    for col in collections:
        if col["data"]["name"].strip().lower() == name_norm:
            return col["key"]
    return None


def fetch_collection_items(zot, collection_key: str) -> list[dict]:
    """Return all non-attachment items in a collection."""
    return zot.everything(
        zot.collection_items(collection_key, itemType="-attachment")
    )


# ---------------------------------------------------------------------------
# Queue builder
# ---------------------------------------------------------------------------

def build_queue_entries(
    items: list[dict],
    known_titles: set[str],
    known_arxiv: set[str],
) -> tuple[list[dict], list[str]]:
    """
    Convert Zotero items to search_queue entries, skipping papers already
    in the bank.  Returns (new_entries, skipped_titles).
    """
    new_entries: list[dict] = []
    skipped: list[str] = []

    for item in items:
        data = item["data"]
        title = data.get("title", "").strip()
        if not title or title.lower() == "untitled":
            continue

        norm = normalise_title(title)
        arxiv_id = extract_arxiv_id(data)
        doi = extract_doi(data)

        # Skip if already in bank
        if norm in known_titles:
            skipped.append(title)
            continue
        if arxiv_id and arxiv_id in known_arxiv:
            skipped.append(title)
            continue

        entry: dict = {"title": title}
        if arxiv_id:
            entry["arxiv_id"] = arxiv_id
        if doi:
            entry["doi"] = doi

        new_entries.append(entry)

    return new_entries, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sync a Zotero collection into the papers_bank acquisition pipeline."
    )
    parser.add_argument("collection", help="Zotero collection name (case-insensitive)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be queued without writing files or running acquire.py",
    )
    args = parser.parse_args()

    cfg = load_config()["zotero"]
    zot = pyzotero.Zotero(cfg["library_id"], cfg["library_type"], cfg["api_key"])

    # ── Find collection ─────────────────────────────────────────────────────
    print(f"Looking for collection '{args.collection}'...")
    col_key = find_collection_key(zot, args.collection)
    if not col_key:
        print(f"Error: collection '{args.collection}' not found in your Zotero library.")
        print("Available collections:")
        for col in zot.everything(zot.collections()):
            print(f"  • {col['data']['name']}")
        sys.exit(1)

    # ── Fetch items ─────────────────────────────────────────────────────────
    print(f"Fetching items from '{args.collection}'...")
    items = fetch_collection_items(zot, col_key)
    print(f"  {len(items)} item(s) found in collection.")

    # ── Load existing bank ──────────────────────────────────────────────────
    known_titles, known_arxiv = load_existing_papers()
    print(f"  {len(known_titles)} paper(s) already in bank (will be skipped).")

    # ── Build queue ──────────────────────────────────────────────────────────
    new_entries, skipped = build_queue_entries(items, known_titles, known_arxiv)

    if skipped:
        print(f"\nSkipped ({len(skipped)} already in bank):")
        for t in skipped:
            print(f"  ✓ {t}")

    if not new_entries:
        print("\nNothing new to acquire. Bank is up to date with this collection.")
        return

    print(f"\nNew papers to acquire ({len(new_entries)}):")
    for e in new_entries:
        hint = e.get("arxiv_id") or e.get("doi") or "title search"
        print(f"  → {e['title']}  [{hint}]")

    if args.dry_run:
        print("\n[dry-run] search_queue.json NOT written. acquire.py NOT called.")
        return

    # ── Write queue ──────────────────────────────────────────────────────────
    QUEUE.write_text(json.dumps(new_entries, indent=2, ensure_ascii=False))
    print(f"\nWritten {len(new_entries)} entr(ies) to search_queue.json.")

    # ── Call acquire.py ──────────────────────────────────────────────────────
    print("Running acquire.py...\n")
    result = subprocess.run(
        [sys.executable, str(ACQUIRE)],
        cwd=str(ROOT),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
