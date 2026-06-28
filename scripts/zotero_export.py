#!/usr/bin/env python3
"""
zotero_export.py — push a Claude-written paper list into a Zotero collection.

Usage:
    python scripts/zotero_export.py <export_file.json> [--dry-run]

    <export_file.json>  Path to the JSON file written by Claude.
                        See zotero_export_example.json for the format.
    --dry-run           Print what would be created without touching Zotero.

What it does:
    1. Reads the export file (collection name, optional subcollections, papers).
    2. Finds or creates the top-level collection in Zotero.
    3. Finds or creates each subcollection under it.
    4. Creates a Zotero item for each paper and places it in the right collection.
       Papers with an arxiv_id and no doi are created as preprints; others as
       conference papers.
    5. Skips papers whose title already exists in the target collection.

Config:
    Reads Zotero credentials from config.json (same as zotero_import.py).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import print_table

from pyzotero import zotero as pyzotero

ROOT   = Path(__file__).parent.parent
CONFIG = ROOT / "config.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_author(name: str) -> dict:
    """Split 'First Last' into pyzotero creator dict. Last word = lastName."""
    parts = name.strip().split()
    if len(parts) == 1:
        return {"creatorType": "author", "firstName": "", "lastName": parts[0]}
    return {"creatorType": "author", "firstName": " ".join(parts[:-1]), "lastName": parts[-1]}


def _build_item(paper: dict, collection_key: str) -> dict:
    arxiv_id = paper.get("arxiv_id", "")
    doi      = paper.get("doi", "")
    item_type = "preprint" if arxiv_id and not doi else "conferencePaper"

    item: dict = {
        "itemType":    item_type,
        "title":       paper["title"],
        "creators":    [_parse_author(a) for a in paper.get("authors", [])],
        "abstractNote": paper.get("abstract", ""),
        "date":        str(paper["year"]) if paper.get("year") else "",
        "url":         paper.get("url", ""),
        "DOI":         doi,
        "tags":        [{"tag": t} for t in paper.get("tags", [])],
        "collections": [collection_key],
    }

    if item_type == "preprint":
        item["repository"] = "arXiv"
        item["archiveID"]  = f"arXiv:{arxiv_id}"
        item["extra"]      = f"arXiv:{arxiv_id}"
    else:
        item["proceedingsTitle"] = paper.get("venue", "")
        if arxiv_id:
            item["extra"] = f"arXiv:{arxiv_id}"

    return item


def _find_or_create_collection(zot, name: str, parent_key: str | None = None) -> str:
    """Return the key of an existing collection matching name, or create it."""
    existing = zot.everything(
        zot.collections_sub(parent_key) if parent_key else zot.collections_top()
    )
    name_norm = name.strip().lower()
    for col in existing:
        if col["data"]["name"].strip().lower() == name_norm:
            return col["key"]

    payload = {"name": name}
    if parent_key:
        payload["parentCollection"] = parent_key
    result = zot.create_collections([payload])
    return result["successful"]["0"]["key"]


def _existing_titles(zot, collection_key: str) -> set[str]:
    items = zot.everything(zot.collection_items(collection_key, itemType="-attachment"))
    return {i["data"].get("title", "").strip().lower() for i in items}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Push a Claude-written paper list into a Zotero collection."
    )
    parser.add_argument("export_file", help="Path to the JSON export file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without touching Zotero")
    args = parser.parse_args()

    export = json.loads(Path(args.export_file).read_text())
    cfg    = json.loads(CONFIG.read_text())["zotero"]
    zot    = pyzotero.Zotero(cfg["library_id"], cfg["library_type"], cfg["api_key"])

    root_name = export["collection"]

    # Flatten sections: list of (subcollection_name_or_None, papers)
    sections: list[tuple[str | None, list[dict]]] = []
    if export.get("papers"):
        sections.append((None, export["papers"]))
    for sub in export.get("subcollections", []):
        sections.append((sub["name"], sub.get("papers", [])))

    rows = []  # (collection, title, status)

    if args.dry_run:
        for sub_name, papers in sections:
            col_label = f"{root_name} / {sub_name}" if sub_name else root_name
            for paper in papers:
                rows.append((col_label, paper["title"], "would create"))
        print()
        print_table(["Collection", "Paper", "Status"], rows)
        print("\n[dry-run] Nothing written to Zotero.")
        return

    root_key = _find_or_create_collection(zot, root_name)

    for sub_name, papers in sections:
        if sub_name:
            col_key   = _find_or_create_collection(zot, sub_name, parent_key=root_key)
            col_label = f"{root_name} / {sub_name}"
        else:
            col_key   = root_key
            col_label = root_name

        existing = _existing_titles(zot, col_key)

        for paper in papers:
            title = paper.get("title", "").strip()
            if not title:
                continue
            if title.lower() in existing:
                rows.append((col_label, title, "skipped (exists)"))
                continue

            item = _build_item(paper, col_key)
            result = zot.create_items([item])
            if result.get("successful"):
                rows.append((col_label, title, "created"))
            else:
                rows.append((col_label, title, f"failed: {result.get('failed', {})}"))

    print()
    print_table(["Collection", "Paper", "Status"], rows)


if __name__ == "__main__":
    main()
