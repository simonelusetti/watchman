#!/usr/bin/env python3
"""Shared utilities for acquire.py, check.py, and zotero_sync.py."""

import re
import unicodedata
from pathlib import Path

import yaml
from prettytable import PrettyTable
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Science-paper filler words not covered by sklearn's general list.
_SCIENCE_STOP = {"based", "using"}

_STOP = ENGLISH_STOP_WORDS | _SCIENCE_STOP


def print_table(headers: list, rows: list, aligns: list | None = None) -> None:
    """Print an auto-sized table using prettytable."""
    _align_map = {"<": "l", ">": "r", "^": "c"}  # type: ignore[assignment]
    t = PrettyTable(field_names=headers)
    for i, header in enumerate(headers):
        t.align[header] = _align_map.get(aligns[i] if aligns else "<", "l")  # type: ignore[index]
    for row in rows:
        padded = list(row) + [""] * (len(headers) - len(row))
        t.add_row(padded)
    print(t)


def normalise_title(title: str) -> str:
    """Lowercase, strip diacritics and punctuation, collapse whitespace."""
    t = unicodedata.normalize("NFKD", title.lower().strip())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_frontmatter(path: Path) -> dict | None:
    """Return YAML frontmatter from a markdown file, or None on failure."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    try:
        return yaml.safe_load(text[3:end]) or {}
    except Exception:
        return None
