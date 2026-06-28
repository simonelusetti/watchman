#!/usr/bin/env python3
"""Shared utilities for acquire.py, check.py, and zotero_sync.py."""

import re
import unicodedata
from pathlib import Path

import yaml
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Science-paper filler words not covered by sklearn's general list.
_SCIENCE_STOP = {"based", "using"}

_STOP = ENGLISH_STOP_WORDS | _SCIENCE_STOP


def print_table(headers: list, rows: list, widths: list, aligns: list | None = None) -> None:
    """Print a fixed-width table with a header row and separator."""
    if aligns is None:
        aligns = ["<"] * len(widths)
    sep  = "  "
    fmt  = sep.join(f"{{:{a}{w}}}" for a, w in zip(aligns, widths))
    rule = sep.join("─" * w for w in widths)

    def _row(values):
        padded = list(values) + [""] * (len(widths) - len(values))
        cells  = [str(v)[:w] for v, w in zip(padded, widths)]
        return fmt.format(*cells)

    print(_row(headers))
    print(rule)
    for row in rows:
        print(_row(row))


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
