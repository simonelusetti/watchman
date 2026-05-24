#!/usr/bin/env python3
"""
check.py — validate the internal consistency of papers_bank.

Usage:
    python scripts/check.py

Output:
    stdout              human-readable summary
    consistency_report.json   machine-readable issue list for agent consumption

Checks performed:
    missing_pdf         a _summary.md exists but its declared pdf: file is absent
    orphan_pdf          a PDF in pdfs/ has no matching _summary.md
    incomplete_fm       a summary is missing one or more required frontmatter fields
    broken_related_id   a related_ids entry names an id that has no summary
    duplicate_title     two summaries share a normalised title (likely the same paper)
    duplicate_arxiv_id  two summaries declare the same arxiv_id (optional field)
    pdf_field_mismatch  the pdf: field in frontmatter doesn't follow the expected
                        convention (pdfs/{id}.pdf)
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml --break-system-packages")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).parent.parent
PAPERS  = ROOT / "papers"
PDFS    = ROOT / "pdfs"
REPORTS = ROOT / "reports"
REPORT  = REPORTS / "check_report.json"

REQUIRED_FIELDS = ["id", "title", "authors", "year", "venue", "tags",
                   "one_liner", "prominence", "related_ids", "pdf"]

VALID_PROMINENCE = {"foundational", "notable", "peripheral"}


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

def parse_frontmatter(path: Path) -> dict | None:
    """Return the YAML frontmatter dict from a markdown file, or None on failure."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    # Find closing ---
    end = text.find("\n---", 3)
    if end == -1:
        return None
    raw = text[3:end]
    try:
        return yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        return {"_parse_error": str(e)}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalise_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Issue builders
# ---------------------------------------------------------------------------

def issue(type_: str, severity: str, message: str, **kwargs) -> dict:
    entry = {"type": type_, "severity": severity, "message": message}
    entry.update(kwargs)
    return entry


# ---------------------------------------------------------------------------
# Main checks
# ---------------------------------------------------------------------------

def collect_summaries() -> list[dict]:
    """Parse every _summary.md and return a list of {path, fm} dicts."""
    records = []
    for p in sorted(PAPERS.glob("*_summary.md")):
        fm = parse_frontmatter(p)
        records.append({"path": p, "fm": fm or {}})
    return records


def check_all(records: list[dict]) -> list[dict]:
    issues = []

    known_ids = set()
    for r in records:
        fm = r["fm"]
        if "_parse_error" in fm:
            issues.append(issue(
                "parse_error", "error",
                f"Could not parse frontmatter in {r['path'].name}: {fm['_parse_error']}",
                file=r["path"].name,
            ))
            continue
        pid = fm.get("id")
        if pid:
            known_ids.add(pid)

    pdf_files = {p.stem for p in PDFS.glob("*.pdf")} if PDFS.exists() else set()
    summary_ids = {r["fm"].get("id") for r in records if "id" in r["fm"]}

    seen_titles: dict[str, str] = {}   # normalised_title -> id
    seen_arxiv: dict[str, str] = {}    # arxiv_id -> id

    for r in records:
        fm   = r["fm"]
        name = r["path"].name

        if "_parse_error" in fm:
            continue

        pid = fm.get("id", "")

        # ── Incomplete frontmatter ──────────────────────────────────────────
        missing = [f for f in REQUIRED_FIELDS if f not in fm or fm[f] is None]
        if missing:
            issues.append(issue(
                "incomplete_frontmatter", "error",
                f"[{pid or name}] Missing required fields: {', '.join(missing)}",
                id=pid, file=name, missing_fields=missing,
            ))

        # ── Invalid prominence value ─────────────────────────────────────────
        prominence = fm.get("prominence")
        if prominence and prominence not in VALID_PROMINENCE:
            issues.append(issue(
                "invalid_prominence", "warning",
                f"[{pid}] prominence='{prominence}' is not one of {sorted(VALID_PROMINENCE)}",
                id=pid, file=name, value=prominence,
            ))

        # ── Missing PDF ─────────────────────────────────────────────────────
        declared_pdf = fm.get("pdf", "")
        if declared_pdf:
            declared_path = ROOT / declared_pdf
            if not declared_path.exists():
                issues.append(issue(
                    "missing_pdf", "error",
                    f"[{pid}] Declared pdf '{declared_pdf}' does not exist on disk.",
                    id=pid, file=name, declared_pdf=declared_pdf,
                ))
        elif pid:
            # No pdf field at all — check conventional path
            if pid not in pdf_files:
                issues.append(issue(
                    "missing_pdf", "error",
                    f"[{pid}] No pdf field; conventional file pdfs/{pid}.pdf also absent.",
                    id=pid, file=name,
                ))

        # ── pdf: field naming convention ────────────────────────────────────
        if declared_pdf and pid:
            expected = f"pdfs/{pid}.pdf"
            if declared_pdf != expected:
                issues.append(issue(
                    "pdf_field_mismatch", "warning",
                    f"[{pid}] pdf field is '{declared_pdf}', expected '{expected}'.",
                    id=pid, file=name, declared=declared_pdf, expected=expected,
                ))

        # ── Broken related_ids ──────────────────────────────────────────────
        for ref in (fm.get("related_ids") or []):
            if ref not in known_ids:
                issues.append(issue(
                    "broken_related_id", "warning",
                    f"[{pid}] related_ids references '{ref}' which has no summary in papers/.",
                    id=pid, file=name, related_id=ref,
                ))

        # ── Duplicate title ─────────────────────────────────────────────────
        title = fm.get("title", "")
        if title and pid:
            norm = normalise_title(title)
            if norm in seen_titles and seen_titles[norm] != pid:
                issues.append(issue(
                    "duplicate_title", "error",
                    f"Papers '{pid}' and '{seen_titles[norm]}' share a normalised title.",
                    ids=[pid, seen_titles[norm]], normalised_title=norm,
                ))
            else:
                seen_titles[norm] = pid

        # ── Duplicate arxiv_id ──────────────────────────────────────────────
        arxiv_id = fm.get("arxiv_id")
        if arxiv_id and pid:
            if arxiv_id in seen_arxiv and seen_arxiv[arxiv_id] != pid:
                issues.append(issue(
                    "duplicate_arxiv_id", "error",
                    f"Papers '{pid}' and '{seen_arxiv[arxiv_id]}' share arxiv_id '{arxiv_id}'.",
                    ids=[pid, seen_arxiv[arxiv_id]], arxiv_id=arxiv_id,
                ))
            else:
                seen_arxiv[arxiv_id] = pid

    # ── Orphan PDFs ─────────────────────────────────────────────────────────
    for stem in sorted(pdf_files):
        if stem not in summary_ids:
            issues.append(issue(
                "orphan_pdf", "warning",
                f"pdfs/{stem}.pdf has no matching _summary.md in papers/.",
                stem=stem, pdf=f"pdfs/{stem}.pdf",
            ))

    return issues


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(records: list[dict], issues: list[dict]) -> dict:
    errors   = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_summaries": len(records),
            "total_pdfs": len(list(PDFS.glob("*.pdf"))) if PDFS.exists() else 0,
            "issue_count": len(issues),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "clean": len(issues) == 0,
        },
        "issues": issues,
    }


def print_report(report: dict) -> None:
    s = report["summary"]
    print(f"\n=== papers_bank consistency check — {report['checked_at'][:10]} ===")
    print(f"  Summaries : {s['total_summaries']}")
    print(f"  PDFs      : {s['total_pdfs']}")
    print(f"  Issues    : {s['issue_count']}  ({s['error_count']} errors, {s['warning_count']} warnings)")

    if s["clean"]:
        print("\n✓ All checks passed.")
        return

    by_type: dict[str, list] = {}
    for iss in report["issues"]:
        by_type.setdefault(iss["type"], []).append(iss)

    for type_, group in sorted(by_type.items()):
        sev = group[0]["severity"].upper()
        print(f"\n[{sev}] {type_} ({len(group)})")
        for iss in group:
            print(f"  • {iss['message']}")

    print(f"\nFull report written to reports/check_report.json")


# ---------------------------------------------------------------------------
# Bank index generator
# ---------------------------------------------------------------------------

INDEX = ROOT / "bank_index.md"


def generate_bank_index(records: list[dict]) -> None:
    """
    Write bank_index.md — a frontmatter-only snapshot of every paper in the bank.

    Each entry is the raw YAML frontmatter block from the corresponding
    _summary.md, with the ## Summary prose body stripped out.  The result is a
    compact, machine-readable file (~5 KB for 70 papers) that the
    papers-bank-librarian skill reads at the start of every session to give it
    full awareness of the bank without loading the full summaries.

    Regenerated automatically at the end of every check.py run.
    """
    valid = [r for r in records if "_parse_error" not in r["fm"]]
    lines = [
        "# papers_bank — bank index",
        f"<!-- auto-generated by check.py on {datetime.now(timezone.utc).strftime('%Y-%m-%d')} -->",
        f"<!-- {len(valid)} papers — frontmatter only; full summaries in papers/{{id}}_summary.md -->",
        "",
    ]

    for r in sorted(valid, key=lambda x: x["fm"].get("id", "")):
        text = r["path"].read_text(encoding="utf-8")
        # Slice out everything before ## Summary (keeps the frontmatter block intact)
        summary_start = text.find("\n## Summary")
        block = text[:summary_start].rstrip() if summary_start != -1 else text.rstrip()
        lines.append(block)
        lines.append("")   # blank line between entries

    INDEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"Bank index written → bank_index.md  ({len(valid)} papers)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not PAPERS.exists():
        print(f"papers/ directory not found at {PAPERS}")
        sys.exit(1)

    records = collect_summaries()
    issues  = check_all(records)
    report  = build_report(records, issues)

    REPORTS.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print_report(report)

    generate_bank_index(records)

    # Exit code 1 if any errors (not warnings) — useful for CI / watcher scripts
    if report["summary"]["error_count"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
