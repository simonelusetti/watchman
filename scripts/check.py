#!/usr/bin/env python3
"""
check.py — validate the internal consistency of papers_bank.

Usage:
    python scripts/check.py

Output:
    stdout                        human-readable summary
    reports/check_report.json     machine-readable issue list
    bank_index.md                 compact frontmatter snapshot (always regenerated)

Checks performed:
    id_filename_mismatch    id field does not match the filename stem
    incomplete_frontmatter  a summary is missing one or more required fields
    empty_field             a required field is present but blank
    authors_not_list        authors field is not a list
    tags_not_list           tags field is not a list
    year_invalid            year is missing, not an integer, or outside 1000–2100
    arxiv_id_format         arxiv_id present but does not match NNNNNv.NNNNN pattern
    invalid_prominence      prominence is not foundational / notable / peripheral
    missing_pdf             declared pdf file does not exist on disk
    orphan_pdf              a PDF in pdfs/ has no matching summary
    pdf_field_mismatch      pdf field does not follow pdfs/{id}.pdf convention
    broken_related_id       a related_ids entry names a nonexistent paper
    related_ids_self        paper lists its own id in related_ids
    related_ids_duplicates  related_ids contains the same id more than once
    duplicate_title         two summaries share a normalised title
    duplicate_arxiv_id      two summaries share an arxiv_id
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import normalise_title, print_table

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).parent.parent
PAPERS  = ROOT / "papers"
PDFS    = ROOT / "pdfs"
REPORTS = ROOT / "reports"
REPORT  = REPORTS / "check_report.json"
INDEX   = ROOT / "bank_index.md"

REQUIRED_FIELDS = ["id", "title", "authors", "year", "venue", "tags",
                   "one_liner", "prominence", "related_ids", "pdf"]

VALID_PROMINENCE = {"foundational", "notable", "peripheral"}
ARXIV_RE         = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(v\d+)?$")


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

def parse_frontmatter(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    try:
        return yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError as e:
        return {"_parse_error": str(e)}


# ---------------------------------------------------------------------------
# Issue builder
# ---------------------------------------------------------------------------

def issue(type_: str, severity: str, message: str, **kwargs) -> dict:
    entry = {"type": type_, "severity": severity, "message": message}
    entry.update(kwargs)
    return entry


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def collect_summaries() -> list[dict]:
    return [{"path": p, "fm": parse_frontmatter(p) or {}}
            for p in sorted(PAPERS.glob("*_summary.md"))]


def check_all(records: list[dict]) -> list[dict]:
    issues = []

    # First pass: collect known ids for reference checks
    known_ids: set[str] = set()
    for r in records:
        fm = r["fm"]
        if "_parse_error" not in fm and fm.get("id"):
            known_ids.add(fm["id"])

    pdf_files    = {p.stem for p in PDFS.glob("*.pdf")} if PDFS.exists() else set()
    summary_ids  = {r["fm"].get("id") for r in records if "id" in r["fm"]}
    seen_titles: dict[str, str] = {}
    seen_arxiv:  dict[str, str] = {}

    for r in records:
        fm   = r["fm"]
        name = r["path"].name

        if "_parse_error" in fm:
            issues.append(issue("parse_error", "error",
                f"[{name}] Could not parse frontmatter: {fm['_parse_error']}",
                file=name))
            continue

        pid = fm.get("id", "")

        # id matches filename
        expected_id = r["path"].stem.removesuffix("_summary")
        if pid and pid != expected_id:
            issues.append(issue("id_filename_mismatch", "error",
                f"[{name}] id '{pid}' does not match filename (expected '{expected_id}')",
                id=pid, file=name, expected_id=expected_id))

        # Required fields present
        missing = [f for f in REQUIRED_FIELDS if f not in fm or fm[f] is None]
        if missing:
            issues.append(issue("incomplete_frontmatter", "error",
                f"[{pid or name}] Missing required fields: {', '.join(missing)}",
                id=pid, file=name, missing_fields=missing))

        # Required fields not empty
        empty = [f for f in REQUIRED_FIELDS
                 if f in fm and fm[f] is not None and fm[f] == ""]
        if empty:
            issues.append(issue("empty_field", "warning",
                f"[{pid}] Empty required fields: {', '.join(empty)}",
                id=pid, file=name, empty_fields=empty))

        # authors is a list
        authors = fm.get("authors")
        if authors is not None and not isinstance(authors, list):
            issues.append(issue("authors_not_list", "error",
                f"[{pid}] authors must be a list, got {type(authors).__name__}",
                id=pid, file=name))

        # tags is a list
        tags = fm.get("tags")
        if tags is not None and not isinstance(tags, list):
            issues.append(issue("tags_not_list", "error",
                f"[{pid}] tags must be a list, got {type(tags).__name__}",
                id=pid, file=name))

        # year is a valid integer
        year = fm.get("year")
        if year is not None:
            if not isinstance(year, int):
                issues.append(issue("year_invalid", "warning",
                    f"[{pid}] year should be an integer, got {type(year).__name__} '{year}'",
                    id=pid, file=name, value=year))
            elif not (1000 <= year <= 2100):
                issues.append(issue("year_invalid", "warning",
                    f"[{pid}] year {year} is outside the expected range 1000–2100",
                    id=pid, file=name, value=year))

        # arxiv_id format
        arxiv_id = fm.get("arxiv_id")
        if arxiv_id is not None:
            if not ARXIV_RE.match(str(arxiv_id)):
                issues.append(issue("arxiv_id_format", "warning",
                    f"[{pid}] arxiv_id '{arxiv_id}' does not match expected format NNNN.NNNNNvN",
                    id=pid, file=name, value=arxiv_id))

        # prominence value
        prominence = fm.get("prominence")
        if prominence and prominence not in VALID_PROMINENCE:
            issues.append(issue("invalid_prominence", "warning",
                f"[{pid}] prominence '{prominence}' is not one of {sorted(VALID_PROMINENCE)}",
                id=pid, file=name, value=prominence))

        # PDF exists
        declared_pdf = fm.get("pdf", "")
        if declared_pdf:
            if not (ROOT / declared_pdf).exists():
                issues.append(issue("missing_pdf", "error",
                    f"[{pid}] Declared pdf '{declared_pdf}' not found on disk",
                    id=pid, file=name, declared_pdf=declared_pdf))
        elif pid and pid not in pdf_files:
            issues.append(issue("missing_pdf", "error",
                f"[{pid}] No pdf field and pdfs/{pid}.pdf not found",
                id=pid, file=name))

        # pdf naming convention
        if declared_pdf and pid and declared_pdf != f"pdfs/{pid}.pdf":
            issues.append(issue("pdf_field_mismatch", "warning",
                f"[{pid}] pdf field is '{declared_pdf}', expected 'pdfs/{pid}.pdf'",
                id=pid, file=name, declared=declared_pdf, expected=f"pdfs/{pid}.pdf"))

        # related_ids
        related = fm.get("related_ids") or []
        for ref in related:
            if ref == pid:
                issues.append(issue("related_ids_self", "warning",
                    f"[{pid}] related_ids contains the paper's own id",
                    id=pid, file=name))
            elif ref not in known_ids:
                issues.append(issue("broken_related_id", "warning",
                    f"[{pid}] related_ids references '{ref}' which has no summary",
                    id=pid, file=name, related_id=ref))

        seen = set()
        for ref in related:
            if ref in seen:
                issues.append(issue("related_ids_duplicates", "warning",
                    f"[{pid}] related_ids contains duplicate '{ref}'",
                    id=pid, file=name, related_id=ref))
            seen.add(ref)

        # Duplicate title
        title = fm.get("title", "")
        if title and pid:
            norm = normalise_title(title)
            if norm in seen_titles and seen_titles[norm] != pid:
                issues.append(issue("duplicate_title", "error",
                    f"'{pid}' and '{seen_titles[norm]}' share a normalised title",
                    ids=[pid, seen_titles[norm]], normalised_title=norm))
            else:
                seen_titles[norm] = pid

        # Duplicate arxiv_id
        if arxiv_id and pid:
            aid = str(arxiv_id).split("v")[0]
            if aid in seen_arxiv and seen_arxiv[aid] != pid:
                issues.append(issue("duplicate_arxiv_id", "error",
                    f"'{pid}' and '{seen_arxiv[aid]}' share arxiv_id '{aid}'",
                    ids=[pid, seen_arxiv[aid]], arxiv_id=aid))
            else:
                seen_arxiv[aid] = pid

    # Orphan PDFs
    for stem in sorted(pdf_files):
        if stem not in summary_ids:
            issues.append(issue("orphan_pdf", "warning",
                f"pdfs/{stem}.pdf has no matching summary in papers/",
                stem=stem, pdf=f"pdfs/{stem}.pdf"))

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
            "total_pdfs":      len(list(PDFS.glob("*.pdf"))) if PDFS.exists() else 0,
            "issue_count":     len(issues),
            "error_count":     len(errors),
            "warning_count":   len(warnings),
            "clean":           len(issues) == 0,
        },
        "issues": issues,
    }


def print_report(report: dict) -> None:
    s   = report["summary"]
    c1, c2, c3 = 22, 5, 50
    rows: list = [("summaries", s["total_summaries"], ""), ("PDFs", s["total_pdfs"], "")]

    if not s["clean"]:
        by_type: dict[str, list] = {}
        for iss in report["issues"]:
            by_type.setdefault(iss["type"], []).append(iss)
        for type_, group in sorted(by_type.items()):
            marker = "!" if group[0]["severity"] == "error" else "~"
            ids    = [i.get("id") or i.get("stem", "?") for i in group]
            short  = [x.replace("_", " ")[:20].strip() for x in ids[:3]]
            notes  = ", ".join(short) + (f" (+{len(ids)-3} more)" if len(ids) > 3 else "")
            rows.append((f"{marker} {type_}", len(group), notes[:c3]))

    print()
    print_table(["Bank", "Count", "Notes"], rows, [c1, c2, c3], ["<", ">", "<"])


# ---------------------------------------------------------------------------
# Bank index
# ---------------------------------------------------------------------------

def generate_bank_index(records: list[dict]) -> None:
    valid = [r for r in records if "_parse_error" not in r["fm"]]
    lines = [
        "# papers_bank — bank index",
        f"<!-- auto-generated by check.py on {datetime.now(timezone.utc).strftime('%Y-%m-%d')} -->",
        f"<!-- {len(valid)} papers — frontmatter only; full summaries in papers/{{id}}_summary.md -->",
        "",
    ]
    for r in sorted(valid, key=lambda x: x["fm"].get("id", "")):
        text = r["path"].read_text(encoding="utf-8")
        summary_start = text.find("\n## Summary")
        block = text[:summary_start].rstrip() if summary_start != -1 else text.rstrip()
        lines.append(block)
        lines.append("")
    INDEX.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not PAPERS.exists():
        print(f"papers/ not found at {PAPERS}")
        sys.exit(1)

    records = collect_summaries()
    issues  = check_all(records)
    report  = build_report(records, issues)

    REPORTS.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print_report(report)

    generate_bank_index(records)

    if report["summary"]["error_count"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
