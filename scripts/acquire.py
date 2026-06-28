#!/usr/bin/env python3
"""
acquire.py — download papers listed in search_queue.json.

Usage:
    python scripts/acquire.py

Input:  search_queue.json   list of paper requests (see format below)
Output: tmp/                downloaded PDFs land here
        reports/acquire_report.json   one entry per paper: found / not_found / download_failed

search_queue.json format:
[
  { "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" },
  { "title": "SimCSE ...", "arxiv_id": "2104.08821" },
  { "title": "Some ACL paper ...", "doi": "10.18653/v1/..." }
]

Only "title" is required. Providing arxiv_id or doi makes acquisition faster and
more reliable — the script will skip fuzzy search and go straight to the source.

Source chain (tried in order until a PDF is found):
  1. arXiv         — direct ID lookup or title search; exponential backoff on 429
  2. DBLP          — title search; covers ACL/EMNLP/NAACL/IEEE/etc. → ACL Anthology PDFs
  3. OpenAlex      — broad open-access index (250M+ works); DOI or title search
  4. Europe PMC    — biomedical / life-sciences open-access repository
  5. Web search    — DuckDuckGo fallback; scans first results for PDF links

Override the source list in config.json:
  { "sources": ["arxiv", "dblp", "openalex", "europepmc", "web_search"] }
"""

import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import _STOP, normalise_title, parse_frontmatter, print_table

import requests
from bs4 import BeautifulSoup

import arxiv as arxiv_lib
import fitz  # pymupdf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).parent.parent
QUEUE   = ROOT / "search_queue.json"
PAPERS  = ROOT / "papers"
REPORTS = ROOT / "reports"
REPORT  = REPORTS / "acquire_report.json"
TMP     = ROOT / "tmp"
CHECK   = ROOT / "scripts" / "check.py"
INGEST  = ROOT / "scripts" / "ingest.py"

_cfg_path = ROOT / "config.json"
try:
    _cfg = json.loads(_cfg_path.read_text())
    _UNPAYWALL_EMAIL = _cfg["zotero"]["email"]
except Exception:
    _cfg = {}
    _UNPAYWALL_EMAIL = "user@example.com"

_DEFAULT_DOMAINS = [
    "arxiv.org", "aclanthology.org", "openreview.net", "semanticscholar.org",
    "aclweb.org", "nlp.stanford.edu", "cs.cornell.edu", "proceedings.mlr.press",
    "papers.nips.cc", "neurips.cc",
    "ncbi.nlm.nih.gov", "europepmc.org", "medrxiv.org", "biorxiv.org",
]
ACADEMIC_DOMAINS = _cfg.get("academic_domains") or _DEFAULT_DOMAINS

HEADERS = {"User-Agent": "paper-acquire/1.0 (research pipeline; contact via github)"}

TITLE_SIMILARITY_THRESHOLD = 0.80  # recall: 80% of query words must appear in result
PDF_CONTENT_THRESHOLD      = 0.75  # slightly looser — PDF title extraction is noisier

# Compiled once; used in _extract_pdf_title to filter non-title spans
_NOISE = re.compile(
    r"arxiv:\d{4}\.\d{4,5}"
    r"|^\d{4}\.\d{4,5}(v\d+)?$"
    r"|\[cs\.|preprint|under review"
    r"|proceedings of|conference on"
    r"|^\d+$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(title: str, n: int = 6) -> str:
    words   = re.sub(r"[^a-z0-9\s]", "", title.lower()).split()
    content = [w for w in words if w not in _STOP]
    return "_".join(content[:n])


def download_pdf(url: str, dest: Path) -> tuple[bool, str]:
    try:
        r = requests.get(url, timeout=30, stream=True, headers=HEADERS)
        r.raise_for_status()
        content = r.content
        if content[:4] != b"%PDF":
            return False, f"not a PDF ({r.headers.get('content-type', '?')})"
        dest.write_bytes(content)
        return True, ""
    except Exception as e:
        return False, str(e)


def _extract_pdf_title(doc) -> str:
    meta = (doc.metadata or {}).get("title", "").strip()
    if (meta and len(meta) > 8
            and not re.search(r"\.(pdf|prn|docx?|tex|odt)\s*$", meta, re.I)
            and "\\" not in meta):
        return meta

    page        = doc[0]
    page_height = page.rect.height

    spans = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        if block.get("bbox", [0, 0, 0, page_height])[1] > page_height * 0.60:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = span.get("size", 0)
                if text and size > 0 and not _NOISE.search(text):
                    spans.append((size, text))

    if not spans:
        return ""

    max_size    = max(s for s, _ in spans)
    title_parts = [t for s, t in spans if s >= max_size * 0.85]
    return " ".join(title_parts[:10])


def verify_pdf_content(pdf_path: Path, query_title: str) -> tuple[bool, float]:
    """Check the downloaded PDF actually contains the paper we asked for.

    Returns (passed, score). Falls back to first-page word recall for scanned / metadata-free PDFs.
    """
    try:
        doc       = fitz.open(str(pdf_path))
        extracted = _extract_pdf_title(doc)

        if extracted:
            doc.close()
            score = title_similarity(query_title, extracted)
            return score >= PDF_CONTENT_THRESHOLD, score

        # Fallback: word recall on first page (scanned / metadata-free PDFs)
        text        = doc[0].get_text()
        doc.close()
        text_words  = set(normalise_title(text).split())
        query_words = {w for w in normalise_title(query_title).split() if w not in _STOP}
        if not query_words:
            return True, 1.0
        recall = len(query_words & text_words) / len(query_words)
        return recall >= PDF_CONTENT_THRESHOLD, recall

    except Exception as e:
        tqdm.write(f"  PDF read error: {e}")
        return False, 0.0


def sanitize_for_search(title: str) -> str:
    title = unicodedata.normalize("NFC", title)
    title = re.sub(r"[–—−‐]", "-", title)
    title = re.sub(r"[''`]",   "'", title)
    title = re.sub(r'[""]',    '"', title)
    return title.encode("ascii", "ignore").decode("ascii").strip()


def title_similarity(query: str, result: str) -> float:
    """Fraction of query words found in result (recall). Ignores stop-words.

    Recall-based so that distinctive query words must be present in the result.
    Symmetric F1 buries missing unique words when titles share a common suffix.
    """
    q_words = {w for w in normalise_title(query).split() if w not in _STOP}
    r_words = {w for w in normalise_title(result).split() if w not in _STOP}
    if not q_words:
        return 0.0
    return len(q_words & r_words) / len(q_words)


# ---------------------------------------------------------------------------
# Source 1: arXiv
# ---------------------------------------------------------------------------

def try_arxiv(paper: dict) -> dict | None:
    arxiv_id = (paper.get("arxiv_id") or "").split("v")[0]
    if arxiv_id:
        return {
            "title": paper["title"], "authors": [], "year": None,
            "venue": "arXiv", "abstract": "", "arxiv_id": arxiv_id,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}", "source": "arxiv",
        }

    backoff     = 10
    client      = arxiv_lib.Client(delay_seconds=5.0, num_retries=1)
    clean_title = sanitize_for_search(paper["title"])

    def _search(query: str) -> list:
        for attempt in range(2):
            try:
                return list(client.results(arxiv_lib.Search(query=query, max_results=3)))
            except arxiv_lib.HTTPError as e:
                if e.status == 429 and attempt == 0:
                    tqdm.write(f"  arXiv rate-limited, waiting {backoff}s...")
                    time.sleep(backoff)
                else:
                    tqdm.write(f"  arXiv error ({e.status}), skipping")
                    return []
            except Exception as e:
                tqdm.write(f"  arXiv error: {e}")
                return []
        return []

    hits = _search(f'ti:"{clean_title}"')
    if not hits:
        keywords = [w for w in clean_title.split() if w.lower() not in _STOP and len(w) > 2]
        if keywords:
            hits = _search(" ".join(f"ti:{w}" for w in keywords[:8]))

    if not hits:
        return None

    best, best_score = max(
        ((p, title_similarity(paper["title"], p.title)) for p in hits),
        key=lambda x: x[1],
    )
    if best_score < TITLE_SIMILARITY_THRESHOLD:
        return None

    return {
        "title":    best.title,
        "authors":  [a.name for a in best.authors],
        "year":     best.published.year,
        "venue":    "arXiv",
        "abstract": best.summary.replace("\n", " "),
        "arxiv_id": best.entry_id.split("/")[-1].split("v")[0],
        "pdf_url":  best.pdf_url,
        "source":   "arxiv",
    }


# ---------------------------------------------------------------------------
# Source 2: DBLP (CS publication index → ACL Anthology PDFs and more)
# ---------------------------------------------------------------------------

DBLP_URL = "https://dblp.org/search/publ/api"
ACL_BASE  = "https://aclanthology.org"


def try_dblp(paper: dict) -> dict | None:
    """Search DBLP and return metadata for papers hosted on ACL Anthology."""
    try:
        r = requests.get(
            DBLP_URL,
            params={"q": sanitize_for_search(paper["title"]), "format": "json", "h": 5},
            timeout=20,
            headers=HEADERS,
        )
        r.raise_for_status()
    except Exception as e:
        tqdm.write(f"  DBLP request failed: {e}")
        return None

    for hit in (r.json().get("result", {}).get("hits", {}).get("hit") or []):
        info  = hit.get("info", {})
        score = title_similarity(paper["title"], info.get("title", ""))
        if score < TITLE_SIMILARITY_THRESHOLD:
            continue

        ee = info.get("ee", "")
        if not isinstance(ee, str):
            continue
        acl_m = re.search(r"aclanthology\.org/(\S+?)/?$", ee)
        if not acl_m:
            continue

        paper_id = acl_m.group(1).rstrip("/")
        raw_authors = info.get("authors", {}).get("author") or []
        if isinstance(raw_authors, dict):
            raw_authors = [raw_authors]
        authors = [a.get("text", a) if isinstance(a, dict) else a for a in raw_authors]
        year_m  = re.search(r"\b(19|20)\d{2}\b", info.get("year", ""))

        return {
            "title":    info.get("title", paper["title"]).rstrip("."),
            "authors":  authors,
            "year":     int(year_m.group()) if year_m else None,
            "venue":    info.get("venue", "ACL Anthology"),
            "abstract": "",
            "acl_id":   paper_id,
            "pdf_url":  f"{ACL_BASE}/{paper_id}.pdf",
            "source":   "acl_anthology",
        }
    return None


# ---------------------------------------------------------------------------
# Source 3: OpenAlex (replaces Semantic Scholar, Unpaywall, PMC, Europe PMC)
# ---------------------------------------------------------------------------

OA_BASE = "https://api.openalex.org/works"


def try_openalex(paper: dict) -> dict | None:
    params = {"mailto": _UNPAYWALL_EMAIL}

    # DOI fast path — direct lookup, no title matching needed
    if paper.get("doi"):
        r = requests.get(f"{OA_BASE}/doi:{paper['doi']}", params=params, timeout=10)
        if r.ok:
            return _oa_parse(r.json())

    # Title search
    r = requests.get(OA_BASE, params={**params, "search": paper["title"], "per_page": 3}, timeout=10)
    if not r.ok:
        return None
    for item in r.json().get("results", []):
        score = title_similarity(paper["title"], item.get("title", ""))
        if score >= TITLE_SIMILARITY_THRESHOLD:
            return _oa_parse(item)
    return None


def _oa_parse(w: dict) -> dict | None:
    best    = w.get("best_oa_location") or {}
    pdf_url = best.get("pdf_url") or (w.get("open_access") or {}).get("oa_url")
    if not pdf_url:
        return None
    ids      = w.get("ids") or {}
    arxiv_m  = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", ids.get("arxiv") or "")
    return {
        "title":    w.get("title", ""),
        "authors":  [a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author")],
        "year":     w.get("publication_year"),
        "venue":    ((w.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
        "abstract": _oa_abstract(w.get("abstract_inverted_index")),
        "arxiv_id": arxiv_m.group(1) if arxiv_m else None,
        "doi":      (ids.get("doi") or "").replace("https://doi.org/", "") or None,
        "pdf_url":  pdf_url,
        "source":   "openalex",
    }


def _oa_abstract(idx: dict | None) -> str:
    if not idx:
        return ""
    length = max(pos for positions in idx.values() for pos in positions) + 1
    words  = [""] * length
    for word, positions in idx.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


# ---------------------------------------------------------------------------
# Source 4: Europe PMC (biomedical / life sciences open access)
# ---------------------------------------------------------------------------

EPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def try_europepmc(paper: dict) -> dict | None:
    try:
        r = requests.get(
            EPMC_URL,
            params={"query": paper["title"], "format": "json", "resultType": "core", "pageSize": 3},
            timeout=15,
            headers=HEADERS,
        )
        r.raise_for_status()
    except Exception as e:
        tqdm.write(f"  Europe PMC request failed: {e}")
        return None

    for item in r.json().get("resultList", {}).get("result", []):
        score = title_similarity(paper["title"], item.get("title", ""))
        if score < TITLE_SIMILARITY_THRESHOLD:
            continue
        pdf_url = next(
            (u["url"] for u in item.get("fullTextUrlList", {}).get("fullTextUrl", [])
             if u.get("documentStyle") == "pdf"),
            None,
        )
        if not pdf_url:
            continue
        return {
            "title":    item.get("title", paper["title"]),
            "authors":  [a.get("fullName", "") for a in item.get("authorList", {}).get("author", [])],
            "year":     item.get("pubYear"),
            "venue":    item.get("journalTitle", ""),
            "abstract": item.get("abstractText", ""),
            "doi":      item.get("doi"),
            "pdf_url":  pdf_url,
            "source":   "europepmc",
        }
    return None


# ---------------------------------------------------------------------------
# Source 5: Web search fallback (DuckDuckGo)
# ---------------------------------------------------------------------------

DDG_URL = "https://html.duckduckgo.com/html/"


def try_web_search(paper: dict) -> dict | None:
    clean_title = sanitize_for_search(paper["title"])
    query = f'"{clean_title}" filetype:pdf OR site:arxiv.org OR site:aclanthology.org OR site:openreview.net'
    try:
        r = requests.post(DDG_URL, data={"q": query, "kl": "us-en"},
                          headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
                          timeout=15)
        r.raise_for_status()
    except Exception as e:
        tqdm.write(f"  web search failed: {e}")
        return None

    for link in BeautifulSoup(r.text, "html.parser").select("a.result__url, a.result__a")[:10]:
        href = link.get("href", "")
        if "uddg=" in href:
            m = re.search(r"uddg=([^&]+)", href)
            if m:
                href = urllib.parse.unquote(m.group(1))
        if not href.startswith("http"):
            continue
        domain = urllib.parse.urlparse(href).netloc.lstrip("www.")
        if not any(href.lower().endswith(".pdf") or d in domain for d in ACADEMIC_DOMAINS):
            continue
        href = _coerce_to_pdf_url(href)
        if not href:
            continue
        return {
            "title": paper["title"], "authors": [], "year": None,
            "venue": "", "abstract": "", "pdf_url": href, "source": "web_search",
        }
    return None


def _coerce_to_pdf_url(url: str) -> str | None:
    m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    m = re.search(r"aclanthology\.org/(\S+?)/?$", url)
    if m:
        pid = m.group(1).rstrip("/")
        return f"https://aclanthology.org/{pid}.pdf" if not pid.endswith(".pdf") else f"https://aclanthology.org/{pid}"
    m = re.search(r"openreview\.net/forum\?id=(\S+)", url)
    if m:
        return f"https://openreview.net/pdf?id={m.group(1)}"
    return url if url.lower().endswith(".pdf") else None


# ---------------------------------------------------------------------------
# Bank index
# ---------------------------------------------------------------------------

def load_bank_index() -> dict:
    """Return a lookup dict keyed by normalised title and "arxiv:{id}"."""
    index: dict = {}
    if not PAPERS.exists():
        return index
    for p in PAPERS.glob("*_summary.md"):
        fm = parse_frontmatter(p)
        if fm is None:
            continue
        entry = {"id": fm.get("id"), "year": fm.get("year"), "arxiv_id": fm.get("arxiv_id")}
        if fm.get("title"):
            index[normalise_title(fm["title"])] = entry
        if fm.get("arxiv_id"):
            index[f"arxiv:{str(fm['arxiv_id']).split('v')[0]}"] = entry
    return index


def already_in_bank(metadata: dict, bank: dict) -> dict | None:
    """Return the existing bank entry for this paper, or None if it's new."""
    arxiv_id = (metadata.get("arxiv_id") or "").split("v")[0]
    if arxiv_id:
        existing = bank.get(f"arxiv:{arxiv_id}")
        if existing:
            return existing

    norm     = normalise_title(metadata.get("title", ""))
    existing = bank.get(norm)
    if existing:
        bank_year, source_year = existing.get("year"), metadata.get("year")
        if bank_year and source_year and source_year > bank_year:
            return None  # newer version — treat as new
        return existing

    return None


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_ALL_SOURCES = {
    "arxiv":         try_arxiv,
    "dblp":          try_dblp,
    "europepmc":     try_europepmc,
    "openalex":      try_openalex,
    "web_search":    try_web_search,
}

_DEFAULT_SOURCES = ["arxiv", "dblp", "openalex", "europepmc", "web_search"]


def _build_sources() -> list:
    try:
        names = json.loads(_cfg_path.read_text()).get("sources") or _DEFAULT_SOURCES
    except Exception:
        names = _DEFAULT_SOURCES
    return [_ALL_SOURCES[n] for n in names if n in _ALL_SOURCES]


SOURCES = _build_sources()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preflight(queue: list, bank: dict) -> tuple[list, list]:
    """Split queue into (already_in_bank, to_fetch)."""
    skip_bank, to_fetch = [], []
    for paper in queue:
        arxiv_id = (paper.get("arxiv_id") or "").split("v")[0]
        existing = bank.get(f"arxiv:{arxiv_id}") if arxiv_id else None
        if not existing:
            existing = bank.get(normalise_title(paper["title"]))
        if existing:
            skip_bank.append((paper, existing))
        else:
            to_fetch.append(paper)
    return skip_bank, to_fetch


def acquire(queue: list) -> list:
    TMP.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    bank = load_bank_index()

    skip_bank, to_fetch = preflight(queue, bank)

    print(f"Pre-flight: {len(queue)} papers — {len(skip_bank)} in bank, {len(to_fetch)} to fetch")

    results = []
    for paper, existing in skip_bank:
        results.append({"query": paper["title"], "status": "already_in_bank", "existing_id": existing.get("id")})

    iterator = tqdm(to_fetch, desc="Acquiring", unit="paper")

    for paper in iterator:
        title      = paper["title"]
        tqdm.write(title)
        result     = None
        had_paywall = False

        for source_fn in SOURCES:
            metadata = source_fn(paper)
            if not metadata:
                continue

            existing = already_in_bank(metadata, bank)
            if existing:
                tqdm.write(f"  already in bank: {existing['id']}")
                result = {"query": title, "status": "already_in_bank", "existing_id": existing["id"]}
                break

            slug     = slugify(title)
            pdf_path = TMP / f"{slug}.pdf"
            cached   = pdf_path.exists() and pdf_path.stat().st_size > 0

            if not cached:
                ok, err = download_pdf(metadata["pdf_url"], pdf_path)
                if not ok:
                    if re.search(r"\b4\d{2}\b", err):
                        had_paywall = True
                    tqdm.write(f"  {metadata['source']}: download failed — {err}")
                    continue

            passed, score = verify_pdf_content(pdf_path, title)

            if not passed and cached:
                # Stale file — try a fresh download before giving up on this source
                pdf_path.unlink(missing_ok=True)
                ok, err = download_pdf(metadata["pdf_url"], pdf_path)
                if not ok:
                    if re.search(r"\b4\d{2}\b", err):
                        had_paywall = True
                    tqdm.write(f"  {metadata['source']}: download failed — {err}")
                    continue
                passed, score = verify_pdf_content(pdf_path, title)

            if passed:
                tqdm.write(f"  found via {metadata['source']} ({score:.2f})")
                result = {"query": title, "status": "found", "pdf": f"tmp/{slug}.pdf",
                          "content_score": score, "metadata": metadata}
                break
            else:
                tqdm.write(f"  {metadata['source']}: content mismatch ({score:.2f})")
                pdf_path.unlink(missing_ok=True)

        if result is None:
            status = "paywalled" if had_paywall else "not_found"
            tqdm.write(f"  {status}")
            result = {"query": title, "status": status}

        results.append(result)
        time.sleep(3)

    return results


_STATUS_LABEL = {
    "found":          "found",
    "already_in_bank": "in bank",
    "paywalled":      "paywalled",
    "not_found":      "not found",
}

_STATUS_DETAIL = {
    "found":          "PDF downloaded and content verified",
    "already_in_bank": "already present in papers/",
    "paywalled":      "metadata found but all PDF URLs returned 4xx",
    "not_found":      "no source returned metadata or a downloadable PDF",
}


def _print_summary(results: list) -> None:
    c1, c2, c3 = 50, 10, 30
    paper_rows = []
    for r in results:
        title = r["query"]
        t     = (title[:c1 - 1] + "…") if len(title) > c1 else title
        label = _STATUS_LABEL.get(r["status"], r["status"])
        notes = f"{r['content_score']:.2f}" if r["status"] == "found" else ""
        paper_rows.append((t, label, notes))
    print()
    print_table(["Paper", "Status", "Notes"], paper_rows, [c1, c2, c3], ["<", ">", "<"])

    counts = {s: sum(1 for r in results if r["status"] == s) for s in _STATUS_LABEL}
    c1, c2, c3 = 12, 5, 48
    status_rows = [
        (label, counts[status], _STATUS_DETAIL[status])
        for status, label in _STATUS_LABEL.items()
        if counts[status]
    ]
    print()
    print_table(["Status", "Count", "Description"], status_rows, [c1, c2, c3], ["<", ">", "<"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Delete tmp/*.pdf before running")
    args = parser.parse_args()

    TMP.mkdir(exist_ok=True)
    if args.clean:
        for f in TMP.glob("*.pdf"):
            f.unlink()

    queue   = json.loads(QUEUE.read_text())
    results = acquire(queue)
    REPORT.write_text(json.dumps({"results": results}, indent=2))

    _print_summary(results)

    sys.stdout.flush()
    subprocess.run([sys.executable, str(INGEST)], cwd=str(ROOT))
    subprocess.run([sys.executable, str(CHECK)], cwd=str(ROOT))


if __name__ == "__main__":
    main()
