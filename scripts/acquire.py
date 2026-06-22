#!/usr/bin/env python3
"""
acquire.py — download papers listed in search_queue.json.

Usage:
    python scripts/acquire.py

Input:  search_queue.json   list of paper requests (see format below)
Output: tmp/                downloaded PDFs land here
        results.json        one entry per paper: found / not_found / download_failed

search_queue.json format:
[
  { "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" },
  { "title": "SimCSE ...", "arxiv_id": "2104.08821" },
  { "title": "Some ACL paper ...", "doi": "10.18653/v1/..." }
]

Only "title" is required. Providing arxiv_id or doi makes acquisition faster and
more reliable — the script will skip fuzzy search and go straight to the source.

Source chain (tried in order until a PDF is found):
  1. arXiv          — direct ID lookup or title search; exponential backoff on 429
  2. ACL Anthology  — title search; covers ACL/EMNLP/NAACL/EACL/etc.
  3. Semantic Scholar — DOI or title search; requires openAccessPdf
  4. Unpaywall      — DOI-based open-access lookup (requires doi in entry)
  5. Web search     — DuckDuckGo fallback; scans first results for PDF links
                      on known academic domains
"""

import json
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    import arxiv as arxiv_lib
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False
    print("Warning: 'arxiv' package not installed. Install with: pip install arxiv")

try:
    import fitz  # pymupdf
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Paths  (all relative to papers_bank/, which is this script's parent)
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).parent.parent
QUEUE   = ROOT / "search_queue.json"
PAPERS  = ROOT / "papers"
REPORTS = ROOT / "reports"
REPORT  = REPORTS / "acquire_report.json"
TMP     = ROOT / "tmp"
CHECK   = ROOT / "scripts" / "check.py"

# Load email for Unpaywall from config.json if present
_cfg_path = ROOT / "config.json"
try:
    _UNPAYWALL_EMAIL = json.loads(_cfg_path.read_text())["zotero"]["email"]
except Exception:
    _UNPAYWALL_EMAIL = "user@example.com"

# Known academic domains the web-search fallback trusts for PDF links
ACADEMIC_DOMAINS = [
    "arxiv.org",
    "aclanthology.org",
    "openreview.net",
    "semanticscholar.org",
    "aclweb.org",
    "nlp.stanford.edu",
    "cs.cornell.edu",
    "proceedings.mlr.press",
    "papers.nips.cc",
    "neurips.cc",
    # Biomedical
    "ncbi.nlm.nih.gov",
    "europepmc.org",
    "medrxiv.org",
    "biorxiv.org",
]

HEADERS = {"User-Agent": "paper-acquire/1.0 (research pipeline; contact via github)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLUG_STOP = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to",
    "with", "by", "via", "from", "at", "as", "is", "are", "its",
    "using", "based", "towards", "toward",
}

def slugify(title: str, n: int = 6) -> str:
    """Derive a stable paper ID from a title.

    Lowercases, strips punctuation, removes common stop words, then takes the
    first *n* content words joined by underscores.  Produces human-readable IDs
    that are immediately recognisable when browsing papers/ or pdfs/.
    """
    words = re.sub(r"[^a-z0-9\s]", "", title.lower()).split()
    content = [w for w in words if w not in _SLUG_STOP]
    return "_".join(content[:n])


def download_pdf(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=30, stream=True, headers=HEADERS)
        r.raise_for_status()
        content = r.content
        if content[:4] != b"%PDF":
            return False
        dest.write_bytes(content)
        return True
    except Exception as e:
        print(f"    download failed: {e}")
        return False


PDF_CONTENT_THRESHOLD = 0.7


def _extract_pdf_title(doc) -> str:
    """
    Extract the paper title from an open fitz document.
    Strategy:
      1. PDF metadata — fast, accurate when present.
      2. Largest-font spans on page 1 — works for most typeset papers.
      3. Empty string if both fail (caller will choose a fallback).
    """
    # --- 1. PDF metadata ---
    meta = (doc.metadata or {}).get("title", "").strip()
    # Some tools set metadata title to junk like "Microsoft Word - paper.docx"
    if meta and len(meta) > 8 and not re.search(r"\.(docx?|tex|odt)\b", meta, re.I):
        return meta

    # --- 2. Font-size analysis on page 1 ---
    page = doc[0]
    page_height = page.rect.height
    page_width  = page.rect.width

    # Patterns that are definitely not the title
    _NOISE = re.compile(
        r"arxiv:\d{4}\.\d{4,5}"           # arXiv stamp
        r"|^\d{4}\.\d{4,5}(v\d+)?$"       # bare arXiv ID
        r"|\[cs\.|preprint|under review"   # submission metadata
        r"|proceedings of|conference on"   # venue headers (caught later anyway)
        r"|^\d+$",                         # lone page numbers
        re.IGNORECASE,
    )

    spans = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:          # skip image blocks
            continue
        # Ignore anything in the bottom 40% of the page (footers, page numbers)
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

    max_size = max(s for s, _ in spans)
    # Collect all spans within 15% of the largest font — titles often span
    # multiple lines and may have slightly inconsistent sizing
    title_parts = [t for s, t in spans if s >= max_size * 0.85]

    # Cap at 10 spans; beyond that we're probably picking up author names
    return " ".join(title_parts[:10])


def verify_pdf_content(pdf_path: Path, query_title: str) -> tuple[bool, float]:
    """
    Verify that a downloaded PDF actually contains the paper we asked for.
    Returns (passed, score) where score is the title_similarity value in [0,1]
    or -1 if the check could not be performed (pymupdf unavailable).

    Method:
      - Extract the PDF's title via metadata or largest-font text on page 1.
      - Run title_similarity between the extracted title and the query.
    Fallback (scanned / unreadable PDF):
      - If no title can be extracted, check word recall across first-page text.
        This is weaker but avoids discarding valid scanned papers outright.
    """
    if not HAS_FITZ:
        return True, -1.0

    try:
        doc = fitz.open(str(pdf_path))
        extracted = _extract_pdf_title(doc)

        if extracted:
            doc.close()
            score = title_similarity(query_title, extracted)
            if score < PDF_CONTENT_THRESHOLD:
                print(f"    PDF title: '{extracted[:80]}'")
            return score >= PDF_CONTENT_THRESHOLD, score

        # Fallback: scanned or metadata-free PDF — word recall on first page
        text = doc[0].get_text()
        doc.close()

        text_words  = set(normalise_title(text).split())
        query_words = {w for w in normalise_title(query_title).split() if w not in _STOP}
        if not query_words:
            return True, 1.0
        recall = len(query_words & text_words) / len(query_words)
        print(f"    PDF title not extractable, using word recall (score={recall:.2f})")
        return recall >= PDF_CONTENT_THRESHOLD, recall

    except Exception as e:
        print(f"    PDF read error: {e}")
        return False, 0.0


def normalise_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)   # replace punctuation with space, not delete
    return re.sub(r"\s+", " ", t).strip()


def sanitize_for_search(title: str) -> str:
    """Convert Unicode punctuation to plain ASCII before building search queries.

    External APIs (arXiv, DuckDuckGo) often choke on Unicode dashes, curly
    quotes, and other typographic characters when they appear inside quoted
    phrase searches.  This function replaces them with their nearest ASCII
    equivalent so the query reaches the API cleanly.
    """
    import unicodedata
    title = unicodedata.normalize("NFC", title)
    title = re.sub(r"[–—−‐]", "-", title)   # all dash variants → hyphen
    title = re.sub(r"[''`]",   "'", title)   # curly/backtick quotes → straight
    title = re.sub(r'[""]',    '"', title)   # curly double quotes → straight
    title = title.encode("ascii", "ignore").decode("ascii")  # drop any remaining non-ASCII
    return title.strip()


# Stop-words to exclude from title similarity (they inflate scores trivially)
_STOP = {"a", "an", "the", "of", "in", "on", "for", "and", "or", "to",
         "is", "are", "with", "by", "via", "from", "at", "as", "its"}

def title_similarity(query: str, result: str) -> float:
    """
    Word-overlap F1 between query and result titles (ignoring stop-words).
    Returns a value in [0, 1]. A score below ~0.5 almost certainly means
    the search returned the wrong paper.
    """
    q_words = {w for w in normalise_title(query).split() if w not in _STOP}
    r_words = {w for w in normalise_title(result).split() if w not in _STOP}
    if not q_words or not r_words:
        return 0.0
    overlap  = q_words & r_words
    precision = len(overlap) / len(r_words)
    recall    = len(overlap) / len(q_words)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)  # F1


TITLE_SIMILARITY_THRESHOLD = 0.65  # reject results below this score


# ---------------------------------------------------------------------------
# Source 1: arXiv
# ---------------------------------------------------------------------------

def try_arxiv(paper: dict) -> dict | None:
    # Fast path: when the queue entry already has an arxiv_id, build the direct
    # PDF URL without touching the API.  The Python arxiv library hits the same
    # rate-limited endpoint for every paper in sequence, causing HTTP 429 failures
    # on runs with more than a handful of papers.  Direct download bypasses this
    # entirely; verify_pdf_content will confirm we got the right file.
    arxiv_id = (paper.get("arxiv_id") or "").split("v")[0]
    if arxiv_id:
        return {
            "title":    paper["title"],
            "authors":  [],
            "year":     None,
            "venue":    "arXiv",
            "abstract": "",
            "arxiv_id": arxiv_id,
            "pdf_url":  f"https://arxiv.org/pdf/{arxiv_id}",
            "source":   "arxiv",
        }

    # Slow path: title-based search via the API (only when no ID is known)
    if not HAS_ARXIV:
        return None

    backoff      = 10  # seconds to wait after a 429 before one retry
    client       = arxiv_lib.Client(delay_seconds=5.0, num_retries=1)
    clean_title  = sanitize_for_search(paper["title"])

    def _arxiv_search(query: str) -> list:
        for attempt in range(2):
            try:
                return list(client.results(arxiv_lib.Search(query=query, max_results=3)))
            except arxiv_lib.HTTPError as e:
                if e.status == 429 and attempt == 0:
                    print(f"    arXiv 429 — waiting {backoff}s then retrying...")
                    time.sleep(backoff)
                else:
                    print(f"    arXiv error ({e.status}), skipping")
                    return []
            except Exception as e:
                print(f"    arXiv unexpected error: {e}, skipping")
                return []
        return []

    # Pass 1: exact phrase search (fast, precise)
    hits = _arxiv_search(f'ti:"{clean_title}"')

    # Pass 2: keyword search — catches titles where the phrase match fails due to
    # word order variation, punctuation differences, or subtitle truncation
    if not hits:
        keywords = [w for w in clean_title.split() if w.lower() not in _SLUG_STOP and len(w) > 2]
        if keywords:
            kw_query = " ".join(f"ti:{w}" for w in keywords[:8])
            hits = _arxiv_search(kw_query)

    if not hits:
        return None

    # Pick the best-matching result across both passes
    best, best_score = None, 0.0
    for p in hits:
        score = title_similarity(paper["title"], p.title)
        if score > best_score:
            best, best_score = p, score

    if best_score < TITLE_SIMILARITY_THRESHOLD:
        print(f"    arXiv title mismatch (score={best_score:.2f}): '{best.title}'")
        return None

    p = best

    return {
        "title":    p.title,
        "authors":  [a.name for a in p.authors],
        "year":     p.published.year,
        "venue":    "arXiv",
        "abstract": p.summary.replace("\n", " "),
        "arxiv_id": p.entry_id.split("/")[-1].split("v")[0],
        "pdf_url":  p.pdf_url,
        "source":   "arxiv",
    }


# ---------------------------------------------------------------------------
# Source 2: ACL Anthology
# ---------------------------------------------------------------------------

ACL_SEARCH  = "https://aclanthology.org/search/?q={query}"
ACL_BASE    = "https://aclanthology.org"
# Matches a paper ID as the full href value: /2023.acl-main.42/ or /P19-1001/
ACL_HREF_RE = re.compile(r'^/(\d{4}\.[a-z\-]+\.\d+|[A-Z]\d{2}-\d{4})/?$')


def try_acl_anthology(paper: dict) -> dict | None:
    query = urllib.parse.quote_plus(paper["title"])
    try:
        r = requests.get(ACL_SEARCH.format(query=query), timeout=15, headers=HEADERS)
        r.raise_for_status()
    except Exception as e:
        print(f"    ACL Anthology request failed: {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # Walk <a> tags whose href looks like a paper ID (not a full-text regex over the
    # entire HTML, which picks up sidebar / "popular papers" entries before the real
    # results).  Check the link text for title similarity so we skip unrelated hits.
    paper_id = None
    for a_tag in soup.find_all("a", href=ACL_HREF_RE):
        m = ACL_HREF_RE.match(a_tag.get("href", ""))
        if not m:
            continue
        candidate_id    = m.group(1)
        candidate_title = a_tag.get_text(strip=True)
        if not candidate_title:
            continue
        score = title_similarity(paper["title"], candidate_title)
        if score >= TITLE_SIMILARITY_THRESHOLD:
            paper_id = candidate_id
            break
        # log only when something was found but rejected
        print(f"    ACL candidate skipped (score={score:.2f}): '{candidate_title[:60]}'")

    if paper_id is None:
        return None

    pdf_url = f"{ACL_BASE}/{paper_id}.pdf"

    # Fetch the paper page for metadata
    try:
        meta_r = requests.get(f"{ACL_BASE}/{paper_id}/", timeout=10, headers=HEADERS)
        meta_r.raise_for_status()
        soup2 = BeautifulSoup(meta_r.text, "html.parser")

        title_tag = soup2.find("h2", id="title") or soup2.find("title")
        title = title_tag.get_text(strip=True) if title_tag else paper["title"]
        title = re.sub(r"\s*\|.*$", "", title).strip()  # strip " | ACL Anthology" suffix

        authors = [a.get_text(strip=True) for a in soup2.select("p.lead a")]

        year_m = re.search(r"\b(19|20)\d{2}\b", paper_id)
        year = int(year_m.group()) if year_m else None

        # Abstract
        abs_tag = soup2.find("div", class_="acl-abstract")
        abstract = abs_tag.get_text(strip=True) if abs_tag else ""

        # Venue from paper ID prefix
        venue_part = paper_id.split(".")[1] if "." in paper_id else paper_id
        venue = venue_part.upper()

    except Exception:
        # Metadata fetch failed — we already verified title on the search page
        title    = paper["title"]
        authors  = []
        year     = None
        abstract = ""
        venue    = "ACL Anthology"

    return {
        "title":    title,
        "authors":  authors,
        "year":     year,
        "venue":    venue,
        "abstract": abstract,
        "acl_id":   paper_id,
        "pdf_url":  pdf_url,
        "source":   "acl_anthology",
    }


# ---------------------------------------------------------------------------
# Source 3: Semantic Scholar
# ---------------------------------------------------------------------------

SS_BASE   = "https://api.semanticscholar.org/graph/v1/paper"
SS_FIELDS = "title,authors,year,venue,abstract,externalIds,openAccessPdf"


def try_semantic_scholar(paper: dict) -> dict | None:
    # DOI direct lookup — the DOI is a specific identifier so we trust the result
    if paper.get("doi"):
        r = requests.get(f"{SS_BASE}/DOI:{paper['doi']}",
                         params={"fields": SS_FIELDS}, timeout=10)
        if r.ok:
            return _ss_parse(r.json())

    # Title search — verify similarity before accepting any result
    r = requests.get(SS_BASE + "/search",
                     params={"query": paper["title"], "fields": SS_FIELDS, "limit": 3},
                     timeout=10)
    if r.ok:
        for item in r.json().get("data", []):
            result_title = item.get("title", "")
            score = title_similarity(paper["title"], result_title)
            if score >= TITLE_SIMILARITY_THRESHOLD:
                return _ss_parse(item)
            print(f"    SS title mismatch (score={score:.2f}): '{result_title[:60]}'")

    return None


def _ss_parse(p: dict) -> dict | None:
    pdf_url = (p.get("openAccessPdf") or {}).get("url")
    if not pdf_url:
        return None
    ext = p.get("externalIds") or {}
    return {
        "title":    p.get("title", ""),
        "authors":  [a["name"] for a in p.get("authors", [])],
        "year":     p.get("year"),
        "venue":    p.get("venue", ""),
        "abstract": (p.get("abstract") or "").replace("\n", " "),
        "arxiv_id": ext.get("ArXiv"),
        "doi":      ext.get("DOI"),
        "pdf_url":  pdf_url,
        "source":   "semantic_scholar",
    }


# ---------------------------------------------------------------------------
# Source 4: Unpaywall (DOI required)
# ---------------------------------------------------------------------------

def try_unpaywall(paper: dict) -> dict | None:
    doi = paper.get("doi", "").strip()
    if not doi:
        return None
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": _UNPAYWALL_EMAIL},
            timeout=10,
        )
        if not r.ok:
            return None
        data = r.json()
        loc  = data.get("best_oa_location")
        if not loc:
            return None
        pdf_url = loc.get("url_for_pdf") or loc.get("url")
        if not pdf_url:
            return None
        return {
            "title":    data.get("title", paper["title"]),
            "authors":  [a.get("name", "") for a in (data.get("z_authors") or [])],
            "year":     data.get("year"),
            "venue":    data.get("journal_name", ""),
            "abstract": "",
            "doi":      doi,
            "pdf_url":  pdf_url,
            "source":   "unpaywall",
        }
    except Exception as e:
        print(f"    Unpaywall error: {e}")
        return None


# ---------------------------------------------------------------------------
# Source 5: PubMed Central (biomedical open-access archive)
# ---------------------------------------------------------------------------

import xml.etree.ElementTree as ET

PMC_SEARCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PMC_SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PMC_OA      = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"


def try_pubmed_central(paper: dict) -> dict | None:
    """Search PubMed Central for open-access biomedical papers.

    Uses the NCBI E-utilities API (no key required; rate-limited to ~3 req/s).
    PDF URLs are resolved via the PMC Open Access service, which only returns
    links for articles that are genuinely open-access — no paywalled hits.
    """
    try:
        # Title search in PMC
        r = requests.get(PMC_SEARCH, params={
            "db":      "pmc",
            "term":    f'"{paper["title"]}"[Title]',
            "retmode": "json",
            "retmax":  3,
        }, timeout=10, headers=HEADERS)
        if not r.ok:
            return None

        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None

        # Verify title similarity on the first hit
        pmcid = ids[0]
        sum_r = requests.get(PMC_SUMMARY, params={
            "db": "pmc", "id": pmcid, "retmode": "json",
        }, timeout=10, headers=HEADERS)
        if not sum_r.ok:
            return None

        doc          = sum_r.json().get("result", {}).get(pmcid, {})
        result_title = doc.get("title", "")
        score        = title_similarity(paper["title"], result_title)
        if score < TITLE_SIMILARITY_THRESHOLD:
            print(f"    PMC title mismatch (score={score:.2f}): '{result_title[:60]}'")
            return None

        # Resolve PDF via the OA service (returns XML with ftp/https links)
        oa_r = requests.get(PMC_OA, params={"id": f"PMC{pmcid}"}, timeout=10, headers=HEADERS)
        if not oa_r.ok:
            return None

        root    = ET.fromstring(oa_r.text)
        pdf_url = None
        for link in root.iter("link"):
            if link.get("format") == "pdf":
                href = link.get("href", "")
                # ftp:// links work fine but some clients prefer https://
                if href.startswith("ftp://"):
                    href = href.replace("ftp://", "https://", 1)
                pdf_url = href
                break

        if not pdf_url:
            return None

        year_m  = re.search(r"\b(19|20)\d{2}\b", doc.get("pubdate", ""))
        authors = [a.get("name", "") for a in doc.get("authors", [])]

        return {
            "title":    result_title or paper["title"],
            "authors":  authors,
            "year":     int(year_m.group()) if year_m else None,
            "venue":    doc.get("source", ""),
            "abstract": "",
            "pdf_url":  pdf_url,
            "source":   "pubmed_central",
        }

    except Exception as e:
        print(f"    PMC error: {e}")
        return None


# ---------------------------------------------------------------------------
# Source 6: Europe PMC (broader biomedical coverage, direct PDF links)
# ---------------------------------------------------------------------------

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def try_europe_pmc(paper: dict) -> dict | None:
    """Search Europe PubMed Central.

    Europe PMC indexes ~40 M biomedical records including PubMed, PMC, preprints,
    and clinical trial records. The API returns structured PDF links with an
    explicit open-access flag, making it easy to filter for downloadable files.
    """
    try:
        r = requests.get(EPMC_SEARCH, params={
            "query":      f'TITLE:"{paper["title"]}"',
            "format":     "json",
            "resultType": "core",
            "pageSize":   3,
        }, timeout=10, headers=HEADERS)
        if not r.ok:
            return None

        for item in r.json().get("resultList", {}).get("result", []):
            result_title = item.get("title", "").rstrip(".")
            score        = title_similarity(paper["title"], result_title)
            if score < TITLE_SIMILARITY_THRESHOLD:
                print(f"    EPMC title mismatch (score={score:.2f}): '{result_title[:60]}'")
                continue

            # Walk fullTextUrlList for an open-access PDF
            pdf_url = None
            for url_obj in item.get("fullTextUrlList", {}).get("fullTextUrl", []):
                if (url_obj.get("documentStyle") == "pdf"
                        and url_obj.get("availability") in ("Open access", "Free")):
                    pdf_url = url_obj.get("url")
                    break

            if not pdf_url:
                continue

            year    = item.get("pubYear")
            authors = [
                f"{a.get('lastName', '')}, {a.get('initials', '')}".strip(", ")
                for a in item.get("authorList", {}).get("author", [])
            ]

            return {
                "title":    result_title or paper["title"],
                "authors":  authors,
                "year":     int(year) if year else None,
                "venue":    item.get("journalTitle", ""),
                "abstract": item.get("abstractText", ""),
                "doi":      item.get("doi"),
                "pdf_url":  pdf_url,
                "source":   "europe_pmc",
            }

        return None

    except Exception as e:
        print(f"    Europe PMC error: {e}")
        return None


# ---------------------------------------------------------------------------
# Source 7: Web search fallback (DuckDuckGo)
# ---------------------------------------------------------------------------

DDG_URL = "https://html.duckduckgo.com/html/"


def try_web_search(paper: dict) -> dict | None:
    clean_title = sanitize_for_search(paper["title"])
    query = f'"{clean_title}" filetype:pdf OR site:arxiv.org OR site:aclanthology.org OR site:openreview.net'
    try:
        r = requests.post(
            DDG_URL,
            data={"q": query, "kl": "us-en"},
            headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"    web search request failed: {e}")
        return None

    soup  = BeautifulSoup(r.text, "html.parser")
    links = soup.select("a.result__url, a.result__a")

    for link in links[:10]:
        href = link.get("href", "")
        # DuckDuckGo wraps URLs — unwrap if needed
        if "uddg=" in href:
            m = re.search(r"uddg=([^&]+)", href)
            if m:
                href = urllib.parse.unquote(m.group(1))

        if not href.startswith("http"):
            continue

        # Only trust known academic domains
        domain = urllib.parse.urlparse(href).netloc.lstrip("www.")
        if not any(href.lower().endswith(".pdf") or d in domain for d in ACADEMIC_DOMAINS):
            continue

        # Convert known landing-page patterns to direct PDF URLs
        href = _coerce_to_pdf_url(href)
        if not href:
            continue

        print(f"    web search hit: {href}")
        return {
            "title":    paper["title"],
            "authors":  [],
            "year":     None,
            "venue":    "",
            "abstract": "",
            "pdf_url":  href,
            "source":   "web_search",
        }

    return None


def _coerce_to_pdf_url(url: str) -> str | None:
    """Convert landing-page URLs to direct PDF links for known sites."""
    # arXiv abs page → PDF
    m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"

    # ACL Anthology paper page → PDF
    m = re.search(r"aclanthology\.org/(\S+?)/?$", url)
    if m:
        pid = m.group(1).rstrip("/")
        if not pid.endswith(".pdf"):
            return f"https://aclanthology.org/{pid}.pdf"
        return f"https://aclanthology.org/{pid}"

    # OpenReview forum → PDF
    m = re.search(r"openreview\.net/forum\?id=(\S+)", url)
    if m:
        return f"https://openreview.net/pdf?id={m.group(1)}"

    # Already a PDF
    if url.lower().endswith(".pdf"):
        return url

    return None


# ---------------------------------------------------------------------------
# Bank index — deduplication against existing papers
# ---------------------------------------------------------------------------

def load_bank_index() -> dict:
    """
    Parse every *_summary.md and return a lookup dict keyed by:
      - normalised title  →  {id, year, arxiv_id}
      - "arxiv:{id}"      →  {id, year, arxiv_id}   (when arxiv_id present)

    Used to skip papers already in the bank and to detect genuine new versions.
    Falls back to an empty dict if PyYAML is not installed.
    """
    index: dict = {}
    if not HAS_YAML or not PAPERS.exists():
        return index

    for p in PAPERS.glob("*_summary.md"):
        text = p.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        end = text.find("\n---", 3)
        if end == -1:
            continue
        try:
            fm = yaml.safe_load(text[3:end]) or {}
        except Exception:
            continue

        entry = {
            "id":       fm.get("id"),
            "year":     fm.get("year"),
            "arxiv_id": fm.get("arxiv_id"),
        }

        title = fm.get("title", "")
        if title:
            index[normalise_title(title)] = entry

        arxiv_id = fm.get("arxiv_id")
        if arxiv_id:
            index[f"arxiv:{str(arxiv_id).split('v')[0]}"] = entry

    return index


def already_in_bank(metadata: dict, bank: dict) -> dict | None:
    """
    Return the existing bank entry if this paper is already catalogued,
    or None if it is genuinely new.

    A paper is considered already present when:
      - Its normalised title matches an existing entry AND
        the source year is not strictly newer than the bank year, OR
      - Its arXiv ID matches an existing entry (version-independent).

    "Strictly newer" means: source year > bank year.  This lets a paper that
    was previously catalogued as a preprint be re-acquired once the peer-reviewed
    version with a different year appears.
    """
    # arXiv ID match — most reliable signal
    arxiv_id = (metadata.get("arxiv_id") or "").split("v")[0]
    if arxiv_id:
        existing = bank.get(f"arxiv:{arxiv_id}")
        if existing:
            return existing

    # Title match with year guard
    norm = normalise_title(metadata.get("title", ""))
    existing = bank.get(norm)
    if existing:
        bank_year   = existing.get("year")
        source_year = metadata.get("year")
        # Both years known and source is strictly newer → treat as new paper
        if bank_year and source_year and source_year > bank_year:
            return None
        return existing

    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

SOURCES = [
    try_arxiv,           # 1. arXiv            — CS / math / physics / quant-bio
    try_acl_anthology,   # 2. ACL Anthology     — NLP / CL venues
    try_semantic_scholar,# 3. Semantic Scholar  — broad; requires openAccessPdf
    try_pubmed_central,  # 4. PubMed Central    — biomedical open-access archive
    try_europe_pmc,      # 5. Europe PMC        — broader biomedical; direct PDF links
    try_unpaywall,       # 6. Unpaywall         — any field, DOI required
    try_web_search,      # 7. DuckDuckGo        — last-resort fallback
]


# ---------------------------------------------------------------------------
# Pre-flight check — batch dedup before any network calls
# ---------------------------------------------------------------------------

def preflight(queue: list, bank: dict) -> tuple[list, list, list]:
    """
    Classify the entire queue before touching any network source.

    Returns three lists:
      - skip_bank:  papers already catalogued in the bank; no work needed
      - skip_tmp:   papers whose PDF already exists in tmp/ with nonzero size;
                    need metadata lookup but NOT a new download
      - to_fetch:   genuinely new papers requiring both metadata lookup + download

    Bank check uses:
      1. arxiv_id from the queue entry (most reliable — no normalisation needed)
      2. Normalised title match against the bank index

    tmp/ check uses slugify(queue_title) as a heuristic. In the rare case the
    metadata title differs significantly from the queue title, the slug won't
    match and the paper will land in to_fetch, causing a harmless re-download.
    """
    skip_bank, skip_tmp, to_fetch = [], [], []

    for paper in queue:
        title    = paper["title"]
        arxiv_id = (paper.get("arxiv_id") or "").split("v")[0]

        # Bank check — arxiv_id first (exact), then normalised title
        existing = None
        if arxiv_id:
            existing = bank.get(f"arxiv:{arxiv_id}")
        if not existing:
            existing = bank.get(normalise_title(title))

        if existing:
            skip_bank.append((paper, existing))
            continue

        # tmp/ check
        slug     = slugify(title)
        pdf_path = TMP / f"{slug}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            skip_tmp.append(paper)
            continue

        to_fetch.append(paper)

    return skip_bank, skip_tmp, to_fetch


def acquire(queue: list) -> list:
    TMP.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    bank    = load_bank_index()
    results = []

    if bank:
        print(f"Bank index loaded: {len([k for k in bank if not k.startswith('arxiv:')])} papers")

    # -------------------------------------------------------------------------
    # Pre-flight: classify the whole queue before any network calls
    # -------------------------------------------------------------------------
    skip_bank, skip_tmp, to_fetch = preflight(queue, bank)

    print(f"\nPre-flight ({len(queue)} paper(s)):")
    print(f"  {len(skip_bank):3d} already in bank  — skipping entirely")
    print(f"  {len(skip_tmp):3d} already in tmp/  — will skip download, fetch metadata only")
    print(f"  {len(to_fetch):3d} new               — full fetch + download")

    # Emit results for papers already in the bank
    for paper, existing in skip_bank:
        print(f"  [bank]  {paper['title'][:70]}")
        results.append({
            "query":       paper["title"],
            "status":      "already_in_bank",
            "existing_id": existing.get("id"),
        })

    if not (skip_tmp or to_fetch):
        return results

    # -------------------------------------------------------------------------
    # Main loop: only runs for skip_tmp + to_fetch (i.e. nothing already in bank)
    # -------------------------------------------------------------------------
    in_tmp_titles = {p["title"] for p in skip_tmp}
    pending       = skip_tmp + to_fetch
    iterator      = tqdm(pending, desc="Acquiring", unit="paper") if HAS_TQDM else pending

    for paper in iterator:
        title   = paper["title"]
        in_tmp  = title in in_tmp_titles
        print(f"\n→ {title}")

        # Fetch metadata from sources
        metadata = None
        for source_fn in SOURCES:
            metadata = source_fn(paper)
            if metadata:
                print(f"  found via {metadata['source']}")
                break

        if not metadata:
            if in_tmp:
                # PDF already present but we can't confirm metadata — treat as found
                slug = slugify(title)
                print(f"  metadata not found; PDF already in tmp/ — reporting as found (verify manually)")
                results.append({
                    "query":    title,
                    "status":   "found",
                    "pdf":      f"tmp/{slug}.pdf",
                    "metadata": {"title": title, "authors": [], "year": None, "venue": ""},
                })
            else:
                print("  not found on any source")
                results.append({"query": title, "status": "not_found"})
            continue

        # Post-metadata bank check — catches edge cases where title normalisation
        # differs between queue entry and metadata (e.g. subtitle truncation)
        existing = already_in_bank(metadata, bank)
        if existing:
            print(f"  already in bank [{existing['id']}, {existing.get('year', '?')}] — skipping")
            results.append({
                "query":       title,
                "status":      "already_in_bank",
                "existing_id": existing["id"],
            })
            continue

        slug     = slugify(metadata["title"])
        pdf_path = TMP / f"{slug}.pdf"

        if in_tmp:
            # PDF already present — skip download, run content check only
            print(f"  PDF already in tmp/ — skipping download")
            passed, score = verify_pdf_content(pdf_path, title)
            if passed:
                print(f"  content check passed (score={score:.2f})" if score >= 0 else "  content check skipped (install pymupdf to enable)")
                results.append({
                    "query":         title,
                    "status":        "found",
                    "pdf":           f"tmp/{slug}.pdf",
                    "content_score": score,
                    "metadata":      metadata,
                })
                continue
            else:
                # Content mismatch on existing file — fall through to re-download
                print(f"  content mismatch on existing file (score={score:.2f}) — re-downloading")

        # Download
        print(f"  downloading → tmp/{slug}.pdf")
        if download_pdf(metadata["pdf_url"], pdf_path):
            passed, score = verify_pdf_content(pdf_path, title)
            if passed:
                print(f"  content check passed (score={score:.2f})" if score >= 0 else "  content check skipped (install pymupdf to enable)")
                results.append({
                    "query":         title,
                    "status":        "found",
                    "pdf":           f"tmp/{slug}.pdf",
                    "content_score": score,
                    "metadata":      metadata,
                })
            else:
                print(f"  content mismatch (score={score:.2f}) — discarding PDF")
                pdf_path.unlink(missing_ok=True)
                results.append({
                    "query":         title,
                    "status":        "content_mismatch",
                    "content_score": score,
                    "metadata":      metadata,
                })
        else:
            results.append({
                "query":    title,
                "status":   "download_failed",
                "metadata": metadata,
            })

        time.sleep(3)   # be polite to APIs

    return results


def main():
    if not QUEUE.exists():
        print(f"No search_queue.json found at {QUEUE}")
        print('Create one, e.g.: [{"title": "Attention Is All You Need"}]')
        return

    queue = json.loads(QUEUE.read_text())
    print(f"Processing {len(queue)} paper(s)...")

    results = acquire(queue)
    REPORT.write_text(json.dumps({"results": results}, indent=2))

    found    = sum(1 for r in results if r["status"] == "found")
    skipped  = sum(1 for r in results if r["status"] == "already_in_bank")
    mismatch = sum(1 for r in results if r["status"] == "content_mismatch")
    print(f"\nDone: {found}/{len(results)} downloaded → tmp/   details in reports/acquire_report.json")
    if skipped:
        print(f"  ({skipped} already in bank — skipped)")
    if mismatch:
        print(f"  ({mismatch} discarded due to content mismatch)")

    # Run consistency check
    if CHECK.exists():
        print("\nRunning check.py...")
        subprocess.run([sys.executable, str(CHECK)], cwd=str(ROOT))


if __name__ == "__main__":
    main()
