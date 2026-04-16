"""
Evidence Collector Module
-------------------------
Collects evidence text from search results.
Supports fast mode (snippets) and slow mode (full article scraping via newspaper3k).
"""

import re

from newspaper import Article, ArticleException


# Boilerplate patterns to strip from scraped articles
BOILERPLATE_PATTERNS = [
    r"(?i)subscribe\s+to\s+our\s+newsletter.*",
    r"(?i)sign\s+up\s+for\s+.*",
    r"(?i)follow\s+us\s+on\s+.*",
    r"(?i)share\s+this\s+article.*",
    r"(?i)read\s+more:.*",
    r"(?i)advertisement\s*\.?",
    r"(?i)sponsored\s+content.*",
    r"(?i)all\s+rights\s+reserved.*",
    r"(?i)copyright\s+©?\s*\d{4}.*",
    r"(?i)cookie\s+policy.*",
]


def clean_evidence_text(text: str) -> str:
    """
    Clean evidence text by removing boilerplate, extra whitespace, and noise.

    Args:
        text: Raw evidence text.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Remove boilerplate patterns
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text)

    # Remove extra whitespace and normalize
    text = re.sub(r"\s+", " ", text).strip()

    return text


def scrape_article(url: str) -> str:
    """
    Scrape full article text from a URL using newspaper3k.

    Args:
        url: The article URL to scrape.

    Returns:
        Extracted article text, or empty string on failure.
    """
    try:
        article = Article(url, request_timeout=15)
        article.download()
        article.parse()

        text = article.text
        if text and len(text.strip()) > 50:
            return text.strip()

        return ""

    except ArticleException as e:
        print(f"[EvidenceCollector] Article parse error for {url}: {e}")
        return ""
    except Exception as e:
        print(f"[EvidenceCollector] Scrape failed for {url}: {e}")
        return ""


def collect_single_evidence(result: dict, mode: str) -> dict | None:
    """
    Collect evidence from a single search result.

    Args:
        result: Search result dict with url, domain, title, snippet.
        mode:   'fast' (use snippet) or 'slow' (scrape full article).

    Returns:
        Evidence dict with url, domain, title, text — or None if no text available.
    """
    url = result.get("url", "")
    domain = result.get("domain", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")

    if not url:
        return None

    evidence_text = ""

    if mode == "slow":
        # Attempt full article scrape
        evidence_text = scrape_article(url)

        if not evidence_text:
            # Fallback to snippet
            print(f"[EvidenceCollector] Scrape failed, using snippet for: {url[:60]}")
            evidence_text = snippet
    else:
        # Fast mode — use snippet directly
        evidence_text = snippet

    # Clean the text
    evidence_text = clean_evidence_text(evidence_text)

    if not evidence_text:
        return None

    return {
        "url": url,
        "domain": domain,
        "title": title,
        "text": evidence_text,
    }


def collect_evidence(results: list[dict], mode: str = "fast") -> list[dict]:
    """
    Collect evidence text from search results.

    Args:
        results: List of search result dicts from web_search module.
        mode:    'fast' — use snippets (default).
                 'slow' — scrape full articles via newspaper3k, fallback to snippet.

    Returns:
        List of evidence dicts with keys: url, domain, title, text.
    """
    if not results:
        return []

    if mode not in ("fast", "slow"):
        print(f"[EvidenceCollector] Unknown mode '{mode}', defaulting to 'fast'.")
        mode = "fast"

    evidence_list = []

    for result in results:
        evidence = collect_single_evidence(result, mode)
        if evidence:
            evidence_list.append(evidence)

    print(f"[EvidenceCollector] Collected {len(evidence_list)} evidence items ({mode} mode).")
    return evidence_list
