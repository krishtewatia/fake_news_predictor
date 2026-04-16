"""
Credibility Scorer Module
-------------------------
Assigns trust scores to evidence based on source domain reputation.
"""

import tldextract

# Domain trust tiers
HIGH_TRUST_DOMAINS = {
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "nytimes.com",
}

MEDIUM_TRUST_DOMAINS = {
    "ndtv.com",
    "thehindu.com",
    "hindustantimes.com",
}

HIGH_TRUST_SCORE = 0.9
MEDIUM_TRUST_SCORE = 0.6
DEFAULT_TRUST_SCORE = 0.3


def extract_domain(url: str) -> str:
    """
    Extract the registered domain from a URL using tldextract.

    Args:
        url: Full URL string.

    Returns:
        Registered domain (e.g., 'bbc.com').
    """
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}"
    return domain.lower()


def get_credibility(domain: str) -> float:
    """
    Get the credibility score for a given domain.

    Args:
        domain: Registered domain string.

    Returns:
        Trust score — 0.9 (high), 0.6 (medium), or 0.3 (default).
    """
    domain = domain.lower()

    if domain in HIGH_TRUST_DOMAINS:
        return HIGH_TRUST_SCORE
    elif domain in MEDIUM_TRUST_DOMAINS:
        return MEDIUM_TRUST_SCORE
    else:
        return DEFAULT_TRUST_SCORE


def attach_credibility(evidence: list[dict]) -> list[dict]:
    """
    Attach credibility scores to each evidence item based on its domain.

    Args:
        evidence: List of evidence dicts, each containing 'url' or 'domain'.

    Returns:
        The same evidence list with 'credibility_score' added to each item.
    """
    if not evidence:
        return evidence

    for item in evidence:
        domain = item.get("domain", "")

        # If domain is missing, try extracting from URL
        if not domain:
            url = item.get("url", "")
            domain = extract_domain(url) if url else ""
            item["domain"] = domain

        item["credibility_score"] = get_credibility(domain)

    return evidence
