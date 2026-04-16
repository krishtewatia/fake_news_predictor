"""
Web Search Module
-----------------
Searches for evidence across NewsAPI and SerpAPI.
Deduplicates results, enforces domain diversity, and returns structured data.
"""

import time
from collections import defaultdict

import requests
import tldextract

from config import NEWSAPI_KEY, SERPAPI_KEY

# Constants
MAX_RESULTS_PER_SOURCE = 5
MAX_RESULTS_PER_DOMAIN = 2
REQUEST_TIMEOUT = 15

# API Endpoints
NEWSAPI_URL = "https://newsapi.org/v2/everything"
SERPAPI_URL = "https://serpapi.com/search"


def extract_domain(url: str) -> str:
    """
    Extract the registered domain from a URL using tldextract.

    Args:
        url: Full URL string.

    Returns:
        Registered domain (e.g., 'bbc.co.uk').
    """
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}"
    return domain.lower()


def search_newsapi(query: str) -> list[dict]:
    """
    Fetch search results from NewsAPI.

    Args:
        query: Search query string.

    Returns:
        List of result dicts with title, url, domain, snippet, published_date.
    """
    if not NEWSAPI_KEY:
        print("[WebSearch] NewsAPI key not configured, skipping.")
        return []

    params = {
        "q": query,
        "pageSize": MAX_RESULTS_PER_SOURCE,
        "sortBy": "relevancy",
        "language": "en",
        "apiKey": NEWSAPI_KEY,
    }

    try:
        response = requests.get(NEWSAPI_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            print(f"[WebSearch] NewsAPI error: {data.get('message', 'Unknown error')}")
            return []

        results = []
        for article in data.get("articles", [])[:MAX_RESULTS_PER_SOURCE]:
            url = article.get("url", "")
            if not url:
                continue

            results.append({
                "title": article.get("title", ""),
                "url": url,
                "domain": extract_domain(url),
                "snippet": article.get("description", "") or article.get("content", ""),
                "published_date": article.get("publishedAt", ""),
            })

        return results

    except requests.exceptions.Timeout:
        print(f"[WebSearch] NewsAPI timeout for query: {query[:50]}")
        return []
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("[WebSearch] NewsAPI rate limit hit, backing off.")
            time.sleep(2)
        else:
            print(f"[WebSearch] NewsAPI HTTP error: {e}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"[WebSearch] NewsAPI request failed: {e}")
        return []


def search_serpapi(query: str) -> list[dict]:
    """
    Fetch search results from SerpAPI.

    Args:
        query: Search query string.

    Returns:
        List of result dicts with title, url, domain, snippet, published_date.
    """
    if not SERPAPI_KEY:
        print("[WebSearch] SerpAPI key not configured, skipping.")
        return []

    params = {
        "q": query,
        "num": MAX_RESULTS_PER_SOURCE,
        "engine": "google",
        "api_key": SERPAPI_KEY,
    }

    try:
        response = requests.get(SERPAPI_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("organic_results", [])[:MAX_RESULTS_PER_SOURCE]:
            url = item.get("link", "")
            if not url:
                continue

            results.append({
                "title": item.get("title", ""),
                "url": url,
                "domain": extract_domain(url),
                "snippet": item.get("snippet", ""),
                "published_date": item.get("date", ""),
            })

        return results

    except requests.exceptions.Timeout:
        print(f"[WebSearch] SerpAPI timeout for query: {query[:50]}")
        return []
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("[WebSearch] SerpAPI rate limit hit, backing off.")
            time.sleep(2)
        else:
            print(f"[WebSearch] SerpAPI HTTP error: {e}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"[WebSearch] SerpAPI request failed: {e}")
        return []


def deduplicate_results(results: list[dict]) -> list[dict]:
    """
    Remove duplicate results based on URL.

    Args:
        results: List of result dicts.

    Returns:
        Deduplicated list of result dicts.
    """
    seen_urls = set()
    unique = []

    for result in results:
        url = result.get("url", "").rstrip("/").lower()
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(result)

    return unique


def enforce_domain_diversity(results: list[dict]) -> list[dict]:
    """
    Limit results to a maximum number per domain.

    Args:
        results: Deduplicated list of result dicts.

    Returns:
        Filtered list with max MAX_RESULTS_PER_DOMAIN results per domain.
    """
    domain_counts = defaultdict(int)
    diverse = []

    for result in results:
        domain = result.get("domain", "")
        if domain_counts[domain] < MAX_RESULTS_PER_DOMAIN:
            diverse.append(result)
            domain_counts[domain] += 1

    return diverse


def search_web(queries: list[str]) -> list[dict]:
    """
    Search for evidence across NewsAPI and SerpAPI for a list of queries.
    Deduplicates results and enforces domain diversity.

    Args:
        queries: List of search query strings.

    Returns:
        List of structured result dicts with keys:
            title, url, domain, snippet, published_date
    """
    if not queries:
        return []

    all_results = []

    for query in queries:
        # Fetch from both APIs
        news_results = search_newsapi(query)
        serp_results = search_serpapi(query)

        all_results.extend(news_results)
        all_results.extend(serp_results)

        # Small delay between queries to be respectful of rate limits
        time.sleep(0.5)

    # Post-processing
    all_results = deduplicate_results(all_results)
    all_results = enforce_domain_diversity(all_results)

    return all_results
