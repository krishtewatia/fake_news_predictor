"""
Input Layer Module
------------------
Handles text ingestion from raw strings or URLs.
Cleans, normalizes, and splits input into sentences using spaCy.
"""

import re
import requests
from bs4 import BeautifulSoup
import spacy

# Load spaCy model once at module level for reusability
nlp = spacy.load("en_core_web_sm")


def fetch_article_from_url(url: str) -> str:
    """
    Fetch article text from a given URL by extracting content from <p> tags.

    Args:
        url: The URL to fetch article content from.

    Returns:
        Extracted article text as a single string.

    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If no article content is found.
    """
    try:
        response = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"
        })
        response.raise_for_status()
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch URL '{url}': {e}")

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    article_text = " ".join(p.get_text() for p in paragraphs)

    if not article_text.strip():
        raise ValueError(f"No article content found at URL: {url}")

    return article_text


def clean_text(text: str) -> str:
    """
    Clean raw text by removing HTML tags, special characters, and normalizing whitespace.

    Args:
        text: The raw text to clean.

    Returns:
        Cleaned and normalized text string.
    """
    # Remove any residual HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove special characters (keep letters, digits, basic punctuation, spaces)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_into_sentences(text: str) -> list[str]:
    """
    Split cleaned text into individual sentences using spaCy.

    Args:
        text: Cleaned text to split.

    Returns:
        List of sentence strings.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def process_input(text: str = None, url: str = None) -> list[str]:
    """
    Main entry point for the input layer.
    Accepts either raw text or a URL, cleans the content, and returns a list of sentences.

    Args:
        text: Raw text input to process.
        url:  URL to fetch article content from.

    Returns:
        List of cleaned sentences extracted from the input.

    Raises:
        ValueError: If neither text nor url is provided.
    """
    if not text and not url:
        raise ValueError("Either 'text' or 'url' must be provided.")

    # If URL is provided, fetch article text from it
    if url:
        try:
            raw_text = fetch_article_from_url(url)
        except (requests.RequestException, ValueError) as e:
            print(f"[InputLayer] Error fetching URL: {e}")
            return []
    else:
        raw_text = text

    # Clean and split into sentences
    cleaned = clean_text(raw_text)

    if not cleaned:
        print("[InputLayer] Warning: Text is empty after cleaning.")
        return []

    sentences = split_into_sentences(cleaned)
    return sentences
