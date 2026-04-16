"""
Query Generator Module
----------------------
Converts extracted claims into concise search engine queries.
Uses spaCy for heuristic SVO extraction and Gemini LLM for refinement.
"""

import re
import spacy
from pipeline.llm_client import generate_text

# Load spaCy model once at module level
nlp = spacy.load("en_core_web_sm")

# Maximum words allowed in a query
MAX_QUERY_WORDS = 15


def extract_svo(doc: spacy.tokens.Doc) -> tuple[str, str, str]:
    """
    Extract subject, main verb, and object from a spaCy Doc.

    Args:
        doc: A spaCy Doc object of a single sentence.

    Returns:
        Tuple of (subject, verb, object) strings. Empty string if not found.
    """
    subject = ""
    verb = ""
    obj = ""

    for token in doc:
        # Find the root verb
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            verb = token.text

        # Find subject (nominal subject or passive subject)
        if token.dep_ in ("nsubj", "nsubjpass"):
            # Include compound words (e.g., "Prime Minister")
            compounds = [t.text for t in token.subtree if t.dep_ in ("compound", "amod")]
            compounds.append(token.text)
            subject = " ".join(compounds)

        # Find direct object or attribute
        if token.dep_ in ("dobj", "attr", "pobj"):
            obj_parts = [t.text for t in token.subtree if t.dep_ in ("compound", "amod", "det") or t == token]
            obj = " ".join(obj_parts)

    return subject, verb, obj


def build_heuristic_query(claim: str) -> str:
    """
    Build a search query from a claim using spaCy SVO extraction.
    Fallback method when Gemini is unavailable.

    Args:
        claim: The claim sentence.

    Returns:
        A heuristic search query string.
    """
    doc = nlp(claim)
    subject, verb, obj = extract_svo(doc)

    # Collect named entities for the query
    entities = [ent.text for ent in doc.ents]

    # Build query from available parts
    parts = []
    if subject:
        parts.append(subject)
    if verb:
        parts.append(verb)
    if obj:
        parts.append(obj)

    # Add named entities not already captured
    for ent in entities:
        if ent not in " ".join(parts):
            parts.append(ent)

    query = " ".join(parts).strip()

    # If extraction yielded very little, fall back to key noun chunks
    if len(query.split()) < 3:
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        query = " ".join(noun_chunks[:4]) if noun_chunks else claim

    return truncate_query(query)


def clean_query(query: str) -> str:
    """
    Clean a generated query by removing quotes, newlines, and extra whitespace.

    Args:
        query: Raw query string.

    Returns:
        Cleaned query string.
    """
    # Remove surrounding quotes
    query = query.strip().strip("\"'`")

    # Remove newlines and excessive whitespace
    query = re.sub(r"[\n\r]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    return query


def truncate_query(query: str) -> str:
    """
    Truncate a query to a maximum number of words.

    Args:
        query: Query string to truncate.

    Returns:
        Truncated query string.
    """
    words = query.split()
    if len(words) > MAX_QUERY_WORDS:
        words = words[:MAX_QUERY_WORDS]
    return " ".join(words)


def claim_to_query(claim: str) -> str:
    """
    Convert a single claim into a search engine query.
    Uses Gemini for refinement with heuristic fallback.

    Args:
        claim: A factual claim sentence.

    Returns:
        A concise search engine query string.
    """
    prompt = (
        "Convert this claim into a short, precise search engine query. "
        "Return ONLY the query, nothing else.\n\n"
        f"{claim}"
    )

    gemini_query = generate_text(prompt)

    if gemini_query:
        cleaned = clean_query(gemini_query)
        return truncate_query(cleaned)

    # Fallback to heuristic if Gemini fails
    print(f"[QueryGenerator] Gemini unavailable, using heuristic for: {claim[:50]}...")
    return build_heuristic_query(claim)


def generate_queries(claims: list[str]) -> list[str]:
    """
    Generate search engine queries from a list of factual claims.

    Args:
        claims: List of claim sentences.

    Returns:
        List of concise search queries, one per claim.
    """
    if not claims:
        return []

    queries = []
    for claim in claims:
        query = claim_to_query(claim)
        if query:
            queries.append(query)

    return queries
