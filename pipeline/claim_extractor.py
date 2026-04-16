"""
Claim Extractor Module
----------------------
Identifies and ranks factual claims from a list of sentences using spaCy.
Scores sentences based on named entities, numbers/dates, and verb density.
Filters out subjective/opinion sentences.
"""

import spacy

# Load spaCy model once at module level
nlp = spacy.load("en_core_web_sm")

# Subjectivity indicators — words that signal opinion rather than fact
OPINION_MARKERS = {
    "think", "believe", "feel", "opinion", "perhaps", "maybe", "probably",
    "might", "could", "seems", "apparently", "arguably", "personally",
    "honestly", "frankly", "hopefully", "unfortunately", "sadly",
    "amazing", "terrible", "wonderful", "horrible", "beautiful", "ugly",
    "best", "worst", "love", "hate", "prefer", "wish", "suppose",
    "doubt", "guess", "imagine", "suspect",
}

# Entity types that indicate factual content
FACTUAL_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY",
    "PERCENT", "QUANTITY", "EVENT", "LAW", "PRODUCT", "NORP",
}

# Token types that represent numbers or dates
NUMERIC_ENTITY_TYPES = {"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"}


def is_opinion(doc: spacy.tokens.Doc) -> bool:
    """
    Check if a sentence is subjective/opinion-based.

    Args:
        doc: A spaCy Doc object of a single sentence.

    Returns:
        True if the sentence is likely an opinion, False otherwise.
    """
    tokens_lower = {token.lemma_.lower() for token in doc}
    overlap = tokens_lower & OPINION_MARKERS
    return len(overlap) >= 2


def count_named_entities(doc: spacy.tokens.Doc) -> int:
    """
    Count named entities that indicate factual content.

    Args:
        doc: A spaCy Doc object.

    Returns:
        Count of factual named entities.
    """
    return sum(1 for ent in doc.ents if ent.label_ in FACTUAL_ENTITY_TYPES)


def count_numeric_references(doc: spacy.tokens.Doc) -> int:
    """
    Count numeric references (numbers, dates, money, percentages).

    Args:
        doc: A spaCy Doc object.

    Returns:
        Count of numeric entity references and numeric tokens.
    """
    entity_count = sum(1 for ent in doc.ents if ent.label_ in NUMERIC_ENTITY_TYPES)
    token_count = sum(1 for token in doc if token.like_num)
    return entity_count + token_count


def compute_verb_density(doc: spacy.tokens.Doc) -> float:
    """
    Compute the ratio of verbs to total tokens in a sentence.
    Higher verb density suggests action-oriented, factual statements.

    Args:
        doc: A spaCy Doc object.

    Returns:
        Verb density as a float between 0 and 1.
    """
    total_tokens = len([t for t in doc if not t.is_punct and not t.is_space])
    if total_tokens == 0:
        return 0.0
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    return verb_count / total_tokens


def score_sentence(doc: spacy.tokens.Doc) -> float:
    """
    Compute a factual claim score for a sentence.
    Higher scores indicate stronger factual claims.

    Scoring weights:
        - Named entities:      3 points each
        - Numeric references:  2 points each
        - Verb density:        1 point (scaled)

    Args:
        doc: A spaCy Doc object of a single sentence.

    Returns:
        Factual claim score as a float.
    """
    entity_score = count_named_entities(doc) * 3.0
    numeric_score = count_numeric_references(doc) * 2.0
    verb_score = compute_verb_density(doc) * 1.0

    return entity_score + numeric_score + verb_score


def extract_claims(sentences: list[str]) -> list[str]:
    """
    Extract the top 3 factual claims from a list of sentences.

    Sentences are scored based on named entities, numeric references,
    and verb density. Opinion/subjective sentences are filtered out.

    Args:
        sentences: List of cleaned sentence strings.

    Returns:
        Top 3 ranked factual claim sentences.
    """
    if not sentences:
        return []

    scored_claims = []

    for sentence in sentences:
        doc = nlp(sentence)

        # Skip very short sentences (less than 4 non-punctuation tokens)
        meaningful_tokens = [t for t in doc if not t.is_punct and not t.is_space]
        if len(meaningful_tokens) < 4:
            continue

        # Filter out opinion/subjective sentences
        if is_opinion(doc):
            continue

        score = score_sentence(doc)
        scored_claims.append((sentence, score))

    # Rank by score in descending order
    scored_claims.sort(key=lambda x: x[1], reverse=True)

    # Return top 3 claims
    top_claims = [claim for claim, _ in scored_claims[:3]]
    return top_claims
