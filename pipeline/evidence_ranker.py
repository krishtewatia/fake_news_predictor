"""
Evidence Ranker Module
----------------------
Filters and ranks evidence by semantic similarity score.
"""


def rank_evidence(evidence: list[dict], top_n: int = 5, threshold: float = 0.3) -> list[dict]:
    """
    Filter and rank evidence items by similarity score.

    Args:
        evidence:  List of evidence dicts, each with a 'similarity_score' key.
        top_n:     Maximum number of results to return.
        threshold: Minimum similarity score to include.

    Returns:
        Top-N evidence items above the threshold, sorted descending by score.
    """
    if not evidence:
        return []

    # Filter below threshold
    filtered = [e for e in evidence if e.get("similarity_score", 0.0) >= threshold]

    # Sort descending by similarity score
    filtered.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

    # Return top N
    return filtered[:top_n]
