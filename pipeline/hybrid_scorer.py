"""
Hybrid Scorer Module
--------------------
Computes a final trust score for each evidence item by combining
semantic similarity, stance confidence, and source credibility.
"""

# Weights for the scoring formula
WEIGHT_SIMILARITY = 0.3
WEIGHT_STANCE = 0.4
WEIGHT_CREDIBILITY = 0.3

# Stance direction mapping
STANCE_DIRECTION = {
    "SUPPORTS": 1.0,
    "REFUTES": -1.0,
    "NEUTRAL": 0.0,
}


def compute_final_score(evidence: list[dict]) -> list[dict]:
    """
    Compute a hybrid final score for each evidence item.

    Formula:
        score = 0.3 * similarity + 0.4 * (stance_confidence * direction) + 0.3 * credibility

    Args:
        evidence: List of evidence dicts with keys:
                  similarity_score, stance, confidence, credibility_score.

    Returns:
        The same evidence list with 'final_score' added to each item.
    """
    if not evidence:
        return evidence

    for item in evidence:
        similarity = item.get("similarity_score", 0.0)
        stance = item.get("stance", "NEUTRAL")
        confidence = item.get("confidence", 0.0)
        credibility = item.get("credibility_score", 0.0)

        direction = STANCE_DIRECTION.get(stance, 0.0)

        score = (
            WEIGHT_SIMILARITY * similarity
            + WEIGHT_STANCE * (confidence * direction)
            + WEIGHT_CREDIBILITY * credibility
        )

        item["final_score"] = round(score, 4)

    return evidence
