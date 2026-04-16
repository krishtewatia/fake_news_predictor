"""
Aggregator Module
-----------------
Aggregates stance counts and computes a credibility-weighted final score
across all evidence items.
"""


def aggregate_results(evidence: list[dict]) -> dict:
    """
    Aggregate evidence into stance counts and a credibility-weighted score.

    Weighted score formula:
        For each evidence item: credibility * direction (+1 SUPPORTS, -1 REFUTES, 0 NEUTRAL)
        Final = sum(weighted) / sum(credibility weights)

    Args:
        evidence: List of evidence dicts with keys:
                  stance, credibility_score.

    Returns:
        Dict with support_count, refute_count, neutral_count, final_score.
    """
    if not evidence:
        return {
            "support_count": 0,
            "refute_count": 0,
            "neutral_count": 0,
            "final_score": 0.0,
        }

    support_count = 0
    refute_count = 0
    neutral_count = 0

    weighted_sum = 0.0
    total_weight = 0.0

    direction_map = {
        "SUPPORTS": 1.0,
        "REFUTES": -1.0,
        "NEUTRAL": 0.0,
    }

    for item in evidence:
        stance = item.get("stance", "NEUTRAL")
        credibility = item.get("credibility_score", 0.3)

        # Count stances
        if stance == "SUPPORTS":
            support_count += 1
        elif stance == "REFUTES":
            refute_count += 1
        else:
            neutral_count += 1

        # Weighted score
        direction = direction_map.get(stance, 0.0)
        weighted_sum += credibility * direction
        total_weight += credibility

    final_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

    return {
        "support_count": support_count,
        "refute_count": refute_count,
        "neutral_count": neutral_count,
        "final_score": final_score,
    }
