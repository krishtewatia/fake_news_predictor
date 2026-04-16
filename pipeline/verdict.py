"""
Verdict Module
--------------
Generates a final verdict (REAL, LIKELY FAKE, or UNCERTAIN)
based on the aggregated score and stance counts.
"""


def generate_verdict(score: float, support_count: int, refute_count: int) -> dict:
    """
    Generate a verdict on the claim's truthfulness.

    Rules:
        - REAL:        score > 0.65 AND support_count >= refute_count
        - LIKELY FAKE: score < 0.35 OR refute_count > support_count * 2
        - UNCERTAIN:   everything else

    Args:
        score:         Aggregated credibility-weighted score (-1.0 to 1.0).
        support_count: Number of supporting evidence items.
        refute_count:  Number of refuting evidence items.

    Returns:
        Dict with verdict, confidence, and reasoning.
    """
    verdict = "UNCERTAIN"
    confidence = 0.0
    reasoning = ""

    if score > 0.65 and support_count >= refute_count:
        verdict = "REAL"
        confidence = round(min(score, 1.0), 4)
        reasoning = (
            f"The claim is supported by {support_count} source(s) with a "
            f"high aggregated score of {score:.2f}. Evidence strongly aligns "
            f"with the claim."
        )

    elif score < 0.35 or refute_count > support_count * 2:
        verdict = "LIKELY FAKE"
        confidence = round(min(abs(1.0 - score), 1.0), 4)
        reasoning = (
            f"The claim is contradicted by {refute_count} source(s) with a "
            f"low aggregated score of {score:.2f}. Evidence suggests the claim "
            f"is unreliable or false."
        )

    else:
        verdict = "UNCERTAIN"
        confidence = round(1.0 - abs(score - 0.5), 4)
        reasoning = (
            f"The evidence is mixed — {support_count} supporting and "
            f"{refute_count} refuting source(s) with a score of {score:.2f}. "
            f"Not enough evidence to make a definitive judgment."
        )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
    }
