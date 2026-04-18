"""
Stance Detector Module
----------------------
Detects whether evidence supports, refutes, or is neutral to a claim
using Natural Language Inference (NLI) with facebook/bart-large-mnli.
"""

from transformers import pipeline

# Load zero-shot NLI pipeline once at module level
nli_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

# Candidate labels for zero-shot classification — these labels must form
# natural sentences when inserted into the hypothesis_template via {}.
CANDIDATE_LABELS = [
    "supports this claim",
    "contradicts this claim",
    "is unrelated to this claim",
]

# Map the candidate labels back to pipeline stance labels
LABEL_TO_STANCE = {
    "supports this claim": "SUPPORTS",
    "contradicts this claim": "REFUTES",
    "is unrelated to this claim": "NEUTRAL",
}


def classify_stance(premise: str, hypothesis: str) -> tuple[str, float]:
    """
    Classify the stance of a premise towards a hypothesis using NLI.

    Args:
        premise:    The evidence text.
        hypothesis: The claim to evaluate against.

    Returns:
        Tuple of (stance, confidence) where stance is SUPPORTS/REFUTES/NEUTRAL.
    """
    try:
        result = nli_pipeline(
            premise,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template="This text {}.",
            multi_label=False,
        )

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        stance = LABEL_TO_STANCE.get(top_label, "NEUTRAL")
        return stance, round(float(top_score), 4)

    except Exception as e:
        print(f"[StanceDetector] Classification error: {e}")
        return "NEUTRAL", 0.0


def detect_stance(claim: str, evidence: list[dict]) -> list[dict]:
    """
    Detect the stance of each evidence item relative to a claim.
    Attaches 'stance' and 'confidence' keys to each evidence dict.

    Args:
        claim:    The factual claim string.
        evidence: List of evidence dicts, each containing a 'text' key.

    Returns:
        The same evidence list with 'stance' and 'confidence' added to each item.
    """
    if not claim or not evidence:
        return evidence

    for item in evidence:
        text = item.get("text", "").strip()

        if not text:
            item["stance"] = "NEUTRAL"
            item["confidence"] = 0.0
            continue

        # Truncate very long texts to avoid model input limits
        if len(text) > 1024:
            text = text[:1024]

        stance, confidence = classify_stance(text, claim)
        item["stance"] = stance
        item["confidence"] = confidence

    return evidence
