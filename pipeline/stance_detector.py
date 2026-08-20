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
CANDIDATE_TEMPLATES = {
    "SUPPORTS": "supports the claim that {}",
    "REFUTES": "contradicts the claim that {}",
    "NEUTRAL": "is unrelated to the claim that {}",
}


def classify_stance(premise: str, claim: str) -> tuple[str, float]:
    """
    Classify the stance of a premise towards a hypothesis using NLI.

    Args:
        premise:    The evidence text.
        hypothesis: The claim to evaluate against.

    Returns:
        Tuple of (stance, confidence) where stance is SUPPORTS/REFUTES/NEUTRAL.
    """
    try:
        candidate_labels = [template.format(claim) for template in CANDIDATE_TEMPLATES.values()]
        label_lookup = {template.format(claim): stance for stance, template in CANDIDATE_TEMPLATES.items()}

        result = nli_pipeline(
            premise,
            candidate_labels=candidate_labels,
            hypothesis_template="This evidence {}.",
            multi_label=False,
        )

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        stance = label_lookup.get(top_label, "NEUTRAL")
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
