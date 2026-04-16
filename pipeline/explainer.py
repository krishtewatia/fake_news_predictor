"""
Explainer Module
----------------
Generates a human-readable explanation of the verdict using Gemini LLM.
"""

from pipeline.llm_client import generate_text


def build_prompt(claim: str, evidence: list[dict], verdict: str) -> str:
    """
    Build a prompt for the LLM to explain the verdict.

    Args:
        claim:    The original factual claim.
        evidence: List of evidence dicts with 'text' and 'stance' keys.
        verdict:  The final verdict string (REAL, LIKELY FAKE, UNCERTAIN).

    Returns:
        Formatted prompt string.
    """
    # Select top 3 evidence texts
    top_evidence = evidence[:3]

    evidence_block = ""
    for i, item in enumerate(top_evidence, 1):
        text = item.get("text", "")[:300]
        stance = item.get("stance", "NEUTRAL")
        evidence_block += f"  {i}. [{stance}] {text}\n"

    prompt = (
        f"Claim: {claim}\n"
        f"Verdict: {verdict}\n\n"
        f"Evidence:\n{evidence_block}\n"
        f"Explain briefly (2-3 sentences) why this verdict was reached "
        f"based on the evidence above. Be concise and factual."
    )

    return prompt


def generate_explanation(claim: str, evidence: list[dict], verdict: str) -> str:
    """
    Generate a short explanation of the verdict using Gemini LLM.

    Args:
        claim:    The factual claim being evaluated.
        evidence: List of evidence dicts (should include 'text' and 'stance').
        verdict:  The verdict string (REAL, LIKELY FAKE, UNCERTAIN).

    Returns:
        A 2-3 sentence explanation string, or a fallback message on failure.
    """
    if not claim or not verdict:
        return "Insufficient information to generate an explanation."

    prompt = build_prompt(claim, evidence, verdict)
    explanation = generate_text(prompt)

    if not explanation:
        # Fallback explanation when LLM is unavailable
        support = sum(1 for e in evidence if e.get("stance") == "SUPPORTS")
        refute = sum(1 for e in evidence if e.get("stance") == "REFUTES")
        return (
            f"The claim was evaluated against {len(evidence)} source(s). "
            f"{support} source(s) support and {refute} source(s) refute the claim. "
            f"Verdict: {verdict}."
        )

    return explanation
