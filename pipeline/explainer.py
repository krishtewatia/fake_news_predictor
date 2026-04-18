"""
Explainer Module
----------------
Uses Gemini LLM as the final arbiter to produce a corrected verdict AND 
a human-readable explanation. The LLM receives the claim + evidence and
independently determines the correct verdict, fixing cases where the
NLI-based mechanical pipeline produces an incorrect label.
"""

import json
import re

from pipeline.llm_client import generate_text


def build_prompt(claim: str, evidence: list[dict], pipeline_verdict: str) -> str:
    """
    Build a prompt for the LLM to act as the final fact-checking arbiter.

    Args:
        claim:             The original factual claim.
        evidence:          List of evidence dicts with 'text' and 'stance' keys.
        pipeline_verdict:  The verdict from the mechanical pipeline (for reference).

    Returns:
        Formatted prompt string.
    """
    # Select top 5 evidence texts for richer context
    top_evidence = evidence[:5]

    evidence_block = ""
    for i, item in enumerate(top_evidence, 1):
        text = item.get("text", "")[:400]
        stance = item.get("stance", "NEUTRAL")
        source = item.get("domain", "unknown")
        evidence_block += f"  {i}. [Source: {source}] [NLI Stance: {stance}] {text}\n"

    prompt = (
        f"You are a fact-checking expert. Analyze the following claim against the evidence provided.\n\n"
        f"Claim: \"{claim}\"\n\n"
        f"Evidence collected from the web:\n{evidence_block}\n"
        f"The automated NLI pipeline suggested: {pipeline_verdict}\n\n"
        f"IMPORTANT: The NLI pipeline can be WRONG — it sometimes confuses topical relevance with factual support. "
        f"YOU must independently determine if the claim is actually true or false based on what the evidence says.\n\n"
        f"Respond with ONLY a valid JSON object (no markdown, no code fences) in this exact format:\n"
        f'{{"verdict": "REAL" or "LIKELY FAKE" or "UNCERTAIN", "confidence": 0.0 to 1.0, "explanation": "2-3 sentence explanation"}}\n\n'
        f"Rules:\n"
        f"- REAL: The evidence clearly confirms the claim is true.\n"
        f"- LIKELY FAKE: The evidence contradicts the claim, or the claim describes events that are not reported in any credible source.\n"
        f"- UNCERTAIN: The evidence is insufficient or mixed.\n"
        f"- If the claim describes an extraordinary event (assassination, major disaster, etc.) and NO evidence confirms it actually happened, it is LIKELY FAKE.\n"
    )

    return prompt


def parse_llm_response(response: str) -> dict | None:
    """
    Parse the LLM's JSON response. Handles cases where the LLM wraps
    the JSON in markdown code fences.

    Returns:
        Parsed dict with verdict, confidence, explanation, or None on failure.
    """
    if not response:
        return None

    # Strip markdown code fences if present
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        verdict = data.get("verdict", "").upper().strip()
        if verdict not in ("REAL", "LIKELY FAKE", "UNCERTAIN"):
            return None
        return {
            "verdict": verdict,
            "confidence": float(data.get("confidence", 0.5)),
            "explanation": data.get("explanation", ""),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def generate_explanation(claim: str, evidence: list[dict], verdict: str) -> dict:
    """
    Use the LLM as the final arbiter to produce a corrected verdict and explanation.

    Args:
        claim:    The factual claim being evaluated.
        evidence: List of evidence dicts (should include 'text' and 'stance').
        verdict:  The pipeline's mechanical verdict (REAL, LIKELY FAKE, UNCERTAIN).

    Returns:
        Dict with keys: verdict, confidence, explanation.
        Falls back to the pipeline verdict if the LLM is unavailable.
    """
    if not claim:
        return {
            "verdict": verdict,
            "confidence": 0.5,
            "explanation": "Insufficient information to generate an explanation.",
        }

    prompt = build_prompt(claim, evidence, verdict)
    raw_response = generate_text(prompt)

    parsed = parse_llm_response(raw_response)
    if parsed and parsed["explanation"]:
        return parsed

    # Fallback: return the pipeline verdict with a generic explanation
    support = sum(1 for e in evidence if e.get("stance") == "SUPPORTS")
    refute = sum(1 for e in evidence if e.get("stance") == "REFUTES")
    return {
        "verdict": verdict,
        "confidence": 0.5,
        "explanation": (
            f"The claim was evaluated against {len(evidence)} source(s). "
            f"{support} source(s) support and {refute} source(s) refute the claim. "
            f"Verdict: {verdict}."
        ),
    }
