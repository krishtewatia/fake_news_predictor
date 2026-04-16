"""
Semantic Similarity Module
--------------------------
Computes cosine similarity between a claim and evidence texts
using sentence-transformers (all-MiniLM-L6-v2).
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once at module level
model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_similarity(claim: str, evidence: list[dict]) -> list[dict]:
    """
    Compute cosine similarity between a claim and each evidence text.
    Attaches a 'similarity_score' key to each evidence dict.

    Args:
        claim:    The factual claim string.
        evidence: List of evidence dicts, each containing a 'text' key.

    Returns:
        The same evidence list with 'similarity_score' added to each item,
        sorted by similarity in descending order.
    """
    if not claim or not evidence:
        return evidence

    # Filter out evidence with missing or empty text
    valid_evidence = [e for e in evidence if e.get("text", "").strip()]

    if not valid_evidence:
        return evidence

    # Encode claim and all evidence texts in one batch
    evidence_texts = [e["text"] for e in valid_evidence]
    embeddings = model.encode([claim] + evidence_texts, show_progress_bar=False)

    claim_embedding = embeddings[0].reshape(1, -1)
    evidence_embeddings = embeddings[1:]

    # Compute cosine similarity between claim and each evidence
    similarities = cosine_similarity(claim_embedding, evidence_embeddings)[0]

    # Attach scores to evidence dicts
    for item, score in zip(valid_evidence, similarities):
        item["similarity_score"] = round(float(score), 4)

    # Mark any skipped items with a zero score
    for item in evidence:
        if "similarity_score" not in item:
            item["similarity_score"] = 0.0

    # Sort by similarity descending
    evidence.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

    return evidence
