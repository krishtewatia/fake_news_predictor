"""
Fake News Detector — FastAPI Backend
-------------------------------------
REST API that orchestrates the full fake news detection pipeline.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables before importing pipeline modules
load_dotenv()

from pipeline.input_layer import process_input
from pipeline.claim_extractor import extract_claims
from pipeline.query_generator import generate_queries
from pipeline.web_search import search_web
from pipeline.evidence_collector import collect_evidence
from pipeline.semantic_similarity import compute_similarity
from pipeline.evidence_ranker import rank_evidence
from pipeline.stance_detector import detect_stance
from pipeline.credibility_scorer import attach_credibility
from pipeline.hybrid_scorer import compute_final_score
from pipeline.aggregator import aggregate_results
from pipeline.verdict import generate_verdict
from pipeline.explainer import generate_explanation
from pipeline import cache

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fake News Detector",
    description="AI-powered fake news detection pipeline using NLP, web evidence, and LLM reasoning.",
    version="1.0.0",
)


# ── Pydantic Models ─────────────────────────────────────────────────────────

class CheckRequest(BaseModel):
    text: str | None = Field(default=None, description="Raw text to fact-check.")
    url: str | None = Field(default=None, description="URL of an article to fact-check.")


class EvidenceItem(BaseModel):
    url: str = ""
    domain: str = ""
    title: str = ""
    text: str = ""
    similarity_score: float = 0.0
    stance: str = "NEUTRAL"
    confidence: float = 0.0
    credibility_score: float = 0.0
    final_score: float = 0.0


class CheckResponse(BaseModel):
    verdict: str
    confidence: float
    explanation: str
    claims: list[str] = []
    evidence: list[EvidenceItem] = []


class HealthResponse(BaseModel):
    status: str
    version: str


class CacheClearResponse(BaseModel):
    message: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/cache/clear", response_model=CacheClearResponse)
async def clear_cache():
    """Clear the results cache."""
    cache.clear()
    return CacheClearResponse(message="Cache cleared successfully.")


@app.post("/check", response_model=CheckResponse)
async def check_news(request: CheckRequest):
    """
    Main endpoint — runs the full fake news detection pipeline.

    Accepts raw text or a URL, processes it through claim extraction,
    web search, evidence analysis, and returns a verdict.
    """
    if not request.text and not request.url:
        raise HTTPException(status_code=400, detail="Either 'text' or 'url' must be provided.")

    try:
        # Build a cache key from the input
        cache_key = request.text or request.url

        # Check cache first
        cached = cache.get(cache_key)
        if cached:
            return CheckResponse(**cached)

        # ── 1. Input Processing ──────────────────────────────────────────
        sentences = process_input(text=request.text, url=request.url)
        if not sentences:
            raise HTTPException(status_code=422, detail="Could not extract any text from the input.")

        # ── 2. Claim Extraction ──────────────────────────────────────────
        claims = extract_claims(sentences)
        if not claims:
            raise HTTPException(status_code=422, detail="No factual claims found in the input.")

        # ── 3. Query Generation ──────────────────────────────────────────
        queries = generate_queries(claims)

        # ── 4. Web Search ────────────────────────────────────────────────
        search_results = search_web(queries)

        if not search_results:
            return CheckResponse(
                verdict="UNCERTAIN",
                confidence=0.0,
                explanation="No evidence found online to verify the claims.",
                claims=claims,
                evidence=[],
            )

        # ── 5. Evidence Collection ───────────────────────────────────────
        evidence_list = collect_evidence(search_results, mode="fast")

        if not evidence_list:
            return CheckResponse(
                verdict="UNCERTAIN",
                confidence=0.0,
                explanation="Could not collect evidence from search results.",
                claims=claims,
                evidence=[],
            )

        # ── 6. Semantic Similarity ───────────────────────────────────────
        primary_claim = claims[0]
        evidence_list = compute_similarity(primary_claim, evidence_list)

        # ── 7. Evidence Ranking ──────────────────────────────────────────
        evidence_list = rank_evidence(evidence_list)

        # ── 8. Stance Detection ──────────────────────────────────────────
        evidence_list = detect_stance(primary_claim, evidence_list)

        # ── 9. Credibility Scoring ───────────────────────────────────────
        evidence_list = attach_credibility(evidence_list)

        # ── 10. Hybrid Scoring ───────────────────────────────────────────
        evidence_list = compute_final_score(evidence_list)

        # ── 11. Aggregation ──────────────────────────────────────────────
        agg = aggregate_results(evidence_list)

        # ── 12. Verdict (mechanical — may be overridden by LLM) ────────
        verdict_result = generate_verdict(
            score=agg["final_score"],
            support_count=agg["support_count"],
            refute_count=agg["refute_count"],
        )

        # ── 13. LLM Arbiter — corrects verdict + generates explanation ──
        llm_result = generate_explanation(
            claim=primary_claim,
            evidence=evidence_list,
            verdict=verdict_result["verdict"],
        )

        # The LLM acts as the final arbiter: its verdict overrides the
        # mechanical pipeline's verdict since it can reason about context.
        final_verdict = llm_result["verdict"]
        final_confidence = llm_result["confidence"]
        explanation = llm_result["explanation"]

        # ── Build Response ───────────────────────────────────────────────
        response_data = {
            "verdict": final_verdict,
            "confidence": final_confidence,
            "explanation": explanation,
            "claims": claims,
            "evidence": [
                {
                    "url": e.get("url", ""),
                    "domain": e.get("domain", ""),
                    "title": e.get("title", ""),
                    "text": e.get("text", "")[:500],
                    "similarity_score": e.get("similarity_score", 0.0),
                    "stance": e.get("stance", "NEUTRAL"),
                    "confidence": e.get("confidence", 0.0),
                    "credibility_score": e.get("credibility_score", 0.0),
                    "final_score": e.get("final_score", 0.0),
                }
                for e in evidence_list
            ],
        }

        # ── 14. Cache Result ─────────────────────────────────────────────
        cache.set(cache_key, response_data)

        return CheckResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
