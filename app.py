"""
Fake News Detector — Streamlit Frontend
-----------------------------------------
Clean, minimal UI for the fake news detection pipeline.
"""

import streamlit as st
import requests

# ── Config ───────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"

VERDICT_COLORS = {
    "REAL": "#10B981",
    "LIKELY FAKE": "#EF4444",
    "UNCERTAIN": "#F59E0B",
}

VERDICT_ICONS = {
    "REAL": "✅",
    "LIKELY FAKE": "🚨",
    "UNCERTAIN": "⚠️",
}

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .main-header p {
        color: #94A3B8;
        font-size: 1rem;
        margin-top: 0;
    }

    .verdict-card {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .verdict-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 0.5rem;
    }

    .verdict-text {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.25rem 0;
    }

    .confidence-text {
        font-size: 1.1rem;
        color: #CBD5E1;
        margin-top: 0.25rem;
    }

    .explanation-box {
        background: rgba(99, 102, 241, 0.08);
        border-left: 3px solid #6366F1;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #E2E8F0;
        line-height: 1.6;
    }

    .evidence-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }

    .evidence-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
    }

    .evidence-meta {
        font-size: 0.8rem;
        color: #94A3B8;
    }

    .stance-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    .badge-supports { background: rgba(16,185,129,0.15); color: #10B981; }
    .badge-refutes  { background: rgba(239,68,68,0.15);  color: #EF4444; }
    .badge-neutral  { background: rgba(245,158,11,0.15); color: #F59E0B; }

    div[data-testid="stForm"] {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.5rem;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔍 Fake News Detector</h1>
    <p>Paste text or a URL to fact-check using AI-powered analysis</p>
</div>
""", unsafe_allow_html=True)


# ── Input Form ───────────────────────────────────────────────────────────────

with st.form("check_form", clear_on_submit=False):
    text_input = st.text_area(
        "News Text",
        placeholder="Paste the news article or claim here...",
        height=140,
    )
    url_input = st.text_input(
        "Or enter a URL",
        placeholder="https://example.com/article",
    )
    submitted = st.form_submit_button("🔎  Check")


# ── Pipeline Call ────────────────────────────────────────────────────────────

if submitted:
    if not text_input.strip() and not url_input.strip():
        st.warning("Please enter some text or a URL to check.")
    else:
        with st.spinner("Analyzing — this may take a moment..."):
            try:
                payload = {}
                if text_input.strip():
                    payload["text"] = text_input.strip()
                if url_input.strip():
                    payload["url"] = url_input.strip()

                response = requests.post(f"{API_URL}/check", json=payload, timeout=120)

                if response.status_code != 200:
                    detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {detail}")
                else:
                    data = response.json()

                    verdict = data.get("verdict", "UNCERTAIN")
                    confidence = data.get("confidence", 0.0)
                    explanation = data.get("explanation", "")
                    claims = data.get("claims", [])
                    evidence = data.get("evidence", [])

                    color = VERDICT_COLORS.get(verdict, "#F59E0B")
                    icon = VERDICT_ICONS.get(verdict, "⚠️")

                    # ── Verdict Card ─────────────────────────────────
                    st.markdown(f"""
                    <div class="verdict-card" style="background: {color}10; border-color: {color}30;">
                        <div class="verdict-label">Verdict</div>
                        <div class="verdict-text" style="color: {color};">
                            {icon} {verdict}
                        </div>
                        <div class="confidence-text">
                            Confidence: {confidence * 100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Explanation ───────────────────────────────────
                    if explanation:
                        st.markdown("#### 💡 Explanation")
                        st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)

                    # ── Claims ────────────────────────────────────────
                    if claims:
                        st.markdown("#### 📌 Extracted Claims")
                        for i, claim in enumerate(claims, 1):
                            st.markdown(f"**{i}.** {claim}")

                    # ── Evidence ──────────────────────────────────────
                    if evidence:
                        st.markdown("#### 📄 Evidence")
                        for item in evidence:
                            stance = item.get("stance", "NEUTRAL")
                            badge_class = f"badge-{stance.lower().replace(' ', '')}"
                            if stance == "LIKELY FAKE":
                                badge_class = "badge-refutes"

                            with st.expander(f"{item.get('title', 'Untitled')} — {item.get('domain', '')}"):
                                st.markdown(f"""
                                <div class="evidence-item">
                                    <span class="stance-badge {badge_class}">{stance}</span>
                                    <span class="evidence-meta" style="margin-left: 0.5rem;">
                                        Similarity: {item.get('similarity_score', 0):.2f} &nbsp;|&nbsp;
                                        Credibility: {item.get('credibility_score', 0):.2f} &nbsp;|&nbsp;
                                        Score: {item.get('final_score', 0):.2f}
                                    </span>
                                    <p style="margin-top: 0.75rem; color: #CBD5E1; line-height: 1.5;">
                                        {item.get('text', '')[:400]}
                                    </p>
                                    <a href="{item.get('url', '#')}" target="_blank"
                                       style="color: #818CF8; font-size: 0.85rem;">
                                        🔗 Open source
                                    </a>
                                </div>
                                """, unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend. Make sure the API is running at http://localhost:8000")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The analysis is taking too long.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
