---
title: Fake News Detector
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.40.0
app_file: app_hf.py
pinned: false
license: mit
---

# 🔍 AI-Powered Fake News Detection Pipeline

A sophisticated **Hybrid AI + Machine Learning pipeline** designed to verify factual claims and detect misinformation. The system analyzes raw text or article URLs through a multi-stage NLP process, cross-references with real-time web evidence, and delivers a reasoned verdict powered by **Deep Learning** and **Large Language Models**.

**🔗 Live Demo**: https://huggingface.co/spaces/krishtewatia/fake-news-detector

---

## ✨ Features

- ✅ **Real-time Fact-Checking**: Instant verification of claims and articles
- 🔗 **Multi-Source Evidence**: Cross-references with NewsAPI and Google Search
- 🤖 **AI-Powered Reasoning**: Advanced NLP models for intelligent analysis
- 📊 **Detailed Analysis**: Confidence scores, evidence sources, and explanations
- 🎯 **14-Stage Pipeline**: Comprehensive verification process

---

## 🚀 How It Works (The Pipeline)

The detector operates through a modular 14-stage pipeline that ensures rigorous cross-validation of information:

### 1. Ingestion & Analysis
*   **Input Layer:** Ingests raw text or fetches article content from URLs using `newspaper3k` and `BeautifulSoup4`.
*   **Claim Extraction:** Uses `spaCy` to identify factual, non-subjective sentences. It scores sentences based on named entities, numeric data, and verb density to extract the most verifiable claims.

### 2. Evidence Gathering
*   **Query Generation:** Transforms extracted claims into optimized search queries using **Gemini 2.0 Flash** via OpenRouter.
*   **Web Search:** Conducts real-time searches across **NewsAPI** and **SerpAPI (Google Search)** to find corroborating or refuting articles.
*   **Evidence Collection:** Scrapes content from search results to build a local dataset for verification.

### 3. Verification & Scoring
*   **Semantic Similarity:** Applies the `all-MiniLM-L6-v2` Sentence-Transformer model to compute cosine similarity between the claim and collected evidence.
*   **Stance Detection:** Utilizes a **BART-Large-MNLI** model (Zero-Shot Classification) to determine if each piece of evidence **Supports**, **Refutes**, or is **Neutral** toward the claim.
*   **Source Credibility:** Assigns trust scores to sources based on domain reputation (e.g., BBC vs. unknown blogs).
*   **Hybrid Scorer:** Combines similarity, stance, and credibility into a unified trust metric for every piece of evidence.

### 4. Verdict & Reasoning
*   **Aggregation:** Computes a weighted average of all evidence scores.
*   **Verdict Engine:** Classifies the final result as **REAL**, **LIKELY FAKE**, or **UNCERTAIN** based on statistical thresholds.
*   **AI Explanation:** Uses **Gemini** to generate a natural language explanation of *why* the verdict was reached, citing specific evidence patterns.

---

## 🛠️ Technology Stack

| Layer | Framework/Library | Purpose |
| :--- | :--- | :--- |
| **Backend** | **FastAPI** | High-performance, asynchronous REST API |
| **Frontend** | **Streamlit** | Interactive dashboard for visualization |
| **NLP Core** | **spaCy** | Sentence tokenization and entity extraction |
| **Deep Learning** | **HuggingFace Transformers** | BART NLI for stance detection |
| **Embeddings** | **Sentence-Transformers** | Semantic similarity computation |
| **Web Scraping** | **newspaper3k / BeautifulSoup4** | Article extraction and HTML parsing |
| **Data Validation** | **Pydantic v2** | Strict request/response validation |

---

## 📡 External APIs

The system integrates several external services for real-time accuracy:

1. **OpenRouter (Gemini 2.0 Flash)**
   - Query optimization and explanation generation
   - Cost-effective LLM reasoning

2. **NewsAPI**
   - Real news articles from verified publishers
   - Sourcing recent, credible content

3. **SerpAPI**
   - Google Search integration
   - Broader web index access

---

## 📁 Project Structure

```
fake_news_detector/
├── app_hf.py                  # Streamlit UI (HF Spaces optimized)
├── main.py                    # FastAPI server & orchestration
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── start.sh                   # Startup script for HF Spaces
├── DEPLOYMENT_GUIDE.md        # Step-by-step deployment instructions
│
└── pipeline/
    ├── __init__.py
    ├── input_layer.py         # Data ingestion & cleaning
    ├── claim_extractor.py     # NLP-based claim identification
    ├── query_generator.py     # LLM-assisted search optimization
    ├── web_search.py          # Multi-source API integration
    ├── evidence_collector.py  # Article scraping & extraction
    ├── semantic_similarity.py # MiniLM-based vector comparison
    ├── evidence_ranker.py     # Evidence ranking logic
    ├── stance_detector.py     # BART NLI classification
    ├── credibility_scorer.py  # Domain trust-based weighting
    ├── hybrid_scorer.py       # Final score calculation
    ├── aggregator.py          # Result aggregation
    ├── verdict.py             # Statistical decision engine
    ├── explainer.py           # Natural language reasoning
    └── cache.py               # Caching layer
```

---

## ⚙️ Setup & Deployment

### Local Development

```bash
# 1. Clone repository
git clone <your-repo>
cd fake_news_detector

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Create .env file
cp .env.example .env
# Add your API keys:
# OPEN_ROUTER_KEY=...
# NEWSAPI_KEY=...
# SERPAPI_KEY=...

# 5. Run backend (Terminal 1)
uvicorn main:app --reload

# 6. Run frontend (Terminal 2)
streamlit run app_hf.py
```

### Deploy to Hugging Face Spaces

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

Quick summary:
1. Create Space at https://huggingface.co/spaces
2. Clone: `git clone https://huggingface.co/spaces/your-username/your-space`
3. Copy files and push
4. Add API keys in Space Settings → Secrets
5. Space auto-deploys!

**Your live URL**: `https://huggingface.co/spaces/your-username/your-space`

---

## 🔑 Required API Keys

Get from:
- **OpenRouter**: https://openrouter.ai (Free tier available)
- **NewsAPI**: https://newsapi.org (Free tier: 100 requests/day)
- **SerpAPI**: https://serpapi.com (Free tier: 100 searches/month)

Add to HF Space Secrets:
```
OPEN_ROUTER_KEY=<your-key>
NEWSAPI_KEY=<your-key>
SERPAPI_KEY=<your-key>
```

---

## 📊 Example Output

```
Input: "Moon landing was faked"

Output:
├── Verdict: LIKELY FAKE (92% confidence)
├── Explanation: "Multiple credible sources (NASA, scientific publications) confirm Apollo missions were real. No credible evidence supports conspiracy theories."
├── Claims:
│   ├── "Moon landing was faked"
│   └── "NASA conspiracy"
└── Evidence:
    ├── NASA Official Records (SUPPORTS, Credibility: 0.98)
    ├── BBC Science Archive (SUPPORTS, Credibility: 0.96)
    └── Conspiracy Theory Blog (REFUTES, Credibility: 0.15)
```

---

## 🚀 Performance Notes

- **First Run**: 2-3 minutes (models download)
- **Typical Query**: 30-60 seconds
- **Max Timeout**: 120 seconds
- **Storage**: ~2GB for all models
- **Recommended**: 2GB+ RAM

---

## 📝 Troubleshooting

| Issue | Solution |
| --- | --- |
| "Backend not responding" | Wait 10-15 seconds for FastAPI to start |
| "Module not found" | Ensure all `pipeline/` files are present |
| API key errors | Verify secrets in HF Space settings |
| Slow responses | Models are loading; first run is slowest |
| Space keeps crashing | Check logs; verify no hardcoded paths |

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

Built with ❤️ using:
- [FastAPI](https://fastapi.tiangolo.com)
- [Streamlit](https://streamlit.io)
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [spaCy](https://spacy.io)

---

**Made by**: [@krishtewatia](https://github.com/krishtewatia)

**Questions?** Open an issue on GitHub or comment on the HF Space!
