# 🔍 Fake News Detection Pipeline

A **hybrid AI + ML pipeline** that detects fake news by analyzing text or article URLs. The system extracts claims, searches for corroborating evidence, evaluates source credibility, and delivers a verdict — **REAL**, **LIKELY FAKE**, or **UNCERTAIN** — powered by NLP models and Google Gemini.

---

## ✨ Features

- **Multi-stage detection pipeline** — 14 modular steps from input to verdict
- **Claim extraction** — spaCy-based NLP to identify factual claims
- **Web evidence search** — Dual-source search via NewsAPI + SerpAPI
- **Semantic similarity** — sentence-transformers (`all-MiniLM-L6-v2`) for claim-evidence matching
- **Stance detection** — HuggingFace NLI (`facebook/bart-large-mnli`) to classify evidence as SUPPORTS / REFUTES / NEUTRAL
- **Source credibility scoring** — Domain-based trust tiers
- **Hybrid scoring system** — Weighted combination of similarity, stance, and credibility
- **Gemini-powered explanation** — Natural language reasoning via Google Gemini API
- **FastAPI backend** — Async REST API with Pydantic models
- **Streamlit frontend** — Clean, minimal UI with color-coded verdicts
- **Caching system** — File-based JSON cache with 24-hour TTL

---

## 📁 Project Structure

```
fake_news_detector/
├── pipeline/
│   ├── __init__.py
│   ├── input_layer.py          # Text/URL ingestion & cleaning
│   ├── claim_extractor.py      # Factual claim extraction
│   ├── query_generator.py      # Search query generation
│   ├── web_search.py           # NewsAPI & SerpAPI search
│   ├── evidence_collector.py   # Evidence text collection
│   ├── semantic_similarity.py  # Claim-evidence similarity
│   ├── evidence_ranker.py      # Evidence filtering & ranking
│   ├── stance_detector.py      # NLI-based stance detection
│   ├── credibility_scorer.py   # Domain trust scoring
│   ├── hybrid_scorer.py        # Final hybrid score computation
│   ├── aggregator.py           # Result aggregation
│   ├── verdict.py              # Verdict generation
│   ├── explainer.py            # LLM-powered explanation
│   ├── llm_client.py           # Google Gemini API client
│   └── cache.py                # File-based JSON cache
├── app.py                      # Streamlit frontend
├── main.py                     # FastAPI backend
├── config.py                   # Environment variable config
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.10+ |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **NLP** | spaCy (`en_core_web_sm`) |
| **Stance Detection** | HuggingFace Transformers (`facebook/bart-large-mnli`) |
| **Semantic Similarity** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **LLM** | Google Gemini API (`gemini-1.5-flash`) |
| **Web Scraping** | BeautifulSoup4, newspaper3k |
| **Search APIs** | NewsAPI, SerpAPI |
| **Domain Parsing** | tldextract |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy model

```bash
python -m spacy download en_core_web_sm
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root (use `.env.example` as a template):

```env
GEMINI_API_KEY=your_gemini_api_key_here
NEWSAPI_KEY=your_newsapi_key_here
SERPAPI_KEY=your_serpapi_key_here
```

| Variable | Source | Required |
|----------|--------|----------|
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) | ✅ |
| `NEWSAPI_KEY` | [NewsAPI.org](https://newsapi.org/) | ✅ |
| `SERPAPI_KEY` | [SerpAPI.com](https://serpapi.com/) | ✅ |

---

## ▶️ How to Run

### Start the Backend

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Start the Frontend

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`

> **Note:** Both the backend and frontend must be running simultaneously. Use two separate terminal windows.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/check` | Run the full fake news detection pipeline |
| `GET` | `/health` | Health check |
| `GET` | `/cache/clear` | Clear the results cache |

### Interactive Docs

Once the backend is running, visit:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 📝 Example Usage

### Request

```bash
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA confirmed that an asteroid will hit Earth in 2025."}'
```

### Response

```json
{
  "verdict": "LIKELY FAKE",
  "confidence": 0.82,
  "explanation": "Multiple credible sources, including NASA's official channels, have not issued any such confirmation. The claim appears to be fabricated and contradicts verified astronomical data.",
  "claims": [
    "NASA confirmed that an asteroid will hit Earth in 2025."
  ],
  "evidence": [
    {
      "url": "https://www.reuters.com/...",
      "domain": "reuters.com",
      "title": "Fact Check: No asteroid threat confirmed by NASA",
      "text": "NASA has not confirmed any asteroid impact...",
      "similarity_score": 0.78,
      "stance": "REFUTES",
      "confidence": 0.91,
      "credibility_score": 0.9,
      "final_score": -0.12
    }
  ]
}
```

---

## 🖼️ Screenshots

<!-- Add screenshots of the Streamlit UI here -->

> _Screenshots coming soon._

---

## 🔮 Future Improvements

- 🧠 **Fine-tuned models** — Train custom models on fake news datasets for higher accuracy
- ⚡ **Real-time fact-checking** — Live monitoring of trending news and social media
- 🌐 **Browser extension** — One-click fact-checking directly in the browser
- 🗄️ **Database integration** — Replace file-based cache with Redis or PostgreSQL
- 🌍 **Multilingual support** — Extend pipeline to support multiple languages
- 📊 **Dashboard analytics** — Track trends in misinformation over time

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  Built with ❤️ using Python, FastAPI, and AI
</p>
