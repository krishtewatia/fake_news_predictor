# 🔍 AI-Powered Fake News Detection Pipeline

A sophisticated **Hybrid AI + Machine Learning pipeline** designed to verify factual claims and detect misinformation. The system analyzes raw text or article URLs through a multi-stage NLP process, cross-references with real-time web evidence, and delivers a reasoned verdict powered by **Deep Learning** and **Large Language Models**.

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

## 🛠️ Frameworks & Libraries

This project leverages state-of-the-art Python libraries for NLP and web services:

| Layer | Framework/Library | Purpose |
| :--- | :--- | :--- |
| **Backend** | **FastAPI** | High-performance, asynchronous REST API. |
| **Frontend** | **Streamlit** | Interactive dashboard for user input and visualization. |
| **NLP Core** | **spaCy** (`en_core_web_sm`) | Sentence tokenization and entity extraction. |
| **Deep Learning** | **HuggingFace Transformers** | Running the BART model for NLI/Stance detection. |
| **Embeddings** | **Sentence-Transformers** | Generating semantic vectors for similarity comparison. |
| **Scraping** | **newspaper3k** / **BS4** | Robust article extraction and HTML parsing. |
| **Validation** | **Pydantic v2** | Strict data modeling and request validation. |

---

## 📡 External APIs

The system integrates several external services to provide real-time accuracy:

1.  **OpenRouter (Gemini 2.0 Flash):**
    *   *Purpose:* Query optimization and natural language explanation generation.
    *   *Why:* Provides high-speed, cost-effective LLM reasoning.
2.  **NewsAPI:**
    *   *Purpose:* Sourcing recent news articles from thousands of verified global publishers.
3.  **SerpAPI:**
    *   *Purpose:* Accessing Google Search results to find evidence in the broader web index.

---

## 📁 Project Structure

```text
fake_news_detector/
├── pipeline/
│   ├── input_layer.py          # Data ingestion & cleaning
│   ├── claim_extractor.py      # NLP-based factual claim identification
│   ├── query_generator.py      # LLM-assisted search optimization
│   ├── web_search.py           # Multi-source API integration
│   ├── evidence_collector.py   # Article scraping & snippet extraction
│   ├── semantic_similarity.py  # MiniLM-based vector comparison
│   ├── stance_detector.py      # BART NLI classification
│   ├── credibility_scorer.py   # Domain trust-based weighting
│   ├── hybrid_scorer.py        # Final score calculation logic
│   ├── verdict.py              # Statistical decision engine
│   └── explainer.py            # Natural language reasoning
├── main.py                     # FastAPI server & pipeline orchestration
├── app.py                      # Streamlit UI
└── requirements.txt            # System dependencies
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
*   Python 3.10 or higher
*   Virtual environment (recommended)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
OPEN_ROUTER_KEY=your_key_here
NEWSAPI_KEY=your_key_here
SERPAPI_KEY=your_key_here
```

### 4. Running the Application
**Terminal 1 (Backend):**
```bash
uvicorn main:app --reload
```

**Terminal 2 (Frontend):**
```bash
streamlit run app.py
```


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
