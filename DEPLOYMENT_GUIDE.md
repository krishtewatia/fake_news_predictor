# 🚀 Deployment Guide: Fake News Detector to Hugging Face Spaces

## Step 1: Prepare Your Local Files

All necessary files have been created:
- ✅ `app_hf.py` - Streamlit app configured for HF Spaces
- ✅ `run.sh` - Startup script to run both backend and frontend
- ✅ `README_HF.md` - Proper HF Spaces metadata file
- ✅ Original files (main.py, requirements.txt, pipeline/, etc.)

## Step 2: Create HF Space

1. Go to **https://huggingface.co/spaces**
2. Click **Create new Space**
3. Fill in:
   - **Space name**: `fake-news-detector` (or your preferred name)
   - **License**: MIT
   - **Visibility**: Public (or Private)
   - **SDK**: Streamlit
4. Click **Create Space**

## Step 3: Clone Your HF Space

```bash
git clone https://huggingface.co/spaces/krishtewatia/fake-news-detector
cd fake-news-detector
```

Replace `krishtewatia` with your HF username and `fake-news-detector` with your space name.

## Step 4: Add Your Project Files

Copy all files from your local project to the cloned space directory:

```bash
# From your HF space directory, copy all project files
cp -r /path/to/your/project/fake_news_detector/* .
```

Key files needed:
```
.
├── app_hf.py (rename from app.py if using HF Spaces)
├── main.py
├── run.sh
├── README_HF.md (rename to README.md for HF)
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── input_layer.py
│   ├── claim_extractor.py
│   ├── query_generator.py
│   ├── web_search.py
│   ├── evidence_collector.py
│   ├── semantic_similarity.py
│   ├── evidence_ranker.py
│   ├── stance_detector.py
│   ├── credibility_scorer.py
│   ├── hybrid_scorer.py
│   ├── aggregator.py
│   ├── verdict.py
│   ├── explainer.py
│   └── cache.py
├── config.py
└── .env.example
```

## Step 5: Configure Secrets

1. In your HF Space, go to **Settings → Variables and Secrets**
2. Add these secrets (get from the respective services):
   ```
   OPEN_ROUTER_KEY=<your_key>
   NEWSAPI_KEY=<your_key>
   SERPAPI_KEY=<your_key>
   ```
3. Any other API keys from your `.env` file

## Step 6: Update README.md

Rename `README_HF.md` to `README.md`:
```bash
mv README_HF.md README.md
```

This file contains the HF-compatible YAML frontmatter.

## Step 7: Commit and Push

```bash
git add .
git commit -m "Deploy fake news detector to HF Spaces"
git push
```

HF Spaces will automatically:
1. Install dependencies from `requirements.txt`
2. Download spaCy models
3. Run your Streamlit app

## Step 8: Access Your Space

Your space will be live at:
```
https://huggingface.co/spaces/krishtewatia/fake-news-detector
```

The Streamlit app will be accessible via the embedded URL.

---

## ⚙️ Configuration Details

### `app_hf.py`
- Modified version of `app.py` for HF Spaces
- Uses environment variable `API_URL` (defaults to `http://localhost:8000`)
- Handles backend connection errors gracefully

### `run.sh`
- Starts FastAPI backend on port 8000
- Waits 5 seconds for backend to initialize
- Starts Streamlit on port 7860 (HF Spaces uses 7860)

### `requirements.txt`
- All dependencies already specified
- HF Spaces will install automatically

### API Keys
- Add via HF Spaces Settings → Variables and Secrets
- Never commit `.env` with real keys
- Use `.env.example` as template

---

## 🔧 Troubleshooting

### "Backend is not responding"
- The backend may be loading. Wait 10-15 seconds.
- Check if `OPEN_ROUTER_KEY`, `NEWSAPI_KEY`, and `SERPAPI_KEY` are set in secrets.

### "Module not found" errors
- Ensure all pipeline files are uploaded
- Check `pipeline/__init__.py` exists
- Verify `requirements.txt` has all dependencies

### Slow response times
- NLP models are large (first run downloads them)
- Web searches can take time
- Timeout set to 120 seconds

### Space keeps restarting
- Check error logs in Space settings
- Verify all API keys are valid
- Ensure no hardcoded paths (use relative paths)

---

## 📝 Notes

- **First load**: May take 2-3 minutes as models download
- **Model sizes**: BART + Sentence-Transformers ≈ 2GB
- **Storage**: HF Spaces provides 50GB by default
- **GPU**: Optional (CPU is sufficient)

---

## 🎯 Your Space URL
Once deployed:
```
https://huggingface.co/spaces/krishtewatia/fake-news-detector
```

Share this link with anyone to use your detector!

---

Need help? Check HF Spaces docs: https://huggingface.co/docs/hub/spaces
