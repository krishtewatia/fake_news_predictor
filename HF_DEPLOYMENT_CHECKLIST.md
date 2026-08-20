# 🎯 Hugging Face Spaces Deployment Checklist

## Pre-Deployment (Local)

- [ ] Test project runs locally with `uvicorn main:app --reload` + `streamlit run app_hf.py`
- [ ] All API keys work (OPEN_ROUTER_KEY, NEWSAPI_KEY, SERPAPI_KEY)
- [ ] `.gitignore` includes `.env` (don't commit real keys)
- [ ] All pipeline files are present and working
- [ ] `requirements.txt` has all dependencies
- [ ] `README.md` has the YAML frontmatter (see `README_FOR_HF.md`)

## HF Space Setup

### Create Space
- [ ] Go to https://huggingface.co/spaces
- [ ] Click "Create new Space"
- [ ] Choose **Streamlit** SDK
- [ ] Space name: `fake-news-detector`
- [ ] License: MIT
- [ ] Visibility: Public
- [ ] Click "Create Space"

### Clone to Local
```bash
git clone https://huggingface.co/spaces/krishtewatia/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
```

### Add Files
Copy all these files to your cloned space:
```
✅ app_hf.py (main Streamlit app)
✅ main.py (FastAPI backend)
✅ start.sh (startup script)
✅ requirements.txt
✅ README.md (with YAML frontmatter from README_FOR_HF.md)
✅ DEPLOYMENT_GUIDE.md
✅ config.py
✅ .env.example
✅ pipeline/ (entire directory)
✅ .gitignore (to exclude .env)
```

### Add Secrets
In HF Space Settings → "Variables and Secrets":
- [ ] `OPEN_ROUTER_KEY` = your key
- [ ] `NEWSAPI_KEY` = your key
- [ ] `SERPAPI_KEY` = your key

(Never commit these to git!)

## Deployment

```bash
# From your HF space directory
git add .
git commit -m "Deploy fake news detector"
git push
```

Space will auto-deploy. Wait 5-10 minutes.

## Verification

- [ ] Space shows "Running" status
- [ ] Can open Space URL
- [ ] Streamlit loads without errors
- [ ] Can enter text/URL
- [ ] Can click "Check" button
- [ ] Backend responds with results

## Post-Deployment

- [ ] Share Space URL: `https://huggingface.co/spaces/krishtewatia/fake-news-detector`
- [ ] Test with sample claims
- [ ] Monitor logs for errors
- [ ] Update README with any tweaks

## Files You Created

| File | Purpose |
| --- | --- |
| `app_hf.py` | Streamlit UI (HF-optimized) |
| `start.sh` | Startup script |
| `README_FOR_HF.md` | Copy this to README.md |
| `DEPLOYMENT_GUIDE.md` | Detailed instructions |
| This file | Checklist |

## Common Issues

### "Cannot connect to backend"
- Backend loading? Wait 15 seconds
- Check API keys in Space Secrets
- Check `start.sh` logs

### "Module not found"
- Verify `pipeline/` folder structure
- Check `pipeline/__init__.py` exists
- Reinstall: `pip install -r requirements.txt`

### "API key invalid"
- Generate new keys from OpenRouter, NewsAPI, SerpAPI
- Add to HF Space Secrets (Settings → Variables and Secrets)
- Restart Space

### Space keeps crashing
- Check Space logs (Settings → Logs)
- Verify no hardcoded paths (use relative)
- Ensure Python 3.10+

## Next Steps

1. ✅ Create Space on HF
2. ✅ Add secrets
3. ✅ Push code
4. ✅ Wait for deployment
5. ✅ Test thoroughly
6. ✅ Share your Space URL!

**Your Space URL**: 
```
https://huggingface.co/spaces/krishtewatia/fake-news-detector
```

---

For detailed setup: See `DEPLOYMENT_GUIDE.md`
For code changes: See `README_FOR_HF.md`
