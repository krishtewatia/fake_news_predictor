---
title: Fake News Detector
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app_hf.py
pinned: false
license: mit
---

# 🔍 Fake News Detector

An AI-powered fake news detection system that analyzes text or URLs using advanced NLP, web evidence gathering, and LLM-based reasoning.

## Features

- ✅ **Real-time Fact-Checking**: Detect misinformation instantly
- 🔗 **Web Evidence Search**: Cross-reference with multiple sources
- 🤖 **AI-Powered Reasoning**: Uses advanced NLP models for verification
- 📊 **Detailed Analysis**: Get confidence scores, evidence sources, and explanations
- 🎯 **Multi-stage Pipeline**: 14-stage verification process

## How It Works

1. **Input Processing**: Accept text or URL
2. **Claim Extraction**: Identify factual claims using NLP
3. **Query Generation**: Optimize searches using AI
4. **Web Evidence**: Search NewsAPI and Google for corroborating/refuting sources
5. **Verification**: Use Sentence-Transformers and BART NLI for analysis
6. **Verdict**: Generate final verdict with confidence score
7. **Explanation**: Provide AI-generated reasoning

## Technologies

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **NLP**: spaCy, HuggingFace Transformers, Sentence-Transformers
- **APIs**: OpenRouter, NewsAPI, SerpAPI

## Setup Requirements

Before deploying, add these secrets in your HF Space settings:

```
OPEN_ROUTER_KEY=your_key_here
NEWSAPI_KEY=your_key_here
SERPAPI_KEY=your_key_here
```

Get these keys from:
- [OpenRouter](https://openrouter.ai)
- [NewsAPI](https://newsapi.org)
- [SerpAPI](https://serpapi.com)

---

Built with ❤️ using Python, FastAPI, and AI
