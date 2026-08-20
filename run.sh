#!/bin/bash
# Startup script for Hugging Face Spaces
# Runs both FastAPI backend and Streamlit frontend

echo "🚀 Starting Fake News Detector on Hugging Face Spaces..."

# Start FastAPI backend in the background
echo "Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for backend to be ready
sleep 5

# Start Streamlit frontend
echo "Starting Streamlit frontend on port 8501..."
streamlit run app_hf.py --server.port=7860 --server.address=0.0.0.0

# Cleanup on exit
trap "kill $FASTAPI_PID" EXIT
