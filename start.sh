#!/bin/bash
# Startup script for Hugging Face Spaces
# Robustly manages both FastAPI and Streamlit processes

set -e

echo "🔍 Fake News Detector - HF Spaces Startup"
echo "=========================================="

# Create necessary directories
mkdir -p logs

# Log file for debugging
LOG_FILE="logs/startup.log"

echo "$(date): Starting services..." | tee -a $LOG_FILE

# Start FastAPI backend
echo "📡 Starting FastAPI backend (http://0.0.0.0:8000)..." | tee -a $LOG_FILE
nohup uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-level info \
    > logs/fastapi.log 2>&1 &

FASTAPI_PID=$!
echo "FastAPI PID: $FASTAPI_PID" | tee -a $LOG_FILE

# Wait for FastAPI to be ready
echo "⏳ Waiting for FastAPI to initialize..." | tee -a $LOG_FILE
sleep 10

# Check if FastAPI is running
if ! kill -0 $FASTAPI_PID 2>/dev/null; then
    echo "❌ FastAPI failed to start. Check logs/fastapi.log" | tee -a $LOG_FILE
    cat logs/fastapi.log
    exit 1
fi

echo "✅ FastAPI is running" | tee -a $LOG_FILE

# Start Streamlit
echo "🎨 Starting Streamlit frontend (http://0.0.0.0:7860)..." | tee -a $LOG_FILE
exec streamlit run app_hf.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --logger.level=info \
    2>&1 | tee -a logs/streamlit.log

# Cleanup on exit
trap "kill $FASTAPI_PID 2>/dev/null || true" EXIT
