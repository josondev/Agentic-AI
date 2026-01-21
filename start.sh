#!/bin/bash
# Start script for Render deployment

echo "🚀 Starting Agentic AI System on Render..."
echo "📦 Python version: $(python --version)"
echo "🔧 Gunicorn version: $(gunicorn --version)"

# Use gunicorn for production
# --workers 1: Single worker (sufficient for free tier)
# --threads 2: Two threads per worker
# --timeout 120: 2 minute timeout for long queries
# --bind 0.0.0.0:$PORT: Listen on all interfaces
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 --log-level info render_app:app
