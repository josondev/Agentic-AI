#!/bin/bash
# Start script for Render deployment

echo "🚀 Starting Agentic AI System on Render..."

# Use gunicorn for production
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 render_app:app
