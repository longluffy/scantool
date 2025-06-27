#!/bin/bash

echo "🚀 Starting Vintern-1B-v3.5 OCR Server"
echo "======================================"

# Check if the server is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Server is already running on port 8000"
    echo "📊 Server health:"
    curl -s http://localhost:8000/health | python -m json.tool
    exit 0
fi

echo "🔧 Starting new server instance..."
echo "📝 Server will load Vintern-1B-v3.5 model automatically"
echo "⏳ This may take a few minutes for first-time setup..."
echo ""

# Navigate to the OCR server directory
cd "$(dirname "$0")/ocrServer"

# Start the server
python working_vintern_server.py
