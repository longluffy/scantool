#!/bin/bash

echo "🇻🇳 Vietnamese OCR System with Vintern-1B-v3.5"
echo "================================================"
echo ""
echo "This system uses Vintern-1B-v3.5 AI model for accurate Vietnamese text extraction."
echo ""

# Check if server is running
echo "🔍 Checking Vintern server status..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Vintern server is running!"
    
    # Get server details
    echo ""
    echo "📊 Server Information:"
    curl -s http://localhost:8000/health | python -c "
import json, sys
data = json.load(sys.stdin)
print(f\"   Status: {data.get('status', 'unknown')}\" )
print(f\"   Model: {data.get('model', 'unknown')}\")
print(f\"   Device: {data.get('device', 'unknown')}\")
print(f\"   Parameters: {data.get('parameters', 'unknown')}\")
print(f\"   Model Loaded: {data.get('model_loaded', False)}\")
"
else
    echo "❌ Vintern server is not running!"
    echo ""
    echo "To start the server, run:"
    echo "   ./start_vintern_server.sh"
    echo ""
    echo "The server will:"
    echo "   • Download Vintern-1B-v3.5 model (if first time)"
    echo "   • Load the model into memory"
    echo "   • Start API server on http://localhost:8000"
    echo ""
    exit 1
fi

echo ""
echo "🖥️  Starting Vietnamese OCR GUI Application..."
echo ""
echo "✨ Features:"
echo "   • Modern wxPython interface"
echo "   • Real-time file preview"
echo "   • Batch processing"
echo "   • Vietnamese text extraction using Vintern-1B-v3.5"
echo "   • Progress tracking"
echo ""

# Start the GUI application
python main.py
