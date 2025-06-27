# 🚀 Vintern-1B-v3_5 Production Guide

## 📁 File Structure (Clean Production Setup)

```
ViScantool/
├── working_vintern_server.py      # 🔧 Main API server
├── working_vintern_client.py      # 🐍 Python client library
├── setup_internvl.py             # ⚙️ Dependency installer
├── README.md                      # 📖 Main documentation
├── SUCCESS_GUIDE.md               # 🎉 Success documentation
└── .venv/                         # 🐍 Python virtual environment
```

## 🎯 Core Files

### 1. `working_vintern_server.py`
- **Purpose**: Production-ready FastAPI server
- **Model**: Vintern-1B-v3_5 (938M parameters)
- **Features**: Vietnamese + English text/image processing
- **Endpoints**: `/health`, `/chat`, `/chat/upload`
- **URL**: http://localhost:8000

### 2. `working_vintern_client.py`
- **Purpose**: Python client for easy integration
- **Class**: `WorkingVinternClient`
- **Methods**: `.chat()`, `.vietnamese_ocr()`, `.document_analysis()`
- **Features**: Health checks, image processing, error handling

### 3. `setup_internvl.py`
- **Purpose**: Install all required dependencies
- **Installs**: PyTorch, Transformers, FastAPI, etc.
- **Usage**: Run once to set up environment

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
python setup_internvl.py

# 2. Start server (wait ~15 seconds for model loading)
python working_vintern_server.py

# 3. Test in another terminal
python working_vintern_client.py

# 4. Health check
curl http://localhost:8000/health
```

## 🔧 API Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
# Response: {"status":"healthy","model_loaded":true,"model":"5CD-AI/Vintern-1B-v3_5","device":"cuda","parameters":"~938M"}
```

### Simple Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Xin chào!", "max_tokens": 100}'
```

### Python Client
```python
from working_vintern_client import WorkingVinternClient

# Create client
client = WorkingVinternClient()

# Text chat
response = client.chat("Bạn có thể giúp tôi phân tích tài liệu không?")

# Vietnamese OCR
text = client.vietnamese_ocr("document.jpg")

# Document Q&A
answer = client.document_analysis("receipt.jpg", "Tổng số tiền là bao nhiêu?")
```

## 🎯 Production Capabilities

### ✅ Working Features
- **Vietnamese OCR**: Extract text from images
- **Document Analysis**: Answer questions about documents
- **Receipt Processing**: Analyze receipts and invoices
- **Bilingual Chat**: Vietnamese and English support
- **Image Description**: Describe images in Vietnamese
- **API Integration**: RESTful API for any application

### 📊 Performance
- **Model Loading**: ~15 seconds (first startup)
- **Response Time**: 2-5 seconds per request
- **Memory Usage**: ~2GB GPU VRAM
- **GPU Support**: CUDA with bfloat16 precision
- **CPU Fallback**: Available if no GPU

## 🔒 Production Considerations

### Security
- Server runs on `0.0.0.0:8000` (network accessible)
- No authentication (add if needed for production)
- Local model execution (data stays private)

### Scaling
- Single model instance per server
- Concurrent requests supported
- Can run multiple instances on different ports
- Consider load balancer for high traffic

### Monitoring
- Health endpoint for status checks
- Logging via Python logging module
- GPU memory monitoring with `nvidia-smi`

## 🛠️ Customization

### Server Configuration
Edit `working_vintern_server.py`:
```python
# Change port
port = 8001

# Adjust generation parameters
generation_config = dict(
    max_new_tokens=2048,
    temperature=0.1,
    num_beams=5
)
```

### Client Configuration
```python
# Custom server URL
client = WorkingVinternClient("http://your-server:8000")

# Custom timeouts
response = client.chat("message", max_tokens=2048)
```

## 🐛 Troubleshooting

### Common Issues
1. **Port 8000 in use**: `sudo lsof -i :8000` then kill process
2. **Model won't load**: Check internet connection and disk space
3. **CUDA out of memory**: Restart or use CPU mode
4. **Slow responses**: Check GPU availability with `nvidia-smi`

### Logs Location
- Server logs: Console output from `working_vintern_server.py`
- Client logs: Console output from client calls
- Model cache: `~/.cache/huggingface/`

## 📈 Next Steps

### Integration Ideas
- **Web App**: Add HTML frontend
- **Database**: Store analysis results
- **Authentication**: Add user management
- **Docker**: Containerize for deployment
- **Kubernetes**: Scale across cluster

### Performance Optimization
- **Quantization**: Reduce model size
- **Caching**: Cache frequent responses
- **Batch Processing**: Handle multiple images
- **Load Balancing**: Distribute requests

---

**This is your production-ready Vietnamese AI document processing system!** 🇻🇳🚀
