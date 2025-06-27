# Vintern-1B-v3_5 Local Hosting 🇻🇳

This project enables you to host and run [Vintern-1B-v3_5](https://huggingface.co/5CD-AI/Vintern-1B-v3_5) locally on your Ubuntu 22.04 system. Vintern-1B-v3_5 is a Vietnamese multimodal model specifically designed for OCR, document analysis, and visual question answering.

**✅ PRODUCTION READY** - This is a clean, working implementation!

## 🌟 Features

- **Vietnamese OCR & Document Analysis**: Extract text and answer questions about Vietnamese documents
- **Receipt Processing**: Analyze receipts and invoices in Vietnamese
- **Chart & Diagram Understanding**: Interpret charts, graphs, and technical diagrams  
- **General Visual Q&A**: Answer questions about images in Vietnamese and English
- **Local API Server**: Host the model locally with FastAPI
- **Easy Integration**: Simple Python client for your applications

## 📋 Requirements

### System Requirements
- **OS**: Ubuntu 22.04 (or similar Linux)
- **Python**: 3.8+
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ free space for model
- **GPU**: Optional but recommended (CUDA-compatible)

### Model Specifications
- **Size**: ~938M parameters
- **Components**: InternViT-300M-448px + Qwen2-0.5B-Instruct
- **Languages**: Vietnamese (primary), English (secondary)
- **Context Length**: 4096 tokens
- **Base Model**: InternVL2.5-1B

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run the setup script to install dependencies
python setup_internvl.py
```

This will install:
- PyTorch & TorchVision  
- Transformers
- FastAPI & Uvicorn
- Required image processing libraries

### 2. Start the API Server

```bash
# Start the Vintern-1B-v3_5 API server
python working_vintern_server.py
```

The server will:
- Download the model from Hugging Face (first run only)
- Load the model into memory (~15 seconds)
- Start serving on `http://localhost:8000`

### 3. Test the Setup

```bash
# Check server health
curl http://localhost:8000/health

# Test Vietnamese chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Xin chào!", "max_tokens": 100}'
```

### 4. Use the Client

```bash
# Try the Python client
python working_vintern_client.py
```
# Run comprehensive tests
python test_vintern.py
```

### 4. Use the Client

```bash
# Try the example client
python vintern_client.py
```

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Text Chat
```python
import requests

response = requests.post("http://localhost:8000/chat", 
    json={
        "message": "Xin chào! Bạn có thể giúp tôi gì?",
        "max_tokens": 1024
    }
)
print(response.json()["response"])
```

### Image Analysis
```python
import base64
import requests

# Read and encode image
with open("document.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/chat",
    json={
        "message": "Mô tả hình ảnh một cách chi tiết.",
        "image_base64": image_data,
        "max_tokens": 1024
    }
)
print(response.json()["response"])
```

## 🐍 Python Client

```python
from working_vintern_client import WorkingVinternClient

# Create client
client = WorkingVinternClient("http://localhost:8000")

# Check health
if client.is_healthy():
    # Text chat
    response = client.chat("Tôi có thể hỏi gì?")
    
    # Image analysis
    response = client.vietnamese_ocr("receipt.jpg")
    
    # Document analysis
    response = client.document_analysis("contract.jpg", "Ai là người ký hợp đồng?")
```

## 🎯 Use Cases

### 1. Receipt Processing
```python
client = WorkingVinternClient()
response = client.chat(
    "Tổng số tiền trong hóa đơn này là bao nhiêu?",
    image_path="receipt.jpg"
)
```

### 2. Document OCR
```python
response = client.chat(
    "Trích xuất tất cả văn bản trong tài liệu này.",
    image_path="document.jpg"
)
```

### 3. Chart Analysis
```python
response = client.chat(
    "Giải thích biểu đồ này.",
    image_path="chart.png"
)
```

## ⚙️ Configuration

### Server Configuration
Edit `vintern_api_server.py` to customize:
- Server host/port
- Model parameters
- Generation settings

### Client Configuration
```python
from working_vintern_client import WorkingVinternClient

client = WorkingVinternClient(
    server_url="http://your-server:8000"
)
```

## 🔧 Advanced Usage

### Custom Generation Parameters
```python
response = client.chat(
    message="Phân tích tài liệu này",
    image_path="document.jpg",
    max_tokens=2048,
    temperature=0.0,        # Deterministic output
    num_beams=3,           # Beam search
    repetition_penalty=2.5  # Avoid repetition
)
```

### Batch Processing
```python
documents = ["doc1.jpg", "doc2.jpg", "doc3.jpg"]
results = []

for doc in documents:
    result = client.ocr_question(doc)
    results.append(result)
```

## 🐛 Troubleshooting

### Common Issues

**Server won't start:**
- Check Python dependencies: `python setup_internvl.py`
- Verify available memory (8GB+ recommended)
- Check port 8000 is available

**Model loading fails:**
- Ensure internet connection for first download
- Check disk space (2GB+ required)
- Verify Hugging Face access

**Slow inference:**
- Use GPU if available (CUDA)
- Reduce max_tokens
- Use temperature=0.0 for faster generation

**Memory issues:**
- Close other applications
- Use CPU-only mode if GPU memory insufficient
- Reduce batch sizes

### Performance Optimization

**For GPU:**
```python
# Server will automatically use CUDA if available
# Model loads in bfloat16 for efficiency
```

**For CPU:**
```python
# Model automatically falls back to float32 on CPU
# Inference will be slower but functional
```

## 📊 Model Performance

Based on benchmarks:
- **OpenViVQA**: 7.7/10 score
- **ViTextVQA**: 7.7/10 score  
- **MTVQA**: 31.7% (Top 3 performance)

Optimized for:
- Vietnamese OCR and text recognition
- Document understanding
- Receipt and invoice processing
- Chart and diagram analysis

## 🤝 Integration Examples

### Web Application
```python
from flask import Flask, request, jsonify
from vintern_client import VinternClient

app = Flask(__name__)
client = VinternClient()

@app.route('/analyze', methods=['POST'])
def analyze_document():
    file = request.files['image']
    question = request.form['question']
    
    response = client.chat_with_image_bytes(
        question, 
        file.read()
    )
    
    return jsonify({'result': response})
```

### Batch Processing Script
```python
import os
from vintern_client import VinternClient

client = VinternClient()

# Process all images in folder
for filename in os.listdir('documents/'):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        result = client.ocr_question(f'documents/{filename}')
        
        # Save results
        with open(f'results/{filename}.txt', 'w') as f:
            f.write(result)
```

## 📚 Resources

- **Model**: [5CD-AI/Vintern-1B-v2](https://huggingface.co/5CD-AI/Vintern-1B-v2)
- **Paper**: [Vintern-1B: An Efficient Multimodal Large Language Model for Vietnamese](https://arxiv.org/abs/2408.12480)
- **Demo**: [Hugging Face Space](https://huggingface.co/spaces/khang119966/Vintern-v2-Demo)

## 📝 License

This project follows the MIT license of the Vintern-1B-v2 model.

## 🙋 Support

If you encounter issues:
1. Check the troubleshooting section
2. Run the test suite: `python test_vintern.py`
3. Verify model compatibility with your system

---

**Ready to process Vietnamese documents with AI!** 🚀🇻🇳
