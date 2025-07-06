# OCR Server Logging Implementation

## ✅ COMPLETED LOGGING ENHANCEMENTS

### 1. **Multiple Logger Configuration**
- **General Logger**: Standard server operations and request handling
- **OCR Debug Logger**: Detailed content logging for debugging OCR results
- **Performance Logger**: Processing time and performance metrics
- **Timestamp Format**: `YYYY-MM-DD HH:MM:SS` for all log entries

### 2. **Request Tracking System**
- **Unique Request IDs**: Each request gets a unique ID (`req_<timestamp>`, `batch_<timestamp>`, `upload_<timestamp>`)
- **Request Lifecycle**: Track from request start to completion
- **Error Tracking**: Log errors with request context and timing

### 3. **Detailed Request Logging**
Each request now logs:
- **Request Details**: Message length, parameters, image presence
- **Image Processing**: Image size, processing time
- **Generation Config**: Model parameters used
- **Response Metrics**: Response length, word count, line count
- **Performance Timing**: Processing time breakdown

### 4. **Content Logging for Debugging**
- **Input Messages**: Full message content (DEBUG level)
- **Response Content**: Complete OCR output (DEBUG level)
- **Error Details**: Full error context and stack traces

### 5. **Performance Metrics**
- **Image Processing Time**: Time spent processing/converting images
- **Generation Time**: Time spent by the model generating responses
- **Total Request Time**: End-to-end request processing time
- **Batch Processing**: Per-page timing and batch averages

## 📊 LOGGING LEVELS

### INFO Level (General Operations)
```
2025-07-05 14:30:15 - __main__ - INFO - [req_1720180215123] New chat request received
2025-07-05 14:30:15 - __main__ - INFO - [req_1720180215123] Message length: 89 characters
2025-07-05 14:30:15 - __main__ - INFO - [req_1720180215123] Has image: True
2025-07-05 14:30:15 - __main__ - INFO - [req_1720180215123] Image size: (800, 600)
2025-07-05 14:30:17 - __main__ - INFO - [req_1720180215123] Response summary: 15 lines, 245 words
```

### DEBUG Level (OCR Content)
```
2025-07-05 14:30:15 - OCR_DEBUG - [req_1720180215123] Input message: Hãy trích xuất toàn bộ văn bản trong hình ảnh này
2025-07-05 14:30:17 - OCR_DEBUG - [req_1720180215123] Response content: CÔNG TY TNHH ABC
Địa chỉ: 123 Nguyễn Huệ, Quận 1, TP.HCM
[... full OCR output ...]
```

### PERFORMANCE Level (Timing)
```
2025-07-05 14:30:15 - PERFORMANCE - [req_1720180215123] Image processing time: 0.245s
2025-07-05 14:30:17 - PERFORMANCE - [req_1720180215123] Generation time: 1.832s
2025-07-05 14:30:17 - PERFORMANCE - [req_1720180215123] Total processing time: 2.089s
```

## 🔧 LOGGING CONFIGURATION

### Logger Setup
```python
# General logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# OCR content debugging logger
ocr_logger = logging.getLogger('ocr_debug')
ocr_logger.setLevel(logging.DEBUG)

# Performance metrics logger
perf_logger = logging.getLogger('performance')
perf_logger.setLevel(logging.INFO)
```

### Request ID Generation
```python
request_id = f"req_{int(time.time() * 1000)}"
batch_id = f"batch_{int(time.time() * 1000)}"
upload_id = f"upload_{int(time.time() * 1000)}"
```

## 🧪 TESTING THE LOGGING

### Test Script
Run the test script to verify logging works:
```bash
python test_ocr_logging.py
```

### Manual Testing
1. Start the server: `python ocrServer/working_vintern_server.py`
2. Watch the console for detailed logs
3. Send test requests to see logging in action

## 📋 LOG OUTPUT EXAMPLES

### Server Startup
```
2025-07-05 14:25:30 - __main__ - INFO - ============================================================
2025-07-05 14:25:30 - __main__ - INFO - 🚀 Starting Vintern-1B-v3.5 API Server...
2025-07-05 14:25:30 - __main__ - INFO - ============================================================
2025-07-05 14:25:30 - __main__ - INFO - Using device: cuda
2025-07-05 14:25:30 - __main__ - INFO - CUDA device: NVIDIA GeForce RTX 4090
2025-07-05 14:25:30 - PERFORMANCE - Model loading time: 15.234s
2025-07-05 14:25:30 - __main__ - INFO - ✅ Model loaded successfully!
```

### Chat Request Processing
```
2025-07-05 14:30:15 - __main__ - INFO - [req_1720180215123] New chat request received
2025-07-05 14:30:15 - __main__ - INFO - [req_1720180215123] Message length: 89 characters
2025-07-05 14:30:15 - __main__ - INFO - [req_1720180215123] Has image: True
2025-07-05 14:30:15 - __main__ - INFO - [req_1720180215123] Max tokens: 1024
2025-07-05 14:30:15 - __main__ - INFO - [req_1720180215123] Temperature: 0.0
2025-07-05 14:30:15 - OCR_DEBUG - [req_1720180215123] Input message: Hãy trích xuất toàn bộ văn bản...
2025-07-05 14:30:15 - __main__ - INFO - [req_1720180215123] Image size: (800, 600)
2025-07-05 14:30:15 - PERFORMANCE - [req_1720180215123] Image processing time: 0.245s
2025-07-05 14:30:17 - PERFORMANCE - [req_1720180215123] Generation time: 1.832s
2025-07-05 14:30:17 - PERFORMANCE - [req_1720180215123] Total processing time: 2.089s
2025-07-05 14:30:17 - __main__ - INFO - [req_1720180215123] Response summary: 15 lines, 245 words
2025-07-05 14:30:17 - OCR_DEBUG - [req_1720180215123] Response content: CÔNG TY TNHH ABC...
```

### Batch Processing
```
2025-07-05 14:35:20 - __main__ - INFO - [batch_1720180520456] New batch OCR request received
2025-07-05 14:35:20 - __main__ - INFO - [batch_1720180520456] Number of images: 3
2025-07-05 14:35:20 - __main__ - INFO - [batch_1720180520456_page_1] Processing page 1/3
2025-07-05 14:35:20 - PERFORMANCE - [batch_1720180520456_page_1] Total page processing time: 2.156s
2025-07-05 14:35:22 - __main__ - INFO - [batch_1720180520456_page_2] Processing page 2/3
2025-07-05 14:35:24 - PERFORMANCE - [batch_1720180520456] Total batch processing time: 6.789s
2025-07-05 14:35:24 - PERFORMANCE - [batch_1720180520456] Average time per page: 2.263s
```

## 🎯 BENEFITS FOR DEBUGGING

1. **Request Tracing**: Track individual requests from start to finish
2. **Performance Analysis**: Identify bottlenecks in processing pipeline
3. **Content Verification**: Verify OCR output quality and accuracy
4. **Error Debugging**: Detailed error context with timing information
5. **Batch Processing**: Monitor multi-page document processing
6. **Resource Usage**: Track processing times for optimization

## 📝 USAGE INSTRUCTIONS

### Enable Debug Content Logging
To see the full OCR content in logs, set the OCR debug logger to DEBUG level:
```python
ocr_logger.setLevel(logging.DEBUG)
```

### Production Logging
For production, you may want to disable content logging:
```python
ocr_logger.setLevel(logging.WARNING)  # Disable content logging
```

### Log File Output
To save logs to a file:
```python
file_handler = logging.FileHandler('ocr_server.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
```

The logging system provides comprehensive debugging capabilities for the OCR server, making it easy to troubleshoot issues and optimize performance for Vietnamese legal document processing.
