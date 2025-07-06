#!/usr/bin/env python3
"""
Working Vintern-1B-v3.5 API Server
Based on the successful test results
"""

import os
import torch
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Union
import uvicorn
import logging
import time
from datetime import datetime
from transformers import AutoModel, AutoTokenizer

# Setup comprehensive logging with timestamp and formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create a separate logger for OCR content debugging
ocr_logger = logging.getLogger('ocr_debug')
ocr_logger.setLevel(logging.DEBUG)
ocr_handler = logging.StreamHandler()
ocr_handler.setFormatter(logging.Formatter('%(asctime)s - OCR_DEBUG - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
ocr_logger.addHandler(ocr_handler)

# Performance logger
perf_logger = logging.getLogger('performance')
perf_logger.setLevel(logging.INFO)
perf_handler = logging.StreamHandler()
perf_handler.setFormatter(logging.Formatter('%(asctime)s - PERFORMANCE - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
perf_logger.addHandler(perf_handler)

# Initialize FastAPI app
app = FastAPI(title="Vintern-1B-v3.5 API Server", version="1.0.0")

# Global model variables
model = None
tokenizer = None
device = None

# Image processing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class ChatRequest(BaseModel):
    message: str
    image_base64: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.0
    num_beams: Optional[int] = 3
    repetition_penalty: Optional[float] = 2.5

class ChatResponse(BaseModel):
    response: str
    model: str = "Vintern-1B-v3_5"

# Add batch processing model for legal documents
class BatchOCRRequest(BaseModel):
    messages: list[str]
    images_base64: list[str]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.0
    num_beams: Optional[int] = 1
    repetition_penalty: Optional[float] = 1.1

class BatchOCRResponse(BaseModel):
    responses: list[str]
    model: str = "Vintern-1B-v3_5"
    processing_time: float

def build_transform(input_size):
    """Build image transformation pipeline"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamic preprocessing for images"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):  # Increased for RTX 5080 - better quality
    """Load and process image for the model - optimized for RTX 5080 GPU"""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    
    transform = build_transform(input_size=input_size)
    # Increased max_num for RTX 5080 - can handle more patches for better accuracy
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model():
    """Load Vintern-1B-v3.5 model optimized for RTX 5080 GPU"""
    global model, tokenizer, device
    
    model_load_start = time.time()
    
    try:
        # Determine device and optimize for RTX 5080
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        if device == "cuda":
            # Enhanced GPU information logging
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 CUDA device: {gpu_name}")
            logger.info(f"💾 CUDA memory: {gpu_memory:.1f}GB")
            logger.info(f"🔧 CUDA version: {torch.version.cuda}")
            logger.info(f"⚡ cuDNN enabled: {torch.backends.cudnn.enabled}")
            
            # RTX 5080 optimization settings
            if "RTX 5080" in gpu_name or gpu_memory >= 15:
                logger.info("🎯 RTX 5080 detected - enabling high-performance optimizations")
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.cuda.empty_cache()  # Clear cache before loading
        
        model_path = "5CD-AI/Vintern-1B-v3_5"
        logger.info(f"Loading model from: {model_path}")
        
        tokenizer_start = time.time()
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            use_fast=False
        )
        tokenizer_time = time.time() - tokenizer_start
        perf_logger.info(f"Tokenizer loading time: {tokenizer_time:.3f}s")
        
        model_start = time.time()
        logger.info("Loading model...")
        if device == "cuda":
            # RTX 5080 optimized loading
            logger.info("🚀 Loading with RTX 5080 optimizations...")
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,  # Optimal for RTX 5080
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=True,  # Enable Flash Attention for RTX 5080
                device_map="auto",    # Automatic device mapping for optimal GPU usage
            ).eval().cuda()
            
            # Additional RTX 5080 optimizations
            if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matrix operations
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.allow_tf32 = True
                
        else:
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False,
                device_map=None
            ).eval()
        
        model_time = time.time() - model_start
        perf_logger.info(f"Model loading time: {model_time:.3f}s")
            
        logger.info("✅ Model loaded successfully!")
        
        # Log GPU memory usage after model loading
        if device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"📊 GPU memory allocated: {memory_allocated:.2f}GB")
            logger.info(f"📊 GPU memory reserved: {memory_reserved:.2f}GB")
        
        # Quick test with performance monitoring
        test_start = time.time()
        test_question = "Xin chào!"
        generation_config = dict(max_new_tokens=50, do_sample=False, num_beams=1, repetition_penalty=1.0)
        response, _ = model.chat(tokenizer, None, test_question, generation_config, history=None, return_history=True)
        test_time = time.time() - test_start
        
        logger.info(f"✅ Model test successful: {response}")
        perf_logger.info(f"Model test time: {test_time:.3f}s")
        
        total_load_time = time.time() - model_load_start
        perf_logger.info(f"Total model loading time: {total_load_time:.3f}s")
        
        return True
        
    except Exception as e:
        load_error_time = time.time() - model_load_start
        logger.error(f"Failed to load model after {load_error_time:.3f}s: {e}")
        logger.error("If Flash Attention failed, trying fallback configuration...")
        
        # Fallback without Flash Attention if it fails
        try:
            if device == "cuda":
                logger.info("🔄 Retrying without Flash Attention...")
                model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_flash_attn=False,  # Disable Flash Attention as fallback
                ).eval().cuda()
                logger.info("✅ Model loaded successfully with fallback configuration!")
                return True
        except Exception as e2:
            logger.error(f"Fallback loading also failed: {e2}")
        
        import traceback
        traceback.print_exc()
        return False

def process_image(image_data: Union[str, bytes, Image.Image]) -> Image.Image:
    """Process image from various input formats"""
    if isinstance(image_data, str):
        # Base64 encoded image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
    elif isinstance(image_data, bytes):
        image = Image.open(BytesIO(image_data))
    elif isinstance(image_data, Image.Image):
        image = image_data
    else:
        raise ValueError("Unsupported image format")
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    startup_time = time.time()
    logger.info("=" * 60)
    logger.info("🚀 Starting Vintern-1B-v3.5 API Server...")
    logger.info("=" * 60)
    
    success = load_model()
    
    startup_duration = time.time() - startup_time
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ Vintern-1B-v3.5 API Server is ready!")
        logger.info(f"✅ Server startup completed in {startup_duration:.3f}s")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ Failed to load model on startup!")
        logger.error(f"❌ Server startup failed after {startup_duration:.3f}s")
        logger.error("=" * 60)
        # Don't raise exception to allow server to start

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vintern-1B-v3_5 API Server is running!", 
        "model": "Vintern-1B-v3_5",
        "status": "healthy" if model is not None else "loading"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with RTX 5080 monitoring"""
    model_loaded = model is not None and tokenizer is not None
    
    health_info = {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "model": "5CD-AI/Vintern-1B-v3_5",
        "device": str(device),
        "parameters": "~938M"
    }
    
    # Add GPU monitoring for RTX 5080
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        free_memory = total_memory - reserved_memory
        
        health_info.update({
            "gpu_name": gpu_name,
            "gpu_memory": {
                "total_gb": round(total_memory, 2),
                "allocated_gb": round(allocated_memory, 2),
                "reserved_gb": round(reserved_memory, 2),
                "free_gb": round(free_memory, 2),
                "utilization_percent": round((allocated_memory / total_memory) * 100, 1)
            },
            "cuda_version": torch.version.cuda,
            "flash_attention": "enabled" if hasattr(model, 'use_flash_attn') else "unknown",
            "optimizations": {
                "tf32_enabled": getattr(torch.backends.cuda.matmul, 'allow_tf32', False) if hasattr(torch.backends, 'cuda') else False,
                "cudnn_benchmark": torch.backends.cudnn.benchmark if hasattr(torch.backends, 'cudnn') else False
            }
        })
    
    return health_info

@app.get("/gpu")
async def gpu_status():
    """RTX 5080 GPU monitoring endpoint"""
    if device != "cuda" or not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    try:
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "memory": {
                "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                "free_gb": round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3, 2)
            },
            "properties": {
                "major": torch.cuda.get_device_properties(0).major,
                "minor": torch.cuda.get_device_properties(0).minor,
                "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count,
                "max_threads_per_multiprocessor": torch.cuda.get_device_properties(0).max_threads_per_multi_processor,
                "shared_memory_per_block": torch.cuda.get_device_properties(0).shared_memory_per_block
            },
            "optimizations": {
                "flash_attention": True,  # Enabled for RTX 5080
                "tf32_matmul": getattr(torch.backends.cuda.matmul, 'allow_tf32', False) if hasattr(torch.backends, 'cuda') else False,
                "cudnn_benchmark": torch.backends.cudnn.benchmark if hasattr(torch.backends, 'cudnn') else False,
                "cudnn_enabled": torch.backends.cudnn.enabled if hasattr(torch.backends, 'cudnn') else False
            }
        }
        
        return gpu_info
        
    except Exception as e:
        return {"error": f"Failed to get GPU status: {str(e)}"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for text and multimodal conversations"""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet, please wait")
        
        # Log request details
        logger.info(f"[{request_id}] New chat request received")
        logger.info(f"[{request_id}] Message length: {len(request.message)} characters")
        logger.info(f"[{request_id}] Has image: {request.image_base64 is not None}")
        logger.info(f"[{request_id}] Max tokens: {request.max_tokens}")
        logger.info(f"[{request_id}] Temperature: {request.temperature}")
        logger.info(f"[{request_id}] Num beams: {request.num_beams}")
        logger.info(f"[{request_id}] Repetition penalty: {request.repetition_penalty}")
        
        # Log the actual message content for debugging
        ocr_logger.debug(f"[{request_id}] Input message: {request.message}")
        
        # Process image if provided
        pixel_values = None
        image_process_start = time.time()
        if request.image_base64:
            try:
                image = process_image(request.image_base64)
                logger.info(f"[{request_id}] Image size: {image.size}")
                
                # RTX 5080 optimized image processing - higher quality
                pixel_values = load_image(image, max_num=12)  # Increased for RTX 5080
                if device == "cuda":
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                else:
                    pixel_values = pixel_values.to(torch.float32)
                
                image_process_time = time.time() - image_process_start
                perf_logger.info(f"[{request_id}] Image processing time: {image_process_time:.3f}s")
                
                # Log GPU memory usage for monitoring
                if device == "cuda":
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    perf_logger.info(f"[{request_id}] GPU memory after image processing: {memory_allocated:.2f}GB")
                
            except Exception as e:
                logger.error(f"[{request_id}] Image processing failed: {e}")
                raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")
        
        # Prepare input
        if pixel_values is not None:
            # Multimodal input
            query = f"<image>\n{request.message}"
        else:
            # Text-only input
            query = request.message
        
        # Generation config
        generation_config = dict(
            max_new_tokens=request.max_tokens,
            do_sample=False if request.temperature == 0 else True,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty
        )
        
        if request.temperature > 0:
            generation_config["temperature"] = request.temperature
        
        # Generate response using model's chat method
        generation_start = time.time()
        with torch.no_grad():
            response, history = model.chat(
                tokenizer, 
                pixel_values, 
                query, 
                generation_config,
                history=None, 
                return_history=True
            )
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Log performance metrics
        perf_logger.info(f"[{request_id}] Generation time: {generation_time:.3f}s")
        perf_logger.info(f"[{request_id}] Total processing time: {total_time:.3f}s")
        perf_logger.info(f"[{request_id}] Response length: {len(response)} characters")
        
        # Log the response content for debugging
        ocr_logger.debug(f"[{request_id}] Response content: {response}")
        
        # Log response summary
        response_lines = response.split('\n')
        logger.info(f"[{request_id}] Response summary: {len(response_lines)} lines, {len(response.split())} words")
        
        return ChatResponse(response=response)
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"[{request_id}] Chat error after {error_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/upload")
async def chat_with_upload(
    message: str = Form(...),
    file: UploadFile = File(None),
    max_tokens: int = Form(1024),
    temperature: float = Form(0.0),
    num_beams: int = Form(3),
    repetition_penalty: float = Form(2.5)
):
    """Chat endpoint with file upload"""
    start_time = time.time()
    upload_id = f"upload_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"[{upload_id}] File upload chat request received")
        logger.info(f"[{upload_id}] Message length: {len(message)} characters")
        logger.info(f"[{upload_id}] File provided: {file is not None}")
        
        if file:
            logger.info(f"[{upload_id}] File name: {file.filename}")
            logger.info(f"[{upload_id}] File type: {file.content_type}")
        
        image_base64 = None
        if file:
            # Read and encode image
            file_process_start = time.time()
            image_bytes = await file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            file_process_time = time.time() - file_process_start
            
            logger.info(f"[{upload_id}] File size: {len(image_bytes)} bytes")
            perf_logger.info(f"[{upload_id}] File processing time: {file_process_time:.3f}s")
        
        # Use the main chat endpoint
        request = ChatRequest(
            message=message,
            image_base64=image_base64,
            max_tokens=max_tokens,
            temperature=temperature,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty
        )
        
        response = await chat(request)
        
        total_time = time.time() - start_time
        perf_logger.info(f"[{upload_id}] Total upload endpoint time: {total_time:.3f}s")
        
        return response
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"[{upload_id}] Upload chat error after {error_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_ocr", response_model=BatchOCRResponse)
async def batch_ocr_legal_documents(request: BatchOCRRequest):
    """Batch OCR endpoint optimized for Vietnamese legal documents"""
    start_time = time.time()
    batch_id = f"batch_{int(time.time() * 1000)}"
    
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet, please wait")
        
        if len(request.messages) != len(request.images_base64):
            raise HTTPException(status_code=400, detail="Number of messages must match number of images")
        
        # Log batch request details
        logger.info(f"[{batch_id}] New batch OCR request received")
        logger.info(f"[{batch_id}] Number of images: {len(request.images_base64)}")
        logger.info(f"[{batch_id}] Max tokens: {request.max_tokens}")
        logger.info(f"[{batch_id}] Temperature: {request.temperature}")
        logger.info(f"[{batch_id}] Num beams: {request.num_beams}")
        logger.info(f"[{batch_id}] Repetition penalty: {request.repetition_penalty}")
        
        responses = []
        
        # Process each image with its corresponding message
        for i, (message, image_base64) in enumerate(zip(request.messages, request.images_base64)):
            page_start_time = time.time()
            page_id = f"{batch_id}_page_{i+1}"
            
            try:
                logger.info(f"[{page_id}] Processing page {i+1}/{len(request.images_base64)}")
                ocr_logger.debug(f"[{page_id}] Input message: {message}")
                
                # Process image
                image_process_start = time.time()
                image = process_image(image_base64)
                logger.info(f"[{page_id}] Image size: {image.size}")
                
                pixel_values = load_image(image, max_num=8)  # Optimized for RTX 5080 batch processing
                if device == "cuda":
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                else:
                    pixel_values = pixel_values.to(torch.float32)
                
                image_process_time = time.time() - image_process_start
                perf_logger.info(f"[{page_id}] Image processing time: {image_process_time:.3f}s")
                
                # Monitor GPU memory usage in batch processing
                if device == "cuda":
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    perf_logger.info(f"[{page_id}] GPU memory: {memory_allocated:.2f}GB")
                
                # Prepare input
                query = f"<image>\n{message}"
                
                # Optimized generation config for legal documents
                generation_config = dict(
                    max_new_tokens=request.max_tokens,
                    do_sample=False,  # Deterministic for legal accuracy
                    num_beams=request.num_beams,
                    repetition_penalty=request.repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Generate response
                generation_start = time.time()
                with torch.no_grad():
                    response, _ = model.chat(
                        tokenizer, 
                        pixel_values, 
                        query, 
                        generation_config,
                        history=None, 
                        return_history=True
                    )
                
                generation_time = time.time() - generation_start
                page_total_time = time.time() - page_start_time
                
                # Log performance metrics
                perf_logger.info(f"[{page_id}] Generation time: {generation_time:.3f}s")
                perf_logger.info(f"[{page_id}] Total page processing time: {page_total_time:.3f}s")
                perf_logger.info(f"[{page_id}] Response length: {len(response)} characters")
                
                # Log the response content for debugging
                ocr_logger.debug(f"[{page_id}] Response content: {response}")
                
                # Log response summary
                response_lines = response.split('\n')
                logger.info(f"[{page_id}] Response summary: {len(response_lines)} lines, {len(response.split())} words")
                
                responses.append(response)
                
                # Clear GPU cache if using CUDA
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                page_error_time = time.time() - page_start_time
                logger.error(f"[{page_id}] Error processing page {i + 1} after {page_error_time:.3f}s: {e}")
                error_response = f"Error processing page {i + 1}: {str(e)}"
                responses.append(error_response)
                ocr_logger.debug(f"[{page_id}] Error response: {error_response}")
        
        processing_time = time.time() - start_time
        
        # Log batch completion
        perf_logger.info(f"[{batch_id}] Total batch processing time: {processing_time:.3f}s")
        perf_logger.info(f"[{batch_id}] Average time per page: {processing_time/len(request.images_base64):.3f}s")
        logger.info(f"[{batch_id}] Batch OCR completed successfully")
        
        return BatchOCRResponse(responses=responses, processing_time=processing_time)
        
    except Exception as e:
        batch_error_time = time.time() - start_time
        logger.error(f"[{batch_id}] Batch OCR error after {batch_error_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Configure server
    host = "0.0.0.0"  # Accept connections from any IP
    port = 8001  # Changed port to avoid conflicts
    
    logger.info("=" * 70)
    logger.info(f"🚀 Starting Vintern-1B-v3.5 server on {host}:{port}")
    logger.info("🎯 RTX 5080 GPU OPTIMIZED VERSION")
    logger.info("=" * 70)
    logger.info("📊 Logging Configuration:")
    logger.info("   • General logs: INFO level")
    logger.info("   • OCR content: DEBUG level (ocr_debug logger)")
    logger.info("   • Performance: INFO level (performance logger)")
    logger.info("   • Timestamps: Enabled")
    logger.info("⚡ RTX 5080 Optimizations:")
    logger.info("   • Flash Attention: Enabled")
    logger.info("   • Higher max_num patches: 12 (vs 6)")
    logger.info("   • TF32 operations: Enabled")
    logger.info("   • cuDNN benchmark: Enabled")
    logger.info("   • GPU memory monitoring: Enabled")
    logger.info("   • Automatic device mapping: Enabled")
    logger.info("=" * 70)
    
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        raise
