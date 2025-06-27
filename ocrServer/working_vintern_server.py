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
from transformers import AutoModel, AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def load_image(image, input_size=448, max_num=12):
    """Load and process image for the model"""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model():
    """Load Vintern-1B-v3.5 model using the working configuration"""
    global model, tokenizer, device
    
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        model_path = "5CD-AI/Vintern-1B-v3_5"
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            use_fast=False
        )
        
        logger.info("Loading model...")
        if device == "cuda":
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False,  # Disable flash attention for compatibility
            ).eval().cuda()
        else:
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False,
                device_map=None
            ).eval()
            
        logger.info("✅ Model loaded successfully!")
        
        # Quick test
        test_question = "Xin chào!"
        generation_config = dict(max_new_tokens=50, do_sample=False, num_beams=1, repetition_penalty=1.0)
        response, _ = model.chat(tokenizer, None, test_question, generation_config, history=None, return_history=True)
        logger.info(f"✅ Model test successful: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
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
    logger.info("Starting Vintern-1B-v3.5 API Server...")
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup!")
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
    """Health check endpoint"""
    model_loaded = model is not None and tokenizer is not None
    return {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "model": "5CD-AI/Vintern-1B-v3_5",
        "device": str(device),
        "parameters": "~938M"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for text and multimodal conversations"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet, please wait")
        
        # Process image if provided
        pixel_values = None
        if request.image_base64:
            try:
                image = process_image(request.image_base64)
                pixel_values = load_image(image, max_num=6)  # Reduced for faster processing
                if device == "cuda":
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                else:
                    pixel_values = pixel_values.to(torch.float32)
            except Exception as e:
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
        with torch.no_grad():
            response, history = model.chat(
                tokenizer, 
                pixel_values, 
                query, 
                generation_config,
                history=None, 
                return_history=True
            )
        
        return ChatResponse(response=response)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
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
    try:
        image_base64 = None
        if file:
            # Read and encode image
            image_bytes = await file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Use the main chat endpoint
        request = ChatRequest(
            message=message,
            image_base64=image_base64,
            max_tokens=max_tokens,
            temperature=temperature,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty
        )
        
        return await chat(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Configure server
    host = "0.0.0.0"  # Accept connections from any IP
    port = 8000
    
    logger.info(f"Starting Vintern-1B-v3.5 server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
