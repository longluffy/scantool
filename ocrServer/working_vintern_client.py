#!/usr/bin/env python3
"""
Working client for Vintern-1B-v3.5 API
"""

import requests
import base64
from pathlib import Path
from typing import Optional
import time

class WorkingVinternClient:
    """Client for the working Vintern-1B-v3.5 API server"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = requests.Session()
    
    def is_healthy(self) -> dict:
        """Check server health with detailed info"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def wait_for_model(self, max_wait: int = 60) -> bool:
        """Wait for model to be loaded"""
        print("⏳ Waiting for model to load...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            health = self.is_healthy()
            if health.get("model_loaded", False):
                print("✅ Model loaded successfully!")
                return True
            elif health.get("status") == "error":
                print(f"❌ Server error: {health.get('error')}")
                return False
            
            print(".", end="", flush=True)
            time.sleep(2)
        
        print(f"\n⏰ Timeout after {max_wait} seconds")
        return False
    
    def chat(self, message: str, image_path: Optional[str] = None, 
             max_tokens: int = 1024, temperature: float = 0.0,
             num_beams: int = 3, repetition_penalty: float = 2.5) -> str:
        """
        Chat with Vintern-1B-v3.5
        
        Args:
            message: Text message (Vietnamese or English)
            image_path: Optional path to image file
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            num_beams: Number of beams for beam search
            repetition_penalty: Repetition penalty
            
        Returns:
            Model response as string
        """
        # Prepare image if provided
        image_base64 = None
        if image_path:
            image_path = Path(image_path)
            if image_path.exists():
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            else:
                raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Prepare request
        payload = {
            "message": message,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty
        }
        
        if image_base64:
            payload["image_base64"] = image_base64
        
        # Send request
        try:
            response = self.session.post(
                f"{self.server_url}/chat",
                json=payload,
                timeout=120  # Allow time for model processing
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection error: {e}")
    
    def vietnamese_ocr(self, image_path: str) -> str:
        """Convenience method for Vietnamese OCR"""
        return self.chat("Trích xuất tất cả văn bản trong ảnh này.", image_path=image_path)
    
    def document_analysis(self, image_path: str, question: str) -> str:
        """Convenience method for document Q&A"""
        return self.chat(question, image_path=image_path)
    
    def describe_image(self, image_path: str) -> str:
        """Convenience method for image description"""
        return self.chat("Mô tả hình ảnh một cách chi tiết.", image_path=image_path)

def demo():
    """Demo the working client"""
    print("🇻🇳 Vintern-1B-v3.5 Client Demo")
    print("=" * 40)
    
    # Create client
    client = WorkingVinternClient()
    
    # Check server health
    print("🔍 Checking server status...")
    health = client.is_healthy()
    print(f"Status: {health}")
    
    if not health.get("model_loaded", False):
        print("🚀 Server found but model not loaded yet.")
        if not client.wait_for_model():
            print("❌ Model failed to load. Please start the server:")
            print("   python working_vintern_server.py")
            return
    
    print("✅ Server is ready!")
    
    # Test 1: Vietnamese conversation
    print("\n💬 Test 1: Vietnamese conversation")
    try:
        response = client.chat("Xin chào! Bạn có thể giúp tôi phân tích tài liệu không?")
        print(f"🤖 Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: English conversation
    print("\n💬 Test 2: English conversation")
    try:
        response = client.chat("Hello! Can you help me analyze documents in Vietnamese?")
        print(f"🤖 Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Image analysis (if you have an image)
    print("\n🖼️ Test 3: Image analysis")
    print("To test image analysis, place an image file and uncomment the code below:")
    print("""
    try:
        response = client.describe_image("your_image.jpg")
        print(f"🤖 Image description: {response}")
        
        response = client.vietnamese_ocr("document.jpg") 
        print(f"🤖 OCR result: {response}")
        
        response = client.document_analysis("receipt.jpg", "Tổng số tiền là bao nhiêu?")
        print(f"🤖 Document analysis: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    """)
    
    print("\n🎉 Demo completed!")
    print("\n📝 Your Vintern-1B-v3.5 API is ready for:")
    print("- Vietnamese OCR and text extraction")
    print("- Document analysis and Q&A")
    print("- Receipt and invoice processing")
    print("- Chart and diagram interpretation")
    print("- General image description")

if __name__ == "__main__":
    demo()
