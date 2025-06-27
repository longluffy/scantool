#!/usr/bin/env python3
"""
Setup script for Vintern-1B-v2 local hosting
This script will help you set up a local API server for Vintern-1B-v2 (Vietnamese multimodal model)
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch>=2.0.0",
        "torchvision", 
        "transformers>=4.37.2",
        "accelerate",
        "sentencepiece",
        "pillow",
        "requests",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "numpy"
    ]
    
    print("Installing required packages...")
    for req in requirements:
        print(f"Installing {req}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])
    
    print("All requirements installed successfully!")

def check_gpu():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available! GPUs detected: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  CUDA not available, will use CPU (slower but works)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

if __name__ == "__main__":
    print("=== Vintern-1B-v2 Setup ===")
    print("Setting up Vietnamese multimodal model hosting environment...")
    
    # Install requirements
    install_requirements()
    
    # Check GPU
    check_gpu()
    
    print("\n✅ Setup complete!")
    print("Next steps:")
    print("1. Run the API server: python working_vintern_server.py")
    print("2. Test with client: python working_vintern_client.py")
    print("3. Check health: curl http://localhost:8000/health")
