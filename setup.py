"""
Quick setup script for Exam Intelligence Engine
Run this after cloning to set up the environment
"""

import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Set up Python environment and install dependencies"""
    print("🚀 Setting up Exam Intelligence Engine...")
    
    # Create virtual environment
    print("📦 Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Install dependencies
    print("⬇️ Installing dependencies...")
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
    
    # Create necessary directories
    print("📁 Creating directories...")
    dirs_to_create = [
        "data/raw",
        "data/processed/embeddings", 
        "outputs",
        "test_data"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Setup complete!")
    print("\n📚 Next steps:")
    print("1. Place PDF files in data/raw/")
    print("2. Run: python main.py --input_dir data/raw")
    print("3. Check WEEK1_PROMPTS.md for implementation details")

if __name__ == "__main__":
    setup_environment()