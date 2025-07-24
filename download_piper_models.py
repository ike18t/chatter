#!/usr/bin/env python3
"""
Download Piper TTS voice models for the Voice AI Assistant.
"""

import urllib.request
import json
from pathlib import Path
import sys

def download_model(model_name: str, models_dir: Path) -> bool:
    """Download a Piper voice model and its config."""
    # Models are now hosted on Hugging Face
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    
    # Map model names to their Hugging Face paths
    model_paths = {
        "en_US-amy-low": "en/en_US/amy/low",
        "en_US-lessac-high": "en/en_US/lessac/high", 
        "en_GB-alan-low": "en/en_GB/alan/low"
    }
    
    hf_path = model_paths.get(model_name)
    if not hf_path:
        print(f"❌ Unknown model path for {model_name}")
        return False
    
    model_url = f"{base_url}/{hf_path}/{model_name}.onnx"
    config_url = f"{base_url}/{hf_path}/{model_name}.onnx.json"
    
    model_path = models_dir / f"{model_name}.onnx"
    config_path = models_dir / f"{model_name}.onnx.json"
    
    try:
        print(f"Downloading {model_name}...")
        
        # Download model file
        print(f"  Downloading {model_name}.onnx...")
        urllib.request.urlretrieve(model_url, model_path)
        
        # Download config file
        print(f"  Downloading {model_name}.onnx.json...")
        urllib.request.urlretrieve(config_url, config_path)
        
        print(f"✅ Successfully downloaded {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        # Clean up partial downloads
        for path in [model_path, config_path]:
            if path.exists():
                path.unlink()
        return False

def main():
    """Download recommended Piper voice models."""
    # Set up models directory
    models_dir = Path.home() / ".local" / "share" / "piper-voices"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Piper models directory: {models_dir}")
    
    # Recommended models for different personas
    recommended_models = [
        "en_US-amy-low",        # Default female voice
        "en_US-lessac-high",    # High-quality male voice
        "en_GB-alan-low",       # British male voice
    ]
    
    print(f"\nDownloading {len(recommended_models)} voice models...")
    
    success_count = 0
    for model_name in recommended_models:
        if download_model(model_name, models_dir):
            success_count += 1
    
    print(f"\n📊 Downloaded {success_count}/{len(recommended_models)} models successfully")
    
    if success_count > 0:
        print("✅ Voice models are ready! You can now use Piper TTS.")
    else:
        print("❌ No models downloaded. Check your internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()