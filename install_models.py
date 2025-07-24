#!/usr/bin/env python3
"""Setup script to install Ollama models required for the voice assistant."""

import subprocess
import sys

def run_command(command):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Install required Ollama models."""
    print("Setting up Voice AI Assistant...")
    print("This will install the required Ollama models:")
    print("- deepseek-r1:7b (for AI responses - faster than 32b)")
    print()
    print("Note: ChatTTS and Whisper models will be downloaded automatically on first use")

    # Check if Ollama is installed
    if not run_command("which ollama"):
        print("\n❌ Ollama is not installed. Please install it first:")
        print("Visit: https://ollama.ai/download")
        sys.exit(1)

    # Pull required models
    models = ["deepseek-r1:7b"]

    for model in models:
        print(f"\nInstalling {model}...")
        if not run_command(f"ollama pull {model}"):
            print(f"Failed to install {model}. Please try manually: ollama pull {model}")
            sys.exit(1)

    print("\n🎉 Setup complete! You can now run the voice assistant:")
    print("uv run voice-assistant")

if __name__ == "__main__":
    main()
