#!/usr/bin/env python3
"""
Test script for HuggingFace integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatter.main import LLMService, ModelManager

def test_huggingface_integration():
    """Test the new HuggingFace-based LLM service."""
    print("🧪 Testing HuggingFace Integration")
    print("=" * 50)

    try:
        # Test ModelManager
        print("1. Testing ModelManager...")
        model_name = "microsoft/DialoGPT-small"  # Small model for testing
        ModelManager.ensure_deepseek_model(model_name)
        print("✅ ModelManager working")

        # Test LLMService
        print("\n2. Testing LLMService...")
        llm = LLMService(model_name)

        # Test simple response
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"}
        ]

        response, status = llm.get_response(messages)
        print(f"Response: {response}")
        print(f"Status: {status}")

        if response:
            print("✅ LLMService working")
        else:
            print("❌ LLMService failed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_huggingface_integration()