#!/usr/bin/env python3
"""
Test script for Llama model with authentication
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
    load_dotenv(".env.local", override=True)  # Override with .env.local if it exists
except ImportError:
    pass

from chatter.main import load_model_and_tokenizer

def test_llama():
    """Test Llama model loading with authentication."""
    print("🦙 Testing Llama Model Loading")
    print("=" * 50)

    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    try:
        # Test the memoized function
        cache = load_model_and_tokenizer(model_name)
        model = cache["model"]
        tokenizer = cache["tokenizer"]

        print("\n🧪 Testing text generation...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Please respond in exactly 10 words."}
        ]

        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"Input: {input_text[:100]}...")

        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate response
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        print(f"Response: '{response.strip()}'")
        print("✅ Llama model working!")

        # Test memoization - should not reload
        print("\n🔄 Testing memoization...")
        cache2 = load_model_and_tokenizer(model_name)
        print("✅ Memoization working (no reload message)")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llama()