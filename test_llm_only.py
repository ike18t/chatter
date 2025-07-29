#!/usr/bin/env python3
"""
Test script for core LLM functionality only
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_llm_core():
    """Test the core LLM functionality."""
    print("🧪 Testing Core LLM Functionality")
    print("=" * 50)

    try:
        # Use a small model for testing
        model_name = "microsoft/DialoGPT-small"

        print(f"1. Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use Mac Silicon optimizations
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"2. Using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            trust_remote_code=True
        )
        model = model.to(device)
        print("✅ Model loaded successfully")

        # Test generation
        print("3. Testing text generation...")
        messages = [{"role": "user", "content": "Hello! How are you?"}]

        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        print(f"Response: {response.strip()}")
        print("✅ Text generation working")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_llm_core()