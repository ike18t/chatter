#!/usr/bin/env python3
"""
Standalone test for Llama model with memoization
"""

import functools
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
    load_dotenv(".env.local", override=True)  # Override with .env.local if it exists
except ImportError:
    pass

@functools.lru_cache(maxsize=2)
def load_model_and_tokenizer(model_name: str):
    """Load and cache HuggingFace model and tokenizer."""
    print(f"📥 Loading {model_name} from HuggingFace...")

    # Get HuggingFace token if available
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token and hf_token.strip():
        print("🔑 Using HuggingFace authentication token")
        auth_kwargs = {"token": hf_token}
    else:
        print("⚠️  No HuggingFace token found - only public models will work")
        auth_kwargs = {}

    try:
        # Try to load the tokenizer first (smaller download)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_kwargs)
        print(f"✅ Tokenizer for {model_name} loaded successfully")

        # Load the model with Mac Silicon optimizations
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            trust_remote_code=True,
            **auth_kwargs
        )
        model = model.to(device)
        print(f"✅ Model {model_name} loaded successfully on {device}")

        return {"model": model, "tokenizer": tokenizer}

    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        if "gated repo" in str(e).lower():
            print("💡 This model requires authentication. Please:")
            print("   1. Get access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
            print("   2. Create a token at: https://huggingface.co/settings/tokens")
            print("   3. Add HUGGINGFACE_HUB_TOKEN=your_token to .env.local")
        raise

def test_llama():
    """Test Llama model loading with authentication."""
    print("🦙 Testing Llama Model Loading with Memoization")
    print("=" * 60)

    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    try:
        # Test the memoized function
        print("1️⃣ First load (should show loading messages):")
        cache = load_model_and_tokenizer(model_name)
        model = cache["model"]
        tokenizer = cache["tokenizer"]

        print("\n🧪 Testing text generation...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Please respond in exactly 5 words."}
        ]

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
                max_new_tokens=15,
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
        print("\n2️⃣ Second load (should use cache, no loading messages):")
        cache2 = load_model_and_tokenizer(model_name)
        print("✅ Memoization working! (no loading messages above)")

        # Show cache info
        print(f"\n📊 Cache info: {load_model_and_tokenizer.cache_info()}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llama()