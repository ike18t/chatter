# Setup Instructions

## Environment Configuration

This project uses a committed `.env` file with default values. You can override these locally without affecting the repository.

### For Gated Models (like Llama 3.1)

1. **Request Access**: Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and request access
2. **Get Token**: Create a token at https://huggingface.co/settings/tokens
3. **Add Token Locally**: Create `.env.local` with your token:

```bash
# .env.local (this file is gitignored)
HUGGINGFACE_HUB_TOKEN=hf_your_actual_token_here
```

### Configuration Options

You can override any setting in `.env.local`:

```bash
# Use a different model
CHATTER_MODEL=microsoft/DialoGPT-medium

# Change server settings
CHATTER_HOST=127.0.0.1
CHATTER_PORT=8080

# Use different Whisper model
CHATTER_WHISPER_MODEL=base
```

## Installation

```bash
# Install dependencies
pip install -e .

# Test the setup
python test_llm_only.py
```

The application will automatically:
- Load settings from `.env` (committed defaults)
- Override with `.env.local` if it exists (your local settings)
- Provide helpful error messages for authentication issues