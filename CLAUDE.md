# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chatter is a Python-based voice AI assistant with push-to-talk functionality that combines speech recognition, AI processing, and text-to-speech synthesis. It provides a web-based interface for voice and text chat with AI personas using local models.

## Common Development Commands

### Environment Setup

```bash
# Install dependencies
uv sync

# Override environment variables (for gated models)
# Create .env.local and add:
HUGGINGFACE_HUB_TOKEN=your_token_here
```

### Running the Application

```bash
# Start the application
uv run chatter
# OR
uv run python run.py

# Access the UI
# Open browser to http://localhost:7860
```

### Testing

```bash
# Run all tests
uv run test

# Run specific test categories
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m slow

# Run a specific test file
uv run pytest tests/test_basic_functionality.py

# Run a specific test
uv run pytest tests/test_basic_functionality.py::TestConversationManager::test_clear_conversation
```

### Code Quality

```bash
# Check code style
uv run ruff check

# Auto-format code
uv run ruff format

# Run static type checking
uv run pyright
```

## Architecture Overview

Chatter combines several AI components in a processing pipeline:

1. **Audio Recording**: `AudioRecorder` class manages push-to-talk audio capture
2. **Speech Recognition**: `TranscriptionService` uses OpenAI Whisper for STT
3. **LLM Processing**: `LLMService` manages Ollama/HuggingFace model interactions
4. **Web Search**: `ToolManager` and `SearchTool` provide web search capabilities
5. **Text-to-Speech**: `TTSService` uses Kokoro TTS for voice synthesis
6. **Conversation Management**: `ConversationManager` handles chat history
7. **UI**: `VoiceChatInterface` provides Gradio-based web interface

The architecture follows these patterns:
- Service-oriented design with clear separation of concerns
- Dependency injection for testability
- Dataclasses for configuration
- Strong typing throughout the codebase
- Error handling with detailed debugging outputs
- Class-based components with single responsibility

## Key Files

- `src/chatter/main.py`: Core implementation of all services
- `src/chatter/tool_manager.py`: Manages web search integration
- `src/chatter/search_tool.py`: Implements web search functionality
- `src/chatter/Prompts/`: Contains AI persona definitions

## Important Notes

1. The application requires Python 3.13+ and uses modern Python features
2. Strict type checking is enforced with pyright
3. LLM models are dynamically loaded from HuggingFace or Ollama
4. Audio processing requires microphone permissions
5. For HuggingFace gated models (like Llama 3.1), you need to request access and set up an API token
6. Type stubs for third-party libraries are in the `typings/` directory

## Common Development Workflows

1. **Adding a new persona**:
   - Create a new markdown file in `src/chatter/Prompts/`
   - Update voice settings in `PersonaManager.setup_voice_settings()`

2. **Adding a new tool**:
   - Implement the tool function in an appropriate module
   - Register the tool in `ToolManager._load_tools()`
   - Add tool definitions in `ToolManager.get_tool_definitions()`

3. **Modifying the UI**:
   - Update the Gradio interface in `VoiceChatInterface.create_interface()`

4. **Fixing type issues**:
   - The project uses strict type checking with pyright
   - Type definitions for external libraries are in `typings/`
   - Common type definitions are in `src/chatter/types.py`
   - Run `uv run pyright` to check for type issues