# Chatter

A Python-based voice AI assistant with push-to-talk functionality that combines speech recognition, AI processing, and text-to-speech synthesis.

## Features

- 🎙️ **Push-to-talk interface** using Gradio web UI
- 🗣️ **Speech-to-text** via local OpenAI Whisper
- 🤖 **AI processing** with Ollama LLM (llama3.1:8b)
- 🔊 **Text-to-speech** using Kokoro TTS with persona-specific voices
- 🌐 **Web search integration** for current information
- 🎭 **Multiple AI personas** with distinct personalities and expertise
- 💬 **Text and voice chat** with streaming responses
- 🔔 **Chime-in functionality** for AI to join conversations naturally
- 🌐 **Web-based interface** accessible from any browser

## Architecture

```
Audio Input → OpenAI Whisper → Ollama LLM + Web Search → Kokoro TTS → Audio Output
```

1. **Recording**: Push-to-talk captures audio input
2. **Transcription**: Local OpenAI Whisper converts speech to text
3. **Processing**: Ollama LLM generates intelligent responses with web search when needed
4. **Synthesis**: Kokoro TTS converts response back to speech using persona-specific voices

## Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai/download) installed and running
- Audio input/output devices

## Installation

1. **Clone and setup the project:**
   ```bash
   cd chatter
   uv sync
   ```

2. **Run the application (models download automatically):**
   ```bash
   uv run chatter
   ```

   Or alternatively:
   ```bash
   uv run python run.py
   ```

On first run, the following models will be downloaded automatically:
- `llama3.1:8b` for AI responses (via Ollama)
- Kokoro TTS models for text-to-speech
- OpenAI Whisper tiny model for speech recognition

## Usage

1. **Start the application:**
   ```bash
   uv run chatter
   ```

2. **Open your browser** to `http://localhost:7860`

3. **Use the interface:**
   - **Voice Mode**: Click "🎙️ Start Recording" → Speak → Click "🔴 Stop & Transcribe" → Review text → Click "Send"
   - **Text Mode**: Type directly in the message box and click "Send" or press Enter
   - **Persona Selection**: Choose different AI personalities from the dropdown
   - **Chime In**: Click "🔔 Chime In" to let the AI contribute to the conversation naturally
   - Listen to the generated audio response (auto-plays)

## Project Structure

```
chatter/
├── src/chatter/
│   ├── main.py              # Main application logic
│   ├── tool_manager.py      # Web search tool management
│   ├── search_tool.py       # Web search implementation
│   └── Prompts/             # Persona prompt files (.md)
├── tests/                   # Pytest test suite
├── typings/                 # Type stubs for third-party libraries
├── run.py                   # Simple runner script
├── pyproject.toml          # Project configuration with strict typing & linting
└── README.md
```

## Dependencies

- **gradio**: Web-based UI framework
- **ollama**: Interface to Ollama models
- **kokoro**: Text-to-speech synthesis
- **sounddevice**: Audio recording
- **numpy**: Audio processing
- **openai-whisper**: Speech recognition
- **torchaudio**: Audio file handling
- **requests**: Web search functionality
- **pytest**: Testing framework
- **ruff**: Code linting and formatting
- **pyright**: Static type checking

## Configuration

The application uses default settings:
- **Sample rate**: 16kHz for speech recognition, 24kHz for TTS
- **Channels**: Mono
- **Server**: localhost:7860
- **LLM Model**: llama3.1:8b (configured in Config class)

## Development

**Running tests:**
```bash
uv run test
```

**Code linting:**
```bash
uv run ruff check
uv run ruff format  # Auto-format code
```

**Type checking:**
```bash
uv run pyright
```

The project uses strict typing with Pyright and comprehensive linting with Ruff.

## Troubleshooting

**Audio issues:**
- Ensure microphone permissions are granted
- Check audio device availability with `python -c "import sounddevice; print(sounddevice.query_devices())"`

**Ollama connection:**
- Verify Ollama is running: `ollama list`
- Ensure models are installed: `ollama pull llama3.1:8b`

**Web search issues:**
- Web search uses DuckDuckGo and requires internet connectivity
- Search failures will be logged but won't crash the application

**Performance:**
- First run may be slower due to model loading
- Kokoro TTS initialization takes a few seconds on startup
- Whisper model will be downloaded automatically on first use

## Acknowledgments

Special thanks to [Aravindh Sridharan](https://www.linkedin.com/in/aravindhsridharan/) for sharing the persona prompts that inspired this project.

## License

MIT License