# Chatter

A Python-based voice AI assistant with push-to-talk functionality that combines speech recognition, AI processing, and text-to-speech synthesis.

## Features

- 🎙️ **Push-to-talk interface** using Gradio web UI
- 🗣️ **Speech-to-text** via local OpenAI Whisper
- 🤖 **AI processing** with Ollama DeepSeek-R1
- 🔊 **Text-to-speech** using Kokoro TTS
- 🌐 **Web-based interface** accessible from any browser

## Architecture

```
Audio Input → OpenAI Whisper → DeepSeek-R1 → Kokoro TTS → Audio Output
```

1. **Recording**: Push-to-talk captures audio input
2. **Transcription**: Local OpenAI Whisper converts speech to text
3. **Processing**: Ollama DeepSeek-R1 generates intelligent responses
4. **Synthesis**: Kokoro TTS converts response back to speech

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
- `deepseek-r1:7b` for AI responses (via Ollama)
- Kokoro TTS models for text-to-speech
- OpenAI Whisper tiny model for speech recognition

## Usage

1. **Start the application:**
   ```bash
   uv run chatter
   ```

2. **Open your browser** to `http://localhost:7860`

3. **Use the interface:**
   - Click "🎤 Start Recording" to begin recording
   - Speak your message
   - Click "⏹️ Stop & Process" to process and get AI response
   - Listen to the generated audio response

## Project Structure

```
chatter/
├── src/chatter/
│   ├── main.py              # Main application logic
│   └── Prompts/             # Persona prompt files
├── run.py                   # Simple runner script
├── pyproject.toml          # Project configuration
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

## Configuration

The application uses default settings:
- **Sample rate**: 16kHz
- **Channels**: Mono
- **Server**: localhost:7860

## Troubleshooting

**Audio issues:**
- Ensure microphone permissions are granted
- Check audio device availability with `python -c "import sounddevice; print(sounddevice.query_devices())"`

**Ollama connection:**
- Verify Ollama is running: `ollama list`
- Ensure models are installed: `ollama pull deepseek-r1:32b`

**Performance:**
- First run may be slower due to model loading
- Kokoro TTS initialization takes a few seconds on startup
- Whisper model will be downloaded automatically on first use

## Acknowledgments

Special thanks to [Aravindh Sridharan](https://www.linkedin.com/in/aravindhsridharan/) for sharing the persona prompts that inspired this project.

## License

MIT License