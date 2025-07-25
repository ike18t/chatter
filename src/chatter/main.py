# Standard library imports
import re
import subprocess
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NotRequired, Optional, TypedDict, cast

# Third-party imports
import gradio as gr
import numpy as np
import ollama
import sounddevice as sd
import whisper
from numpy.typing import NDArray

# Local imports
from .tool_manager import ToolManager


# Type definitions
class SerializedToolCall(TypedDict):
    id: str
    type: str
    function: dict[str, str]


class MessageDict(TypedDict):
    role: str
    content: str
    tool_call_id: NotRequired[str | None]  # For tool response messages
    tool_calls: NotRequired[
        list[SerializedToolCall]
    ]  # For assistant messages with tool calls


class HistoryEntry(TypedDict):
    user: str | None
    assistant: str | None


# Gradio interface types
GradioHistory = list[list[str]]
AudioTuple = tuple[int, NDArray[np.float32]] | None
ProcessingYield = tuple[str, bool, GradioHistory, AudioTuple]
GradioUpdate = dict[str, str | bool]


def _check_tts_availability() -> bool:
    """Check if TTS dependencies are available."""
    try:
        import kokoro

        # Test that we can actually use it
        _ = kokoro.__name__  # Use the import
        return True
    except ImportError:
        print(
            "Warning: kokoro-tts not available. Install with: pip install kokoro soundfile"
        )
        return False


# Constants
TTS_AVAILABLE = _check_tts_availability()


# Audio processing constants
MIN_AUDIO_DURATION = 0.1  # seconds
MIN_AUDIO_RMS_THRESHOLD = 0.001
AUDIO_TEST_DURATION = 0.5  # seconds
PERMISSION_TEST_SLEEP = 0.5  # seconds


@dataclass(frozen=True)
class Config:
    """Application configuration (immutable).

    This configuration class is frozen to prevent accidental modification
    after initialization, ensuring consistent behavior throughout the application.
    """

    DEEPSEEK_MODEL: str = "llama3.1:8b"  # Known tool-capable model
    WHISPER_MODEL: str = "tiny"
    SAMPLE_RATE: int = 16000  # Standard rate for speech recognition
    TTS_SAMPLE_RATE: int = 24000  # Kokoro TTS optimal sample rate
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 7860
    SYSTEM_PROMPT: str = """
        You are a friendly (but humorous) chat bot. Your output will be spoken so try to make it sound like natural language.

        When you search the web, trust and use the search results since they contain current information that may be more accurate than your training data.
    """


class RecordingState(Enum):
    """Recording states."""

    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class PersonaManager:
    """Manages persona prompts and voice settings from the Prompts directory."""

    def __init__(self, prompts_dir: str | None = None):
        if prompts_dir is None:
            # Use the Prompts directory relative to this file
            current_dir = Path(__file__).parent
            default_prompts_dir = current_dir / "Prompts"
            self.prompts_dir = default_prompts_dir
        else:
            self.prompts_dir = Path(prompts_dir)
        self.personas: dict[str, str] = {}
        self.voice_settings: dict[str, dict[str, str | float]] = {}
        self.load_personas()
        self.setup_voice_settings()

    def load_personas(self) -> None:
        """Load all persona files from the Prompts directory."""
        self.personas = {"Default": Config.SYSTEM_PROMPT}  # Add default first

        if not self.prompts_dir.exists():
            print(f"Warning: Prompts directory not found at {self.prompts_dir}")
            return

        # Load all .md files from the Prompts directory
        for persona_file in sorted(self.prompts_dir.glob("*.md")):
            try:
                with persona_file.open(encoding="utf-8") as f:
                    content = f.read().strip()

                # Extract persona name from filename (remove number prefix and .md extension)
                persona_name = persona_file.stem
                # Remove number prefixes like "0. ", "1. ", etc.
                if ". " in persona_name and persona_name.split(". ")[0].isdigit():
                    persona_name = persona_name.split(". ", 1)[1]

                # Use persona content as-is
                persona_content = content

                self.personas[persona_name] = persona_content
                print(f"Loaded persona: {persona_name}")

            except Exception as e:
                print(f"Error loading persona from {persona_file}: {e}")

    def get_persona_names(self) -> list[str]:
        """Get list of available persona names."""
        return list(self.personas.keys())

    def get_persona_prompt(self, persona_name: str) -> str:
        """Get the system prompt for a specific persona."""
        base_prompt = self.personas.get(persona_name, Config.SYSTEM_PROMPT)

        # Add web search instruction to all personas (unless it's already there)
        web_search_instruction = "\n\nWhen you search the web, trust and use the search results since they contain current information that may be more accurate than your training data."

        if web_search_instruction.strip() not in base_prompt:
            return base_prompt + web_search_instruction
        return base_prompt

    def get_default_persona(self) -> str:
        """Get the default persona name."""
        return "Default"

    def setup_voice_settings(self) -> None:
        """Setup unique voice models for each persona using Kokoro TTS."""
        # Kokoro TTS voices: af_ = American female, am_ = American male, bf_ = British female, bm_ = British male
        self.voice_settings = {
            "Default": {"voice": "af_sarah", "speed": 1.0},
            "Product Manager Prompt": {"voice": "am_michael", "speed": 0.9},
            "Software Architect": {"voice": "bm_george", "speed": 0.8},
            "Developer": {"voice": "af_nova", "speed": 1.1},
            "Code Explainer": {"voice": "bm_lewis", "speed": 0.8},
            "Code Reviewer": {"voice": "am_adam", "speed": 0.9},
            "Devops Engineer": {"voice": "af_jessica", "speed": 1.0},
            "Security Engineer": {"voice": "bm_george", "speed": 0.8},
            "Performance Engineer": {"voice": "am_michael", "speed": 0.9},
            "SRE": {"voice": "am_adam", "speed": 0.9},
            "QA Engineer": {"voice": "af_bella", "speed": 1.0},
            "Rogue Engineer": {"voice": "af_heart", "speed": 1.2},
            "Tech Documenter": {"voice": "bf_emma", "speed": 0.8},
            "Changelog Reviewer": {"voice": "bm_lewis", "speed": 0.8},
            "Test Engineer": {"voice": "af_nicole", "speed": 1.0},
        }

    def get_voice_settings(self, persona_name: str) -> dict[str, str | float]:
        """Get voice settings for a specific persona."""
        return self.voice_settings.get(persona_name, self.voice_settings["Default"])


class ModelManager:
    """Manages model downloading and availability."""

    @staticmethod
    def ensure_deepseek_model(model_name: str) -> None:
        """Ensure DeepSeek model is available, download if needed."""
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, check=True
            )
            if model_name in result.stdout:
                print(f"✅ {model_name} is already available")
                return

            print(f"📥 Downloading {model_name}... (this may take a few minutes)")
            subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"✅ {model_name} downloaded successfully")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error with Ollama: {e}")
            print("Please make sure Ollama is installed and running")
            raise
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first:")
            print("Visit: https://ollama.ai/download")
            raise

    @staticmethod
    def ensure_kokoro_model() -> None:
        """Ensure Kokoro TTS model is available."""
        # Kokoro model is downloaded automatically on first use
        # No manual downloading required like Piper
        print("✅ Kokoro TTS will download model automatically on first use")


class AudioRecorder:
    """Handles audio recording functionality."""

    def __init__(self, sample_rate: int = Config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data: list[NDArray[np.float32]] = []
        self.record_thread: threading.Thread | None = None

        # Check audio devices on initialization
        try:
            import sounddevice as sd
            from sounddevice import DeviceInfo, DeviceList

            devices = cast(DeviceList, sd.query_devices())
            print("Available audio devices:")
            for i, device_info in enumerate(devices):
                if device_info["max_input_channels"] > 0:
                    print(
                        f"  Input device {i}: {device_info['name']} - Channels: {device_info['max_input_channels']}, Sample Rate: {device_info['default_samplerate']}"
                    )

            default_device = cast(DeviceInfo, sd.query_devices(kind="input"))
            print(f"\n✅ Using system default input device: {default_device['name']}")

            # Test microphone permissions
            print("\n🔍 Testing microphone permissions...")
            self._test_microphone_permissions()

        except Exception as e:
            print(f"Error querying audio devices: {e}")

    def _test_microphone_permissions(self) -> None:
        """Test if microphone permissions are properly granted."""
        try:
            test_audio: list[NDArray[np.float32]] = []

            def permission_test_callback(
                indata: NDArray[np.float32],
                _frames: int,
                _time: float,
                status: Optional["sd.CallbackFlags"],
            ) -> None:
                if status:
                    print(f"⚠️  Audio stream status: {status}")
                test_audio.append(indata.copy())

            # Record for test duration to test permissions using system default
            print("Testing with system default device")

            with sd.InputStream(
                callback=permission_test_callback,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=1024,
            ):
                time.sleep(AUDIO_TEST_DURATION)

            if test_audio:
                audio_data = np.concatenate(test_audio)
                print(f"Test audio shape: {audio_data.shape}")
                print(f"Test audio dtype: {audio_data.dtype}")
                print(
                    f"Test audio range: {audio_data.min():.6f} to {audio_data.max():.6f}"
                )
                print(f"Test audio std: {audio_data.std():.6f}")

                if np.all(audio_data == 0):
                    print("⚠️  Audio data is all zeros - investigating cause...")
                    # Check if it's actually capturing but with wrong format
                    print(f"Captured {len(test_audio)} chunks")
                    for i, chunk in enumerate(test_audio[:3]):  # Check first 3 chunks
                        print(
                            f"  Chunk {i}: shape={chunk.shape}, dtype={chunk.dtype}, range={chunk.min():.6f} to {chunk.max():.6f}"
                        )
                else:
                    max_val = np.max(np.abs(audio_data))
                    print(f"✅ Microphone working! Max audio level: {max_val:.6f}")
                    if max_val < MIN_AUDIO_RMS_THRESHOLD:
                        print(
                            "⚠️  Audio level very low - check microphone volume or speak louder"
                        )
            else:
                print("⚠️  No audio data captured during permission test")

        except Exception as e:
            print(f"⚠️  Permission test failed: {e}")
            print("   This might indicate a permission or hardware issue")

    def start_recording(self) -> str:
        """Start push-to-talk recording."""
        if self.recording:
            return "Already recording..."

        try:
            print("=== Starting Recording Debug ===")

            # Check if we can query devices
            try:
                from sounddevice import DeviceInfo

                default_device = cast(DeviceInfo, sd.query_devices(kind="input"))
                print(f"Default input device found: {default_device['name']}")
                print(f"Device info: {default_device}")
                default_sr = int(default_device["default_samplerate"])

                if default_sr != self.sample_rate:
                    print(
                        f"Adjusting sample rate: {self.sample_rate} Hz -> {default_sr} Hz"
                    )
                    self.sample_rate = default_sr
                else:
                    print(f"Using sample rate: {self.sample_rate} Hz")

                # Also check what sounddevice reports as defaults
                print("sounddevice default settings:")
                print(f"  default.device: {sd.default.device}")
                print(f"  default.samplerate: {sd.default.samplerate}")
                print(f"  default.dtype: {sd.default.dtype}")

            except Exception as device_error:
                print(f"Device query failed: {device_error}")
                return f"❌ Cannot access audio devices: {device_error}"

            # Test if we can create a stream briefly
            try:
                print("Testing audio stream creation...")
                test_data: list[NDArray[np.float32]] = []

                def test_callback(
                    indata: NDArray[np.float32],
                    _frames: int,
                    _time: float,
                    _status: Optional["sd.CallbackFlags"],
                ) -> None:
                    test_data.append(indata.copy())

                # Test with system default device
                print("Testing stream with system default device")
                with sd.InputStream(
                    callback=test_callback,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32,
                    blocksize=1024,
                ):
                    time.sleep(0.1)  # Brief test
                print(
                    f"Audio stream test successful. Captured {len(test_data)} chunks."
                )

                # Check if the test data contains actual audio
                if test_data:
                    test_array = np.concatenate(test_data)
                    print(
                        f"Stream test audio: min={test_array.min():.6f}, max={test_array.max():.6f}"
                    )
                    if np.all(test_array == 0):
                        print(
                            "⚠️  Stream test also returned zeros - device/driver issue?"
                        )

            except Exception as stream_error:
                print(f"Stream creation failed: {stream_error}")
                return f"❌ Audio stream error: {stream_error}"

            # If we get here, audio should work
            self.recording = True
            self.audio_data = []
            self.record_thread = threading.Thread(target=self._record_audio)
            self.record_thread.daemon = True
            self.record_thread.start()
            print("Recording thread started successfully")

            # Give the thread a moment to start
            time.sleep(0.1)
            print(f"Thread is alive: {self.record_thread.is_alive()}")
            print(f"Initial recording state: {self.recording}")
            return "🔴 Recording... (Release button to process)"

        except Exception as e:
            self.recording = False
            print(f"Unexpected error starting recording: {e}")
            print(f"Error type: {type(e).__name__}")
            return f"❌ Recording failed: {str(e)}"

    def stop_recording(self) -> NDArray[np.float32] | None:
        """Stop recording and return audio data."""
        if not self.recording:
            print("⚠️ Stop called but not recording")
            return None

        print("🛑 Stopping recording...")
        self.recording = False
        time.sleep(0.2)  # Wait for recording thread to finish

        if not self.audio_data:
            print("❌ No audio data captured")
            return None

        print(f"📊 Concatenating {len(self.audio_data)} audio chunks...")
        try:
            # Concatenate all audio chunks
            audio_array = np.concatenate(self.audio_data).astype(np.float32)
            print(
                f"✅ Final audio array: shape={audio_array.shape}, dtype={audio_array.dtype}"
            )
            print(
                f"Audio stats: min={audio_array.min():.4f}, max={audio_array.max():.4f}, mean={audio_array.mean():.4f}"
            )

            # Basic validation
            if len(audio_array) == 0:
                print("❌ Empty audio array after concatenation")
                return None

            if np.all(audio_array == 0):
                print(
                    "❌ Audio array contains only zeros - MICROPHONE PERMISSION ISSUE!"
                )
                print("📋 To fix this:")
                print(
                    "1. Open System Preferences/Settings > Security & Privacy > Privacy > Microphone"
                )
                print("2. Find 'Python' or 'Terminal' in the list")
                print("3. Check the box to allow microphone access")
                print("4. Restart this application")
                print(
                    "5. You may need to grant permission to the specific Python executable"
                )
                return None

            return audio_array

        except Exception as concat_error:
            print(f"❌ Error concatenating audio data: {concat_error}")
            return None

    def _record_audio(self) -> None:
        """Internal method to record audio continuously."""
        print("🎤 _record_audio thread started!")

        def audio_callback(
            indata: NDArray[np.float32],
            _frames: int,
            _time: float,
            status: Optional["sd.CallbackFlags"],
        ) -> None:
            if status:
                print(f"Audio callback status: {status}")
            if self.recording:
                try:
                    self.audio_data.append(indata.copy().flatten())
                    # Only print occasionally to avoid spam
                    if len(self.audio_data) % 10 == 0:
                        print(f"🔴 Captured {len(self.audio_data)} audio chunks")
                except Exception as callback_error:
                    print(f"Error in audio callback: {callback_error}")

        try:
            print(f"🎙️ Starting continuous audio recording at {self.sample_rate} Hz...")
            print(f"Recording state at start: {self.recording}")

            print("🎙️ Using system default device")

            with sd.InputStream(
                callback=audio_callback,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=1024,
            ):
                print("✅ Audio stream opened for continuous recording")
                chunk_count = 0
                while self.recording:
                    time.sleep(0.1)  # Check every 100ms
                    current_chunks = len(self.audio_data)
                    if current_chunks > chunk_count:
                        print(f"📊 Recording... {current_chunks} audio chunks captured")
                        chunk_count = current_chunks

            print(f"🔴 Audio stream closed. Total chunks: {len(self.audio_data)}")

        except Exception as e:
            print(f"❌ Error in continuous audio recording: {e}")
            print(f"Sample rate attempted: {self.sample_rate} Hz")
            print("Possible issues:")
            print("1. Microphone permissions not granted")
            print("2. Another app is using the microphone")
            print("3. Hardware/driver issue")
            print("4. Incompatible audio format")
            self.recording = False


class TranscriptionService:
    """Handles speech-to-text transcription."""

    def __init__(self, model_name: str = Config.WHISPER_MODEL):
        print("Loading Whisper model...")
        self.model = whisper.load_model(model_name)

    def transcribe(
        self, audio_data: NDArray[np.float32], sample_rate: int
    ) -> tuple[str | None, str]:
        """Transcribe audio data to text."""
        try:
            print("=== Transcription Debug ===")
            print(f"Audio data shape: {audio_data.shape}")
            print(f"Audio data type: {audio_data.dtype}")
            print(f"Sample rate: {sample_rate} Hz")
            print(f"Audio duration: {len(audio_data) / sample_rate:.2f} seconds")
            print(f"Audio min/max: {audio_data.min():.4f} / {audio_data.max():.4f}")
            print(f"Audio RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")

            # Check if audio is too short
            duration = len(audio_data) / sample_rate
            if duration < MIN_AUDIO_DURATION:
                return (
                    None,
                    f"❌ Audio too short: {duration:.2f}s (minimum {MIN_AUDIO_DURATION}s)",
                )

            # Check if audio is too quiet
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < MIN_AUDIO_RMS_THRESHOLD:
                return None, f"❌ Audio too quiet: RMS {rms:.6f} (try speaking louder)"

            # Normalize audio for Whisper (Whisper expects float32 in range [-1, 1])
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
                print(f"Normalized audio by {max_val:.4f}")
            else:
                return None, "❌ Audio contains no signal (all zeros)"

            # Resample to 16kHz if needed (Whisper's expected sample rate)
            whisper_sample_rate = 16000
            if sample_rate != whisper_sample_rate:
                print(f"Resampling from {sample_rate} Hz to {whisper_sample_rate} Hz...")
                from scipy import signal

                num_samples = int(len(audio_data) * whisper_sample_rate / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                sample_rate = whisper_sample_rate
                print(f"Resampled audio shape: {audio_data.shape}")

            print("Sending to Whisper for transcription...")
            result = self.model.transcribe(audio_data)

            print(f"Whisper raw result: {result}")
            transcribed_text = result["text"].strip()
            confidence = result.get("confidence", "unknown")
            language = result.get("language", "unknown")

            print(f"Transcribed text: '{transcribed_text}'")
            print(f"Confidence: {confidence}")
            print(f"Language: {language}")

            if not transcribed_text:
                return None, "❌ No speech detected - try speaking more clearly"

            return transcribed_text, f'🎤 Transcribed: "{transcribed_text}"'

        except Exception as e:
            print(f"Transcription exception: {e}")
            print(f"Exception type: {type(e).__name__}")
            return None, f"❌ Transcription Error: {str(e)}"


class LLMService:
    """Handles LLM interactions."""

    def __init__(self, model_name: str = Config.DEEPSEEK_MODEL):
        self.model_name = model_name
        # Import search tool
        # Initialize tool manager
        self.tool_manager = ToolManager()
        self.search_available = self.tool_manager.has_tools
        self.tools = self.tool_manager.get_tool_definitions()

    def get_response(self, messages: list[MessageDict]) -> tuple[str | None, str]:
        """Get response from LLM (non-streaming)."""
        try:
            print(f"🔍 LLM Request: {len(messages)} messages")
            print(f"🔍 Tools available: {len(self.tools) if self.tools else 0}")
            print(f"🔍 Model: {self.model_name}")
            print(f"🔍 Last message: {messages[-1]['content'][:100]}...")

            # First call with tools available
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                tools=self.tools if self.tools else None,
                stream=False,
            )

            # Check if the model wants to use tools
            tool_calls = getattr(response.message, "tool_calls", None)

            print(f"🔍 Model response received. Tool calls: {tool_calls is not None}")
            print(
                f"🔍 Response content: {response.message.content[:100] if response.message.content else 'None'}..."
            )

            if tool_calls:
                print(f"🔧 Model made {len(tool_calls)} tool calls")
                # Process tool calls using ToolManager
                messages_with_tools = self.tool_manager.process_tool_calls(
                    tool_calls, messages, response.message.content or ""
                )

                # Get final response with tool results
                final_response = ollama.chat(
                    model=self.model_name, messages=messages_with_tools, stream=False
                )

                raw_response = final_response.message.content or ""
            else:
                raw_response = response.message.content or ""

            cleaned_response = self.parse_deepseek_response(raw_response)
            return cleaned_response, "🤖 AI responded, generating speech..."

        except Exception as e:
            return None, f"❌ Response Error: {str(e)}"

    def get_streaming_response(self, messages: list[MessageDict]):
        """Get streaming response from LLM."""
        try:
            print(f"🔍 STREAMING LLM Request: {len(messages)} messages")
            print(
                f"🔍 STREAMING Tools available: {len(self.tools) if self.tools else 0}"
            )
            print(f"🔍 STREAMING Model: {self.model_name}")
            print(f"🔍 STREAMING Last message: {messages[-1]['content'][:100]}...")

            # First call with tools available
            response_stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                tools=self.tools if self.tools else None,
                stream=True,
            )

            accumulated_response = ""
            tool_calls: list[ollama.ToolCall] = []

            # Process the streaming response
            for chunk in response_stream:
                if "message" in chunk:
                    message = chunk["message"]

                    # Handle content
                    if "content" in message and message["content"]:
                        content = message["content"]
                        accumulated_response += content
                        yield content, accumulated_response

                    # Collect tool calls
                    if "tool_calls" in message and message["tool_calls"]:
                        print(
                            f"🔧 STREAMING: Found tool calls in chunk: {len(message['tool_calls'])}"
                        )
                        tool_calls.extend(message["tool_calls"])

            # If there were tool calls, process them
            print(f"🔍 STREAMING: Total tool calls collected: {len(tool_calls)}")
            if tool_calls:
                try:
                    print("🔍 STREAMING: Processing tool calls...")
                    print(f"🔍 STREAMING: Tool calls structure: {type(tool_calls[0])}")
                    print(f"🔍 STREAMING: Tool call content: {tool_calls[0]}")

                    print("🔍 STREAMING: About to yield search status...")
                    yield (
                        "🔍 Searching...",
                        "🔍 Searching for additional information...",
                    )
                    print("🔍 STREAMING: Yield completed successfully")
                except Exception as e:
                    print(f"❌ STREAMING: Error in tool processing: {e}")
                    import traceback

                    traceback.print_exc()
                    return

                # Process tool calls using ToolManager
                try:
                    messages_with_tools = self.tool_manager.process_tool_calls(
                        tool_calls, messages, accumulated_response
                    )
                    print(
                        f"🔍 STREAMING: Tool processing completed with {len(messages_with_tools)} messages"
                    )
                except Exception as e:
                    print(f"❌ STREAMING: Error processing tools: {e}")
                    return

                print(
                    f"🔍 STREAMING: Making final call to model with {len(messages_with_tools)} messages"
                )
                print(
                    f"🔍 STREAMING: Final message preview: {messages_with_tools[-1]['content'][:200]}..."
                )

                # Get final streaming response with tool results
                final_stream = ollama.chat(
                    model=self.model_name, messages=messages_with_tools, stream=True
                )
                print("🔍 STREAMING: Final stream started")

                final_accumulated = ""
                for chunk in final_stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        final_accumulated += content
                        yield content, final_accumulated

        except Exception as e:
            yield None, f"❌ Response Error: {str(e)}"

    def parse_deepseek_response(self, raw_response: str) -> str:
        """Parse DeepSeek-R1 response to extract only the final answer."""
        # Remove thinking tags and content
        patterns = [
            r"<think>.*?</think>",
            r"<thinking>.*?</thinking>",
            r"<thought>.*?</thought>",
            r"<reasoning>.*?</reasoning>",
        ]

        cleaned = raw_response
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

        # Clean up extra whitespace
        cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)
        return cleaned.strip()


class TTSService:
    """Handles text-to-speech synthesis using Kokoro TTS."""

    def __init__(self, persona_manager: PersonaManager):
        if not TTS_AVAILABLE:
            print(
                "❌ kokoro-tts not available. Install with: pip install kokoro soundfile"
            )
            self.available = False
            return

        print("Initializing Kokoro TTS engine...")
        self.persona_manager = persona_manager
        self.available = True

        # Initialize Kokoro pipeline
        try:
            # Use language code 'a' for English (American)
            from kokoro import KPipeline

            self.pipeline = KPipeline(lang_code="a")
            print("✅ Kokoro TTS pipeline initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Kokoro pipeline: {e}")
            self.available = False

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS by removing markdown blocks and emojis."""
        import re

        print(f"Input text: '{text}'")

        # Remove specific markdown blocks that shouldn't be spoken
        cleaned_text = text

        # Remove code blocks entirely (```...```)
        cleaned_text = re.sub(r"```[\s\S]*?```", "", cleaned_text, flags=re.MULTILINE)
        print(f"After code block removal: '{cleaned_text}'")

        # Remove inline code (`code`)
        cleaned_text = re.sub(r"`[^`]+`", "", cleaned_text)
        print(f"After inline code removal: '{cleaned_text}'")

        # Remove headers but keep the text (# Header -> Header)
        cleaned_text = re.sub(
            r"^#{1,6}\s*(.+)$", r"\1", cleaned_text, flags=re.MULTILINE
        )
        print(f"After header cleanup: '{cleaned_text}'")

        # Remove list markers but keep text (- item -> item, 1. item -> item)
        cleaned_text = re.sub(r"^\s*[-*+]\s+", "", cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r"^\s*\d+\.\s+", "", cleaned_text, flags=re.MULTILINE)
        print(f"After list marker removal: '{cleaned_text}'")

        # Remove links but keep the text ([text](url) -> text)
        cleaned_text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned_text)
        print(f"After link cleanup: '{cleaned_text}'")

        # Remove bold/italic markdown but keep text (**bold** -> bold, *italic* -> italic)
        cleaned_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned_text)
        cleaned_text = re.sub(r"\*([^*]+)\*", r"\1", cleaned_text)
        print(f"After bold/italic cleanup: '{cleaned_text}'")

        # Remove emojis
        emoji_pattern = re.compile(
            "["
            + "\U0001f600-\U0001f64f"  # emoticons
            + "\U0001f300-\U0001f5ff"  # symbols & pictographs
            + "\U0001f680-\U0001f6ff"  # transport & map symbols
            + "\U0001f1e0-\U0001f1ff"  # flags
            + "\U0001f900-\U0001f9ff"  # supplemental symbols
            + "]+",
            flags=re.UNICODE,
        )

        cleaned_text = emoji_pattern.sub("", cleaned_text)
        print(f"After emoji removal: '{cleaned_text}'")

        # Clean up extra whitespace and empty lines
        cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)  # Remove empty lines
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)  # Collapse whitespace
        cleaned_text = cleaned_text.strip()

        print(f"Final cleaned text: '{cleaned_text}'")

        return cleaned_text

    def synthesize(  # noqa: PLR0915
        self, text: str, persona_name: str = "Default"
    ) -> tuple[NDArray[np.float32] | None, str]:
        """Convert text to speech using persona-specific Kokoro voice."""
        if not self.available:
            return None, "❌ TTS not available"

        try:
            # Clean text for TTS: remove emojis and other non-speech elements
            cleaned_text = self._clean_text_for_tts(text)
            print(f"Original text: {text}")
            print(f"Cleaned text: {cleaned_text}")

            if not cleaned_text.strip():
                return None, "❌ TTS Error: No speakable text after cleaning"

            # Get voice settings for the persona
            voice_settings = self.persona_manager.get_voice_settings(persona_name)
            voice_name = str(voice_settings.get("voice", "af_sarah"))
            speed = float(voice_settings.get("speed", 1.0))

            print(f"Synthesizing with voice: {voice_name}, speed: {speed}")

            # Generate audio using Kokoro (returns a generator)
            try:
                print("Calling Kokoro pipeline...")
                audio_generator = self.pipeline(cleaned_text, voice=voice_name)
                print("Got audio generator, starting iteration...")
            except Exception as synth_error:
                print(f"Failed to create audio generator: {synth_error}")
                return None, f"❌ TTS Error: Synthesis failed: {synth_error}"

            # Collect audio data from generator
            audio_chunks: list[NDArray[np.float32]] = []
            chunk_count = 0
            try:
                # Process Kokoro generator chunks with proper typing
                for _, _, audio in audio_generator:
                    chunk_count += 1

                    # Type guard and processing for audio data
                    if audio is not None:
                        try:
                            audio_len = len(audio) if hasattr(audio, "__len__") else 0
                            audio_shape = (
                                audio.shape
                                if hasattr(audio, "shape")
                                else f"len={audio_len}"
                            )
                            print(f"Got chunk {chunk_count}: audio shape {audio_shape}")

                            if audio_len > 0:
                                # Convert PyTorch tensor to numpy array
                                processed_audio = audio
                                if hasattr(audio, "detach") and not isinstance(
                                    audio, np.ndarray
                                ):  # It's a PyTorch tensor
                                    processed_audio = cast(
                                        NDArray[np.float32],
                                        audio.detach().cpu().numpy(),
                                    )
                                    print(f"Converted tensor to numpy: {processed_audio.shape}")

                                # Ensure audio is numpy array and float32
                                if isinstance(processed_audio, np.ndarray):
                                    if processed_audio.dtype != np.float32:
                                        processed_audio = cast(
                                            NDArray[np.float32],
                                            processed_audio.astype(np.float32),
                                        )
                                    else:
                                        processed_audio = cast(NDArray[np.float32], processed_audio)
                                    audio_chunks.append(processed_audio)
                                    print(f"Added chunk {chunk_count} to list")
                        except Exception as chunk_error:
                            print(
                                f"Error processing chunk {chunk_count}: {chunk_error}"
                            )
                    else:
                        print(f"Chunk {chunk_count} has no valid audio data")

                print(f"Finished iterating, got {len(audio_chunks)} valid chunks")
            except Exception as gen_error:
                print(f"Generator iteration failed: {gen_error}")
                return None, f"❌ TTS Error: Generator failed: {gen_error}"

            if not audio_chunks:
                return None, "❌ TTS Error: No audio generated"

            # Concatenate all chunks into single array
            try:
                audio_data: NDArray[np.float32] = np.concatenate(audio_chunks)
                print(f"Concatenated {len(audio_chunks)} chunks into audio array")
            except ValueError as concat_error:
                print(f"Concatenation error: {concat_error}")
                print(
                    f"Chunk shapes: {[chunk.shape for chunk in audio_chunks[:5]]}"
                )  # Show first 5
                return None, "❌ TTS Error: Failed to concatenate audio chunks"

            # Kokoro outputs at 24kHz by default
            kokoro_sample_rate = 24000
            print(
                f"Generated audio: {len(audio_data)} samples at {kokoro_sample_rate}Hz"
            )
            print(f"Audio range: {audio_data.min():.4f} to {audio_data.max():.4f}")
            print(f"Audio RMS: {np.sqrt(np.mean(audio_data**2)):.6f}")

            # Apply speed adjustment by resampling (speed > 1.0 = faster, < 1.0 = slower)
            if speed != 1.0:
                from scipy import signal

                speed_adjusted_length = int(len(audio_data) / speed)
                audio_data = signal.resample(audio_data, speed_adjusted_length)
                print(f"Applied speed adjustment: {speed}x")

            # Resample to target sample rate if needed
            if kokoro_sample_rate != Config.TTS_SAMPLE_RATE:
                from scipy import signal

                num_samples = int(
                    len(audio_data) * Config.TTS_SAMPLE_RATE / kokoro_sample_rate
                )
                audio_data = signal.resample(audio_data, num_samples)
                print(
                    f"Resampled from {kokoro_sample_rate}Hz to {Config.TTS_SAMPLE_RATE}Hz"
                )

            print(
                f"Final audio for playback: {len(audio_data)} samples, range: {audio_data.min():.4f} to {audio_data.max():.4f}"
            )

            return audio_data, "🔊 Speech generated with Kokoro TTS"

        except Exception as e:
            print(f"Kokoro TTS Error: {e}")
            return None, f"❌ TTS Error: {str(e)}"


class ConversationManager:
    """Manages conversation history."""

    def __init__(self, persona_manager: PersonaManager):
        self.history: list[MessageDict] = []
        self.persona_manager = persona_manager
        self.current_persona = persona_manager.get_default_persona()

    def set_persona(self, persona_name: str) -> None:
        """Set the current persona."""
        if persona_name in self.persona_manager.get_persona_names():
            self.current_persona = persona_name
        else:
            print(f"Warning: Persona '{persona_name}' not found, using default")
            self.current_persona = self.persona_manager.get_default_persona()

    def get_current_persona(self) -> str:
        """Get the current persona name."""
        return self.current_persona

    def add_user_message(self, text: str) -> None:
        """Add user message to history."""
        self.history.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        """Add assistant message to history with current persona."""
        self.history.append({"role": "assistant", "content": text})

    def get_messages_for_llm(self) -> list[MessageDict]:
        """Get messages formatted for LLM with current persona."""
        system_prompt = self.persona_manager.get_persona_prompt(self.current_persona)
        system_message: MessageDict = {"role": "system", "content": system_prompt}
        return [system_message] + self.history

    def get_chat_history(self) -> list[list[str]]:
        """Get formatted chat history for display."""
        chat_history: list[list[str]] = []
        for message in self.history:
            if message["role"] == "user":
                chat_history.append(["👤 You", message["content"]])
            else:
                # Use original content for display (no cleaning needed)
                cleaned_content = message["content"]
                # Use current persona for display
                if self.current_persona == "Default":
                    prefix = "🤖 Assistant"
                else:
                    prefix = f"🤖 {self.current_persona}"
                chat_history.append([prefix, cleaned_content])
        return chat_history

    def clear(self) -> None:
        """Clear conversation history."""
        self.history = []


class VoiceAssistant:
    """Main voice assistant orchestrator."""

    def __init__(self):
        self.recorder = AudioRecorder()
        self.transcription = TranscriptionService()
        self.llm = LLMService()
        self.persona_manager = PersonaManager()
        self.tts = TTSService(self.persona_manager)
        self.conversation = ConversationManager(self.persona_manager)
        self.state = RecordingState.IDLE

    def start_recording(self) -> str:
        """Start recording audio."""
        if self.state != RecordingState.IDLE:
            return "Already processing..."

        self.state = RecordingState.RECORDING
        return self.recorder.start_recording()

    def stop_recording_and_process(self) -> Generator[ProcessingYield]:
        """Stop recording and process the audio through the pipeline."""
        if self.state != RecordingState.RECORDING:
            yield "Not currently recording", False, [], None
            return

        self.state = RecordingState.PROCESSING

        # Stop recording and get audio
        audio_data = self.recorder.stop_recording()
        if audio_data is None:
            self.state = RecordingState.IDLE
            yield "No audio recorded", False, self.conversation.get_chat_history(), None
            return

        # Stage 1: Transcription
        transcribed_text, transcription_status = self.transcription.transcribe(
            audio_data, self.recorder.sample_rate
        )

        if transcribed_text is None:
            self.state = RecordingState.IDLE
            yield (
                transcription_status,
                False,
                self.conversation.get_chat_history(),
                None,
            )
            return

        # Add user message and update UI
        self.conversation.add_user_message(transcribed_text)
        yield transcription_status, False, self.conversation.get_chat_history(), None

        # Stage 2: LLM Response
        llm_response, llm_status = self.llm.get_response(
            self.conversation.get_messages_for_llm()
        )

        if llm_response is None:
            self.state = RecordingState.IDLE
            yield llm_status, False, self.conversation.get_chat_history(), None
            return

        # Show "generating speech" status without adding response to chat yet
        yield llm_status, False, self.conversation.get_chat_history(), None

        # Stage 3: TTS Generation
        audio_output, _tts_status = self.tts.synthesize(
            llm_response, self.conversation.get_current_persona()
        )
        audio_tuple = (
            (Config.TTS_SAMPLE_RATE, audio_output) if audio_output is not None else None
        )

        # Only add assistant message to chat when audio is ready to play
        self.conversation.add_assistant_message(llm_response)

        self.state = RecordingState.IDLE
        yield (
            f'✅ Complete: "{transcribed_text}"',
            False,
            self.conversation.get_chat_history(),
            audio_tuple,
        )

    def clear_conversation(self) -> tuple[str, GradioHistory]:
        """Clear the conversation history."""
        self.conversation.clear()
        return "Conversation cleared", []


class VoiceChatInterface:
    """Enhanced ChatInterface with voice capabilities."""

    def __init__(self, assistant: VoiceAssistant):
        self.assistant = assistant

    def respond_to_message(self, message: str, _history: list[list[str]]) -> str:
        """Process a text message and return AI response."""
        # Add user message to conversation
        self.assistant.conversation.add_user_message(message)

        # Get LLM response
        llm_response, _ = self.assistant.llm.get_response(
            self.assistant.conversation.get_messages_for_llm()
        )

        if llm_response is None:
            return "Sorry, I encountered an error processing your message."

        # Add assistant message to conversation
        self.assistant.conversation.add_assistant_message(llm_response)

        # Generate TTS audio (for consistency, though not used in text interface)
        self.assistant.tts.synthesize(
            llm_response, self.assistant.conversation.get_current_persona()
        )

        # Return response as-is
        return llm_response

    def create_interface(self) -> gr.Blocks:  # noqa: PLR0915
        """Create a custom interface that combines ChatInterface with voice controls."""

        with gr.Blocks(
            title="Chatter",
            theme="soft",
            css="""
                .minimal-audio {
                    height: 30px !important;
                    opacity: 0.3;
                    transition: opacity 0.3s ease;
                }
                .minimal-audio:hover {
                    opacity: 1;
                }
                #tts-audio {
                    max-height: 40px !important;
                }
                .loading-spinner {
                    display: inline-block;
                    width: 20px;
                    height: 20px;
                    border: 3px solid #f3f3f3;
                    border-radius: 50%;
                    border-top: 3px solid #3498db;
                    animation: spin 1s linear infinite;
                    margin-right: 8px;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .processing-status {
                    display: flex;
                    align-items: center;
                    color: #666;
                    font-style: italic;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    margin: 5px 0;
                }
            """,
        ) as interface:
            gr.Markdown("# 💬 Chatter")
            gr.Markdown(
                "**Conversational AI with voice input - Use the recording button to talk**"
            )

            # Persona selection
            with gr.Row():
                persona_dropdown = gr.Dropdown(
                    choices=self.assistant.persona_manager.get_persona_names(),
                    value=self.assistant.conversation.get_current_persona(),
                    label="🎭 Select Persona",
                    info="Choose the AI assistant's personality and expertise",
                    scale=3,
                )
                chime_btn = gr.Button("🔔 Chime In", variant="primary", scale=1)

            # Main chat interface
            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True,
                bubble_full_width=False,
                render_markdown=True,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )

            # Message input (will be populated by voice or typed)
            msg_input = gr.Textbox(
                placeholder="Type your message or use voice input...",
                container=False,
                scale=4,
            )

            # Voice controls
            with gr.Row():
                start_btn = gr.Button("🎙️ Start Recording", variant="primary", size="lg")
                stop_btn = gr.Button(
                    "🔴 Stop & Transcribe", variant="stop", size="lg", visible=False
                )
                submit_btn = gr.Button("Send", variant="primary", size="lg")

            # Status display for processing feedback
            status_display = gr.HTML(
                value="", visible=False, elem_classes=["processing-status"]
            )

            # Audio for TTS (minimally visible for functionality)
            audio_output = gr.Audio(
                type="numpy",
                autoplay=True,
                visible=True,
                show_label=False,
                elem_id="tts-audio",
                elem_classes=["minimal-audio"],
            )

            # State management
            recording_state = gr.State(False)

            def create_loading_html(message: str) -> str:
                """Create HTML with loading spinner."""
                return f'<div class="processing-status"><div class="loading-spinner"></div>{message}</div>'

            def change_persona(persona_name: str) -> GradioUpdate:
                """Change the current persona."""
                self.assistant.conversation.set_persona(persona_name)
                return gr.update(
                    value=f"✅ Switched to {persona_name} persona", visible=True
                )

            def clear_chat_sync() -> tuple[GradioHistory, str, GradioUpdate]:
                """Sync conversation history when chatbot is cleared."""
                self.assistant.conversation.clear()
                return [], "", gr.update(value="", visible=False)

            # Hook into chatbot clear functionality
            chatbot.clear(
                fn=clear_chat_sync, outputs=[chatbot, msg_input, status_display]
            )

            def chime_in():
                """Get the current persona to chime in on the conversation."""
                if not self.assistant.conversation.history:
                    # No conversation yet, just start naturally
                    chime_prompt = "Start a conversation naturally from your perspective and expertise."
                else:
                    # Jump into the conversation naturally without meta-commentary
                    chime_prompt = "Continue this conversation naturally with your own thoughts, questions, or insights. Don't announce that you're chiming in - just contribute to the discussion as if you were already part of it."

                # Show chiming status
                yield (
                    self.assistant.conversation.get_chat_history(),
                    "",
                    gr.update(
                        value=create_loading_html(
                            f"{self.assistant.conversation.get_current_persona()} is thinking..."
                        ),
                        visible=True,
                    ),
                    None,
                )

                # Add the chime prompt as a user message (but don't display it in chat)
                messages_for_llm = (
                    self.assistant.conversation.get_messages_for_llm()
                    + [{"role": "user", "content": chime_prompt}]
                )

                # Get streaming LLM response
                try:
                    accumulated_response = ""
                    current_history = self.assistant.conversation.get_chat_history()

                    # Add empty assistant message to show streaming
                    current_history.append(
                        [f"🤖 {self.assistant.conversation.get_current_persona()}", ""]
                    )

                    for (
                        chunk_content,
                        full_response,
                    ) in self.assistant.llm.get_streaming_response(messages_for_llm):
                        if chunk_content is None:  # Error case
                            yield (
                                current_history,
                                "",
                                gr.update(value=f"❌ {full_response}", visible=True),
                                None,
                            )
                            return

                        # Update the response in real-time
                        accumulated_response = full_response
                        cleaned_response = self.assistant.llm.parse_deepseek_response(
                            accumulated_response
                        )
                        current_history[-1][1] = cleaned_response
                        yield (
                            current_history,
                            "",
                            gr.update(
                                value=create_loading_html(
                                    f"{self.assistant.conversation.get_current_persona()} is responding..."
                                ),
                                visible=True,
                            ),
                            None,
                        )

                    # Clean the final response
                    final_response = self.assistant.llm.parse_deepseek_response(
                        accumulated_response
                    )
                    current_history[-1][1] = final_response

                    # Add assistant message to conversation (this will show in chat)
                    self.assistant.conversation.add_assistant_message(final_response)

                    # Show TTS generation status
                    yield (
                        self.assistant.conversation.get_chat_history(),
                        "",
                        gr.update(
                            value=create_loading_html("Generating speech..."),
                            visible=True,
                        ),
                        None,
                    )

                    # Generate TTS audio
                    audio_data, _ = self.assistant.tts.synthesize(
                        final_response,
                        self.assistant.conversation.get_current_persona(),
                    )
                    audio_output_data = (
                        (Config.TTS_SAMPLE_RATE, audio_data)
                        if audio_data is not None
                        else None
                    )

                except Exception as e:
                    yield (
                        self.assistant.conversation.get_chat_history(),
                        "",
                        gr.update(value=f"❌ Error: {str(e)}", visible=True),
                        None,
                    )
                    return

                yield (
                    self.assistant.conversation.get_chat_history(),
                    "",
                    gr.update(value="✅ Chimed in!", visible=True),
                    audio_output_data,
                )

                # Hide status after a brief delay
                time.sleep(2)
                yield (
                    self.assistant.conversation.get_chat_history(),
                    "",
                    gr.update(value="", visible=False),
                    audio_output_data,
                )

            def start_recording():
                if self.assistant.state != RecordingState.IDLE:
                    return (
                        False,
                        gr.update(visible=True),
                        gr.update(visible=False),
                        "",
                        gr.update(value="", visible=False),
                    )

                self.assistant.state = RecordingState.RECORDING
                result = self.assistant.recorder.start_recording()
                print(f"Recording started: {result}")  # Debug
                return (
                    True,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "",
                    gr.update(value=result, visible=True),
                )

            def stop_recording_and_transcribe():
                if self.assistant.state != RecordingState.RECORDING:
                    # Return generator for consistent interface
                    yield (
                        False,
                        gr.update(visible=True),
                        gr.update(visible=False),
                        chatbot.value,
                        "",
                        gr.update(value="", visible=False),
                        None,
                    )
                    return

                self.assistant.state = RecordingState.PROCESSING

                # Show processing status
                yield (
                    False,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    chatbot.value,
                    "",
                    gr.update(
                        value=create_loading_html("Processing audio..."), visible=True
                    ),
                    None,
                )

                # Stop recording and get the transcribed text
                print(
                    f"Recording state before stop: {self.assistant.recorder.recording}"
                )
                print(
                    f"Audio chunks before stop: {len(self.assistant.recorder.audio_data) if self.assistant.recorder.audio_data else 0}"
                )
                audio_data = self.assistant.recorder.stop_recording()
                print(f"Audio data received: {audio_data is not None}")
                if audio_data is not None:
                    print(f"Audio data length: {len(audio_data)}")
                else:
                    print(
                        "❌ Audio data is None - check console for stop_recording debug output"
                    )

                if audio_data is None:
                    self.assistant.state = RecordingState.IDLE
                    yield (
                        False,
                        gr.update(visible=True),
                        gr.update(visible=False),
                        chatbot.value,
                        "",
                        gr.update(value="No audio recorded", visible=True),
                        None,
                    )
                    return

                # Show transcription status
                yield (
                    False,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    chatbot.value,
                    "",
                    gr.update(
                        value=create_loading_html("Transcribing speech..."),
                        visible=True,
                    ),
                    None,
                )

                # Transcribe the audio
                transcribed_text, transcription_status = (
                    self.assistant.transcription.transcribe(
                        audio_data, self.assistant.recorder.sample_rate
                    )
                )
                print(f"Transcription result: {transcribed_text}")  # Debug

                if transcribed_text is None:
                    self.assistant.state = RecordingState.IDLE
                    yield (
                        False,
                        gr.update(visible=True),
                        gr.update(visible=False),
                        chatbot.value,
                        "",
                        gr.update(
                            value=transcription_status or "Transcription failed",
                            visible=True,
                        ),
                        None,
                    )
                    return

                # Populate the input box with transcribed text for user review/editing
                self.assistant.state = RecordingState.IDLE
                yield (
                    False,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    chatbot.value,
                    transcribed_text,
                    gr.update(
                        value=f'✅ Transcribed: "{transcribed_text}" - Review and click Send',
                        visible=True,
                    ),
                    None,
                )

                # Hide status after a brief delay
                time.sleep(3)
                yield (
                    False,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    chatbot.value,
                    transcribed_text,
                    gr.update(value="", visible=False),
                    None,
                )

            def process_message(message: str, history: list[list[str | None]]):
                if not message.strip():
                    return history, "", gr.update(value="", visible=False), None

                # Add user message to history
                history = history + [[message, None]]

                # Show AI processing status
                yield (
                    history,
                    "",
                    gr.update(
                        value=create_loading_html("AI is thinking..."), visible=True
                    ),
                    None,
                )

                # Add user message to conversation
                self.assistant.conversation.add_user_message(message)

                # Get streaming LLM response
                try:
                    accumulated_response = ""
                    for (
                        chunk_content,
                        full_response,
                    ) in self.assistant.llm.get_streaming_response(
                        self.assistant.conversation.get_messages_for_llm()
                    ):
                        if chunk_content is None:  # Error case
                            yield (
                                history,
                                "",
                                gr.update(value=f"❌ {full_response}", visible=True),
                                None,
                            )
                            return

                        # Update the response in real-time
                        accumulated_response = full_response
                        history[-1][1] = self.assistant.llm.parse_deepseek_response(
                            accumulated_response
                        )
                        yield (
                            history,
                            "",
                            gr.update(
                                value=create_loading_html("AI is responding..."),
                                visible=True,
                            ),
                            None,
                        )

                    # Clean the final response
                    final_response = self.assistant.llm.parse_deepseek_response(
                        accumulated_response
                    )
                    history[-1][1] = final_response

                    # Add assistant message to conversation
                    self.assistant.conversation.add_assistant_message(final_response)

                    # Show TTS generation status
                    yield (
                        history,
                        "",
                        gr.update(
                            value=create_loading_html("Generating speech..."),
                            visible=True,
                        ),
                        None,
                    )

                    # Generate TTS audio
                    audio_output_data = None
                    audio_data, _ = self.assistant.tts.synthesize(
                        final_response,
                        self.assistant.conversation.get_current_persona(),
                    )
                    if audio_data is not None:
                        audio_output_data = (Config.TTS_SAMPLE_RATE, audio_data)

                    yield (
                        history,
                        "",
                        gr.update(value="✅ Complete!", visible=True),
                        audio_output_data,
                    )

                    # Hide status after a brief delay
                    time.sleep(2)
                    yield (
                        history,
                        "",
                        gr.update(value="", visible=False),
                        audio_output_data,
                    )

                except Exception as e:
                    yield (
                        history,
                        "",
                        gr.update(value=f"❌ Error: {str(e)}", visible=True),
                        None,
                    )

            # Event handlers
            start_btn.click(
                fn=start_recording,
                outputs=[
                    recording_state,
                    start_btn,
                    stop_btn,
                    msg_input,
                    status_display,
                ],
            )

            stop_btn.click(
                fn=stop_recording_and_transcribe,
                outputs=[
                    recording_state,
                    start_btn,
                    stop_btn,
                    chatbot,
                    msg_input,
                    status_display,
                    audio_output,
                ],
            )

            submit_btn.click(
                fn=process_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input, status_display, audio_output],
            )

            msg_input.submit(
                fn=process_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input, status_display, audio_output],
            )

            # Persona and clear chat handlers
            persona_dropdown.change(
                fn=change_persona, inputs=[persona_dropdown], outputs=[status_display]
            )

            chime_btn.click(
                fn=chime_in, outputs=[chatbot, msg_input, status_display, audio_output]
            )

            # Enhanced audio auto-play functionality
            interface.load(
                None,
                None,
                None,
                js="""
                function() {

                    // Enhanced audio auto-play
                    function tryPlayAudio(audioElement) {
                        if (audioElement && (audioElement.src || audioElement.srcObject)) {
                            console.log('Attempting to play audio...');
                            audioElement.play()
                                .then(() => {
                                    console.log('✅ Audio playing successfully');
                                })
                                .catch(e => {
                                    console.log('❌ Auto-play blocked:', e);
                                    // Try to enable audio on user interaction
                                    document.addEventListener('click', function enableAudio() {
                                        audioElement.play().catch(console.log);
                                        document.removeEventListener('click', enableAudio);
                                    }, { once: true });
                                });
                        }
                    }

                    // Monitor for audio updates
                    const observer = new MutationObserver(function(mutations) {
                        mutations.forEach(function(mutation) {
                            // Check for new audio elements
                            mutation.addedNodes.forEach(function(node) {
                                if (node.nodeType === 1) {
                                    const audioElements = node.querySelectorAll ? node.querySelectorAll('audio') : [];
                                    const directAudio = node.tagName === 'AUDIO' ? [node] : [];
                                    const allAudio = [...audioElements, ...directAudio];

                                    allAudio.forEach(audioElement => {
                                        setTimeout(() => tryPlayAudio(audioElement), 50);
                                    });
                                }
                            });

                            // Check for audio source changes
                            if (mutation.type === 'attributes' && mutation.target.tagName === 'AUDIO') {
                                setTimeout(() => tryPlayAudio(mutation.target), 50);
                            }
                        });
                    });

                    // Also check existing audio elements periodically
                    setInterval(() => {
                        const audioElements = document.querySelectorAll('audio');
                        audioElements.forEach(audio => {
                            if (audio.src && audio.paused && audio.readyState >= 2) {
                                tryPlayAudio(audio);
                            }
                        });
                    }, 500);

                    observer.observe(document.body, {
                        childList: true,
                        subtree: true,
                        attributes: true,
                        attributeFilter: ['src']
                    });
                }
                """,
            )

        return interface


def main():
    """Main function to run the voice assistant."""
    print("Starting Chatter...")
    print("Loading models...")

    # Ensure all required models are available
    ModelManager.ensure_deepseek_model(Config.DEEPSEEK_MODEL)
    ModelManager.ensure_kokoro_model()

    assistant = VoiceAssistant()
    chat_interface = VoiceChatInterface(assistant)
    interface = chat_interface.create_interface()

    print("Interface created, launching server...")
    interface.launch(
        server_name=Config.SERVER_HOST, server_port=Config.SERVER_PORT, share=False
    )


if __name__ == "__main__":
    main()
