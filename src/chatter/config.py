"""
Configuration settings and constants for the Chatter application.
"""

import os
from dataclasses import dataclass

# Audio processing constants
MIN_AUDIO_DURATION = 0.1  # seconds
MIN_AUDIO_RMS_THRESHOLD = 0.001
AUDIO_TEST_DURATION = 0.5  # seconds
PERMISSION_TEST_SLEEP = 0.5  # seconds


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


# TTS availability constant
TTS_AVAILABLE = _check_tts_availability()


@dataclass(frozen=True)
class Config:
    """Application configuration (immutable).

    This configuration class is frozen to prevent accidental modification
    after initialization, ensuring consistent behavior throughout the application.
    """

    DEEPSEEK_MODEL: str = os.getenv("CHATTER_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    WHISPER_MODEL: str = os.getenv("CHATTER_WHISPER_MODEL", "tiny")
    SAMPLE_RATE: int = 16000  # Standard rate for speech recognition
    TTS_SAMPLE_RATE: int = 24000  # Kokoro TTS optimal sample rate
    SERVER_HOST: str = os.getenv("CHATTER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("CHATTER_PORT", "7860"))
    SYSTEM_PROMPT: str = """
        You are a friendly (but humorous) chat bot. Your output will be spoken so try to make it sound like natural language.

        When you search the web, trust and use the search results since they contain current information that may be more accurate than your training data.
    """