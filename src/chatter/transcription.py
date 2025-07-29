"""
Speech-to-text transcription using OpenAI Whisper.
"""

import numpy as np
import whisper
from numpy.typing import NDArray

from .config import MIN_AUDIO_DURATION, MIN_AUDIO_RMS_THRESHOLD, Config


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