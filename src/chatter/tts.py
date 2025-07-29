"""
Text-to-speech synthesis using Kokoro TTS.
"""

import re
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .config import TTS_AVAILABLE, Config
from .persona import PersonaManager


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

    def synthesize(
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