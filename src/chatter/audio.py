"""
Audio recording functionality for the Chatter application.
"""

import threading
import time
from typing import Optional, cast

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from .config import (
    AUDIO_TEST_DURATION,
    MIN_AUDIO_RMS_THRESHOLD,
    Config,
)
from .types import DeviceInfo


class AudioRecorder:
    """Handles audio recording functionality."""

    def __init__(self, sample_rate: int = Config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data: list[NDArray[np.float32]] = []
        self.record_thread: threading.Thread | None = None

        # Check audio devices on initialization
        try:
            from sounddevice import DeviceList

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
                # Device info is a dict, not a separate DeviceInfo class

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
