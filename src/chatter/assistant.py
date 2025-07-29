"""
Main voice assistant orchestrator.
"""

from collections.abc import Generator
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .audio import AudioRecorder
from .config import Config
from .conversation import ConversationManager
from .llm import LLMService
from .persona import PersonaManager
from .transcription import TranscriptionService
from .tts import TTSService


class RecordingState(Enum):
    """Recording states."""

    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


# Type alias for Gradio yield
ProcessingYield = tuple[str, bool, list[list[str]], tuple[int, NDArray[np.float32]] | None]
GradioHistory = list[list[str]]


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
