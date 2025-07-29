"""
Type stubs for kokoro library.
Only covers the functionality we actually use in our codebase.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from numpy.typing import NDArray
import numpy as np

# Kokoro TTS types
KokoroAudioData = Union[NDArray[np.float32], Any]  # Audio can be tensor or numpy
KokoroGeneratorState = Any  # Internal generator state
KokoroPhonemeState = Any  # Internal phoneme state
KokoroAudioChunk = Tuple[KokoroGeneratorState, KokoroPhonemeState, KokoroAudioData]

class KPipeline:
    """Kokoro TTS Pipeline for text-to-speech synthesis."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        **kwargs: Any
    ) -> None: ...

    def __call__(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs: Any
    ) -> Iterator[KokoroAudioChunk]: ...  # Returns generator of (state, phonemes, audio)

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs: Any
    ) -> Tuple[NDArray[np.float32], int]: ...  # Direct synthesis returns (audio, sample_rate)

# Available voices (common ones)
AVAILABLE_VOICES: List[str]

# Utility functions
def list_voices() -> List[str]: ...

def load_model(model_name: str = "kokoro-v0_19") -> KPipeline: ...