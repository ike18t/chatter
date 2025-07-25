"""
Type stubs for kokoro library.
Only covers the functionality we actually use in our codebase.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray
import numpy as np

class KPipeline:
    """Kokoro TTS Pipeline for text-to-speech synthesis."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs: Any
    ) -> None: ...
    
    def __call__(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs: Any
    ) -> Tuple[NDArray[np.float32], int]: ...  # (audio_data, sample_rate)
    
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs: Any
    ) -> Tuple[NDArray[np.float32], int]: ...

# Available voices (common ones)
AVAILABLE_VOICES: List[str]

# Utility functions
def list_voices() -> List[str]: ...

def load_model(model_name: str = "kokoro-v0_19") -> KPipeline: ...