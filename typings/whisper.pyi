"""
Type stubs for whisper library.
Only covers the functionality we actually use in our codebase.
"""

from typing import Any, Dict, Optional, Union
from numpy.typing import NDArray
import numpy as np

# Model type
class Whisper:
    def transcribe(
        self,
        audio: Union[str, NDArray[np.float32]],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs: Any
    ) -> Dict[str, Any]: ...

# Main functions
def load_model(
    name: str,
    device: Optional[str] = None,
    download_root: Optional[str] = None,
    in_memory: bool = False
) -> Whisper: ...

def available_models() -> list[str]: ...

# Audio processing utilities
def load_audio(file: str, sr: int = 16000) -> NDArray[np.float32]: ...

def pad_or_trim(array: NDArray[np.float32], length: int = 480000) -> NDArray[np.float32]: ...

def log_mel_spectrogram(
    audio: NDArray[np.float32],
    n_mels: int = 80,
    padding: int = 0
) -> NDArray[np.float32]: ...