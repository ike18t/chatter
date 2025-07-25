"""
Type stubs for sounddevice library.
Only covers the functionality we actually use in our codebase.
"""

from typing import Any, Callable, Dict, List, Optional, Union
from numpy.typing import NDArray
import numpy as np

# Device info type
from typing import TypedDict

class DeviceInfo(TypedDict):
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi: int
    index: int

DeviceList = List[DeviceInfo]

# Main functions we use
def query_devices(device: Optional[Union[int, str]] = None, kind: Optional[str] = None) -> Union[DeviceList, DeviceInfo]: ...

def check_input_settings(
    device: Optional[Union[int, str]] = None,
    channels: Optional[int] = None,
    dtype: Optional[Any] = None,
    samplerate: Optional[float] = None,
) -> None: ...

def check_output_settings(
    device: Optional[Union[int, str]] = None,
    channels: Optional[int] = None,
    dtype: Optional[Any] = None,
    samplerate: Optional[float] = None,
) -> None: ...

# Stream classes
class InputStream:
    def __init__(
        self,
        samplerate: Optional[float] = None,
        blocksize: Optional[int] = None,
        device: Optional[Union[int, str]] = None,
        channels: Optional[int] = None,
        dtype: Optional[Any] = None,
        callback: Optional[Callable] = None,
        **kwargs: Any
    ) -> None: ...
    
    def __enter__(self) -> 'InputStream': ...
    def __exit__(self, *args: Any) -> None: ...

class OutputStream:
    def __init__(
        self,
        samplerate: Optional[float] = None,
        blocksize: Optional[int] = None,
        device: Optional[Union[int, str]] = None,
        channels: Optional[int] = None,
        dtype: Optional[Any] = None,
        callback: Optional[Callable] = None,
        **kwargs: Any
    ) -> None: ...
    
    def __enter__(self) -> 'OutputStream': ...
    def __exit__(self, *args: Any) -> None: ...
    def write(self, data: NDArray[Any]) -> None: ...

# Callback signature types
CallbackData = NDArray[np.float32]

# Status can be None or have status flags
class CallbackFlags:
    input_underflow: bool
    input_overflow: bool
    output_underflow: bool
    output_overflow: bool
    priming_output: bool

CallbackStatus = Optional[CallbackFlags]

# Callback function type
AudioCallback = Callable[[CallbackData, int, float, CallbackStatus], None]