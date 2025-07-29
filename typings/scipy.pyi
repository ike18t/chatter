"""
Type stubs for scipy library.
Only covers the functionality we actually use in our codebase.
"""

from typing import Optional, Union
from numpy.typing import NDArray
import numpy as np

# Signal processing module
class signal:
    """Signal processing functions."""

    @staticmethod
    def resample(
        x: NDArray[np.float32],
        num: int,
        t: Optional[NDArray[np.float32]] = None,
        axis: int = 0,
        window: Optional[Union[str, tuple]] = None,
        domain: str = 'time'
    ) -> NDArray[np.float32]: ...