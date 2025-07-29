"""
Minimal type stubs for transformers library to address the most critical type checking issues.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

# Type aliases
PathLike = Union[str, Path]

class BatchEncoding(Dict[str, Any]):
    pass

class PreTrainedModel:
    device: torch.device
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, **kwargs: Any) -> "PreTrainedModel": ...
    
    def to(self, device: Any) -> "PreTrainedModel": ...
    
    def generate(self, **kwargs: Any) -> torch.Tensor: ...

class PreTrainedTokenizer:
    eos_token_id: Any
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, **kwargs: Any) -> "PreTrainedTokenizer": ...
    
    def apply_chat_template(
        self, 
        conversation: Any,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: Any
    ) -> Any: ...
    
    def __call__(self, text: Any, **kwargs: Any) -> BatchEncoding: ...
    
    def decode(self, token_ids: Any, **kwargs: Any) -> str: ...

class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, **kwargs: Any) -> PreTrainedModel: ...

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, **kwargs: Any) -> PreTrainedTokenizer: ...