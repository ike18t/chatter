"""
Common type definitions for Chatter.

This module contains shared TypedDict and other type definitions
used throughout the application to ensure type consistency.
"""

from typing import Any, NotRequired, TypedDict

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer


class SerializedToolCall(TypedDict):
    """TypedDict for serialized tool calls in assistant messages."""
    
    id: str
    type: str
    function: dict[str, Any]  # Using Any to accommodate both string and other values


class MessageDict(TypedDict):
    """TypedDict for message objects in the conversation.
    
    This is used for both internal message handling and LLM API communication.
    """
    
    role: str
    content: str
    tool_call_id: NotRequired[str | None]  # For tool response messages
    tool_calls: NotRequired[
        list[SerializedToolCall]
    ]  # For assistant messages with tool calls
    persona: NotRequired[str | None]  # For assistant messages - tracks which persona generated them


class DeviceInfo(TypedDict):
    """Represents audio device information from sounddevice."""

    name: str
    index: int
    hostapi: int
    max_input_channels: int
    max_output_channels: int
    default_low_input_latency: float
    default_low_output_latency: float
    default_high_input_latency: float
    default_high_output_latency: float
    default_samplerate: float


class ModelCache(TypedDict):
    """Cache for loaded model and tokenizer."""
    
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer