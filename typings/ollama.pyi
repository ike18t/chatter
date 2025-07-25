"""Type stubs for ollama library."""

from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, Mapping, Literal

class ToolFunction:
    name: str
    arguments: Dict[str, Any]

class ToolCall:
    id: Optional[str]
    function: ToolFunction

class Message(Dict[str, Any]):
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]]
    
    def __getitem__(self, key: str) -> Any: ...
    def __contains__(self, key: str) -> bool: ...

class ChatResponse(Dict[str, Any]):
    message: Message
    model: str
    created_at: str
    response: str
    done: bool
    tool_calls: Optional[List[ToolCall]]
    
    def __getitem__(self, key: str) -> Any: ...
    def __contains__(self, key: str) -> bool: ...

from typing import overload

@overload
def chat(
    model: str,
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Any]]] = None,
    stream: Literal[False] = False,
    think: Optional[bool] = None,
    format: Optional[Union[Dict[str, Any], Literal['', 'json']]] = None,
    options: Optional[Union[Mapping[str, Any], Any]] = None,
    keep_alive: Optional[Union[float, str]] = None
) -> ChatResponse: ...

@overload
def chat(
    model: str,
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Any]]] = None,
    stream: Literal[True],
    think: Optional[bool] = None,
    format: Optional[Union[Dict[str, Any], Literal['', 'json']]] = None,
    options: Optional[Union[Mapping[str, Any], Any]] = None,
    keep_alive: Optional[Union[float, str]] = None
) -> Iterator[ChatResponse]: ...