"""
Tool Management System

Handles tool definitions, execution, and integration with LLM services.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict


if TYPE_CHECKING:
    from ollama import ToolCall


class SerializedToolCall(TypedDict):
    id: str
    type: str
    function: dict[str, Any]


class MessageDict(TypedDict):
    role: str
    content: str
    tool_call_id: NotRequired[str | None]  # For tool response messages
    tool_calls: NotRequired[
        list[SerializedToolCall]
    ]  # For assistant messages with tool calls


@dataclass(frozen=True)
class ToolConfig:
    """Configuration for tool management.

    This dataclass is frozen to prevent accidental modification after creation.
    """

    web_search_description: str = (
        "Search the web for current information when you don't know something "
        "or need recent data. Use this when you encounter knowledge gaps, "
        "need recent updates, or are asked about current events. "
        "CRITICAL: ALWAYS use web search for ANY question about current events, "
        "politics, recent news, current office holders, or anything that could "
        "have changed since your training. Do NOT rely on training data for "
        "current information - ALWAYS search first."
    )
    default_tool_id: str = "web_search"
    search_result_instruction: str = (
        "Current web search results (use this information):\n\n{search_result}\n\n"
        "Please base your response on the search results above, as they contain "
        "more current information than your training data."
    )
    priority_message: str = (
        "You have received web search results. Use the factual information from "
        "these search results in your response, as they contain current information "
        "that may be more accurate than your training data."
    )


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors.

    This exception is raised when a tool fails to execute properly,
    providing more specific error handling than generic exceptions.
    """


class ToolManager:
    """Manages tool definitions and execution for LLM services.

    This class handles loading, defining, and executing tools for LLM
    integration, providing a clean separation between tool management
    and the main LLM service logic.
    """

    def __init__(self, config: ToolConfig | None = None) -> None:
        """
        Initialize the tool manager.

        Args:
            config: Tool configuration (uses default if not provided)
        """
        self.config = config or ToolConfig()
        self.available_tools: dict[str, Any] = {}
        self._load_tools()

    def _load_tools(self) -> None:
        """Load available tools.

        Currently loads the web search tool if available.
        Can be extended to load additional tools in the future.
        """
        try:
            from .search_tool import web_search

            self.available_tools["web_search"] = web_search
            print("✅ Web search tool loaded successfully")
        except ImportError as e:
            print(f"⚠️  Web search tool not available: {e}")

    @property
    def has_tools(self) -> bool:
        """Check if any tools are available.

        Returns:
            True if at least one tool is available, False otherwise
        """
        return len(self.available_tools) > 0

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get tool definitions for LLM.

        Returns:
            List of tool definitions in OpenAI function calling format
        """
        if "web_search" not in self.available_tools:
            return []

        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": self.config.web_search_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    def execute_tool(self, tool_name: str, **kwargs: Any) -> str:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If tool execution fails
        """
        if tool_name not in self.available_tools:
            raise ToolExecutionError(f"Tool '{tool_name}' not available")

        try:
            return self.available_tools[tool_name](**kwargs)
        except Exception as e:
            raise ToolExecutionError(f"Tool '{tool_name}' execution failed: {e}") from e

    def serialize_tool_calls(
        self, tool_calls: list["ToolCall"]
    ) -> list[SerializedToolCall]:
        """
        Convert tool calls to serializable format.

        Args:
            tool_calls: Raw tool calls from LLM response

        Returns:
            List of serialized tool calls in OpenAI format
        """
        return [
            {
                "id": getattr(tc, "id", self.config.default_tool_id),
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]

    def process_tool_calls(
        self,
        tool_calls: list["ToolCall"],
        messages: list[MessageDict],
        content: str = "",
    ) -> list[MessageDict]:
        """
        Process tool calls and return enhanced messages.

        Args:
            tool_calls: Tool calls from LLM response
            messages: Original conversation messages
            content: Assistant message content before tool calls

        Returns:
            Updated messages list with tool responses

        Note:
            This method handles the complete workflow of:
            1. Serializing tool calls
            2. Executing each tool
            3. Adding tool responses to messages
            4. Adding system messages to prioritize search results
        """
        # Serialize tool calls
        serialized_calls = self.serialize_tool_calls(tool_calls)

        # Add assistant message with tool calls
        assistant_message: MessageDict = {
            "role": "assistant",
            "content": content,
            "tool_calls": serialized_calls,
        }
        messages_with_tools: list[MessageDict] = messages + [assistant_message]

        # Execute each tool call
        for tool_call in tool_calls:
            if tool_call.function.name == "web_search":
                try:
                    query = tool_call.function.arguments["query"]
                    print(f"🔍 Executing search for: {query}")

                    search_result = self.execute_tool("web_search", query=query)
                    print(f"🔍 Search result preview: {search_result[:200]}...")

                    # Create enhanced result with instruction
                    enhanced_result = self.config.search_result_instruction.format(
                        search_result=search_result
                    )

                    # Add tool response message
                    tool_response: MessageDict = {
                        "role": "tool",
                        "content": enhanced_result,
                        "tool_call_id": getattr(
                            tool_call, "id", self.config.default_tool_id
                        ),
                    }
                    messages_with_tools.append(tool_response)

                except ToolExecutionError as e:
                    print(f"❌ Tool execution failed: {e}")
                    # Add error message as tool response
                    error_response: MessageDict = {
                        "role": "tool",
                        "content": f"Search failed: {e}",
                        "tool_call_id": getattr(
                            tool_call, "id", self.config.default_tool_id
                        ),
                    }
                    messages_with_tools.append(error_response)

        # Add priority message to encourage using search results
        priority_message: MessageDict = {
            "role": "system",
            "content": self.config.priority_message,
        }
        messages_with_tools.insert(-1, priority_message)

        return messages_with_tools

    def get_tool_count(self) -> int:
        """Get the number of available tools.

        Returns:
            Number of tools currently available
        """
        return len(self.available_tools)

    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a specific tool is available.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is available, False otherwise
        """
        return tool_name in self.available_tools
