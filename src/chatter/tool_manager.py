"""
Tool Management System

Handles tool definitions, execution, and integration with LLM services.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass 
class ToolConfig:
    """Configuration for tool management."""
    web_search_description: str = (
        "Search the web for current information when you don't know something or need recent data. "
        "Use this when you encounter knowledge gaps, need recent updates, or are asked about current events. "
        "IMPORTANT: Always prioritize and use the search results over your training knowledge, "
        "especially for current events and recent information."
    )
    default_tool_id: str = "web_search"
    search_result_instruction: str = (
        "Current web search results (use this information):\n\n{search_result}\n\n"
        "Please base your response on the search results above, as they contain more current "
        "information than your training data."
    )
    priority_message: str = (
        "You have received web search results. Use the factual information from these search "
        "results in your response, as they contain current information that may be more "
        "accurate than your training data."
    )


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors."""
    pass


class ToolManager:
    """Manages tool definitions and execution for LLM services."""
    
    def __init__(self, config: ToolConfig = None):
        """
        Initialize the tool manager.
        
        Args:
            config: Tool configuration (uses default if not provided)
        """
        self.config = config or ToolConfig()
        self.available_tools = {}
        self._load_tools()
    
    def _load_tools(self) -> None:
        """Load available tools."""
        try:
            from .search_tool import web_search
            self.available_tools['web_search'] = web_search
            print("✅ Web search tool loaded successfully")
        except ImportError as e:
            print(f"⚠️  Web search tool not available: {e}")
    
    @property
    def has_tools(self) -> bool:
        """Check if any tools are available."""
        return len(self.available_tools) > 0
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        Get tool definitions for LLM.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        if 'web_search' not in self.available_tools:
            return []
        
        return [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": self.config.web_search_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for"
                        }
                    },
                    "required": ["query"]
                }
            }
        }]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
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
            raise ToolExecutionError(f"Tool '{tool_name}' execution failed: {e}")
    
    def serialize_tool_calls(self, tool_calls) -> List[Dict]:
        """
        Convert tool calls to serializable format.
        
        Args:
            tool_calls: Raw tool calls from LLM response
            
        Returns:
            List of serialized tool calls
        """
        return [
            {
                'id': getattr(tc, 'id', self.config.default_tool_id),
                'function': {
                    'name': tc.function.name,
                    'arguments': tc.function.arguments
                }
            }
            for tc in tool_calls
        ]
    
    def process_tool_calls(self, tool_calls, messages: List[Dict], content: str = "") -> List[Dict]:
        """
        Process tool calls and return enhanced messages.
        
        Args:
            tool_calls: Tool calls from LLM response
            messages: Original conversation messages
            content: Assistant message content before tool calls
            
        Returns:
            Updated messages list with tool responses
        """
        # Serialize tool calls
        serialized_calls = self.serialize_tool_calls(tool_calls)
        
        # Add assistant message with tool calls
        messages_with_tools = messages + [{
            'role': 'assistant',
            'content': content,
            'tool_calls': serialized_calls
        }]
        
        # Execute each tool call
        for tool_call in tool_calls:
            if tool_call.function.name == 'web_search':
                try:
                    query = tool_call.function.arguments['query']
                    print(f"🔍 Executing search for: {query}")
                    
                    search_result = self.execute_tool('web_search', query=query)
                    print(f"🔍 Search result preview: {search_result[:200]}...")
                    
                    # Create enhanced result with instruction
                    enhanced_result = self.config.search_result_instruction.format(
                        search_result=search_result
                    )
                    
                    # Add tool response message
                    messages_with_tools.append({
                        'role': 'tool',
                        'content': enhanced_result,
                        'tool_call_id': getattr(tool_call, 'id', self.config.default_tool_id)
                    })
                    
                except ToolExecutionError as e:
                    print(f"❌ Tool execution failed: {e}")
                    # Add error message as tool response
                    messages_with_tools.append({
                        'role': 'tool',
                        'content': f"Search failed: {e}",
                        'tool_call_id': getattr(tool_call, 'id', self.config.default_tool_id)
                    })
        
        # Add priority message to encourage using search results
        messages_with_tools.insert(-1, {
            'role': 'system',
            'content': self.config.priority_message
        })
        
        return messages_with_tools
    
    def get_tool_count(self) -> int:
        """Get the number of available tools."""
        return len(self.available_tools)
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a specific tool is available."""
        return tool_name in self.available_tools