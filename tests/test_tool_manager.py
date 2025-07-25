"""
Unit tests for tool manager functionality.
"""

import pytest
from chatter.tool_manager import ToolManager, MessageDict


class TestToolManager:
    """Test cases for the tool manager."""

    @pytest.fixture
    def tool_manager(self) -> ToolManager:
        """Create tool manager instance for testing."""
        return ToolManager()

    @pytest.mark.unit
    def test_tool_manager_initialization(self, tool_manager: ToolManager) -> None:
        """Test tool manager initializes correctly."""
        assert tool_manager.has_tools is True
        tools = tool_manager.get_tool_definitions()
        assert isinstance(tools, list)
        assert len(tools) > 0

    @pytest.mark.unit
    def test_tool_definitions_structure(self, tool_manager: ToolManager) -> None:
        """Test tool definitions have correct structure."""
        tools = tool_manager.get_tool_definitions()
        
        for tool in tools:
            assert isinstance(tool, dict)
            assert "type" in tool
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]

    @pytest.mark.unit
    def test_web_search_tool_present(self, tool_manager: ToolManager) -> None:
        """Test that web search tool is present."""
        tools = tool_manager.get_tool_definitions()
        
        web_search_tool = None
        for tool in tools:
            if tool["function"]["name"] == "web_search":
                web_search_tool = tool
                break
        
        assert web_search_tool is not None
        assert "description" in web_search_tool["function"]
        assert "CRITICAL" in web_search_tool["function"]["description"]  # Our enhancement

    @pytest.mark.unit
    def test_message_dict_typing(self, tool_manager: ToolManager) -> None:
        """Test MessageDict typing works correctly."""
        # Test basic message
        basic_message: MessageDict = {
            "role": "user",
            "content": "test message"
        }
        
        # Test tool response message
        tool_message: MessageDict = {
            "role": "tool", 
            "content": "search results",
            "tool_call_id": "test_id"
        }
        
        messages = [basic_message, tool_message]
        
        # Should not raise type errors
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "tool"

    @pytest.mark.unit
    def test_tool_count(self, tool_manager: ToolManager) -> None:
        """Test tool count functionality."""
        count = tool_manager.get_tool_count()
        assert isinstance(count, int)
        assert count > 0

    @pytest.mark.integration
    def test_tool_execution_structure(self, tool_manager: ToolManager) -> None:
        """Test tool execution workflow structure."""
        # This tests the structure without actually executing tools
        tools = tool_manager.get_tool_definitions()
        
        # Verify we have the expected web search tool
        web_search_found = False
        for tool in tools:
            if tool["function"]["name"] == "web_search":
                web_search_found = True
                # Check it has required parameters
                assert "parameters" in tool["function"]
                assert "properties" in tool["function"]["parameters"]
                assert "query" in tool["function"]["parameters"]["properties"]
        
        assert web_search_found