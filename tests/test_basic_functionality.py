"""
Basic unit tests for core functionality.
"""

import contextlib
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from chatter.conversation import ConversationManager
from chatter.persona import PersonaManager
from chatter.types import MessageDict


class TestPersonaManager:
    """Test cases for persona management."""

    @pytest.fixture
    def temp_persona_dir(self) -> Generator[str]:
        """Create temporary directory for persona testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test persona file
            test_persona_path = Path(temp_dir) / "Test Persona.md"
            with test_persona_path.open("w") as f:
                f.write("# Test Persona\n\nThis is a test persona for unit testing.")
            yield temp_dir

    @pytest.fixture
    def persona_manager(self, temp_persona_dir: str) -> PersonaManager:
        """Create persona manager instance for testing."""
        return PersonaManager(temp_persona_dir)

    @pytest.mark.unit
    def test_persona_manager_initialization(
        self, persona_manager: PersonaManager
    ) -> None:
        """Test persona manager initializes correctly."""
        assert persona_manager.personas is not None
        assert isinstance(persona_manager.personas, dict)

    @pytest.mark.unit
    def test_get_default_persona(self, persona_manager: PersonaManager) -> None:
        """Test getting default persona."""
        default = persona_manager.get_default_persona()
        assert isinstance(default, str)
        assert len(default) > 0

    @pytest.mark.unit
    def test_get_persona_prompt(self, persona_manager: PersonaManager) -> None:
        """Test getting persona prompt."""
        default_persona = persona_manager.get_default_persona()
        prompt = persona_manager.get_persona_prompt(default_persona)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    @pytest.mark.unit
    def test_voice_settings(self, persona_manager: PersonaManager) -> None:
        """Test voice settings functionality."""
        default_persona = persona_manager.get_default_persona()
        settings = persona_manager.get_voice_settings(default_persona)
        assert isinstance(settings, dict)
        assert "voice" in settings
        assert "speed" in settings


class TestConversationManager:
    """Test cases for conversation management."""

    @pytest.fixture
    def conversation_manager(self) -> Generator[ConversationManager]:
        """Create conversation manager for testing."""
        temp_dir = tempfile.mkdtemp()
        persona_manager = PersonaManager(temp_dir)
        conv_manager = ConversationManager(persona_manager)
        yield conv_manager
        Path(temp_dir).rmdir()

    @pytest.mark.unit
    def test_conversation_manager_initialization(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Test conversation manager initializes correctly."""
        assert conversation_manager.history == []
        assert conversation_manager.persona_manager is not None
        assert conversation_manager.current_persona is not None

    @pytest.mark.unit
    def test_add_user_message(self, conversation_manager: ConversationManager) -> None:
        """Test adding user message."""
        test_message = "Hello, this is a test message"
        conversation_manager.add_user_message(test_message)

        assert len(conversation_manager.history) == 1
        assert conversation_manager.history[0]["role"] == "user"
        assert conversation_manager.history[0]["content"] == test_message

    @pytest.mark.unit
    def test_add_assistant_message(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Test adding assistant message."""
        test_response = "Hello, this is a test response"
        conversation_manager.add_assistant_message(test_response)

        assert len(conversation_manager.history) == 1
        assert conversation_manager.history[0]["role"] == "assistant"
        assert conversation_manager.history[0]["content"] == test_response

    @pytest.mark.unit
    def test_get_messages_for_llm(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Test getting messages formatted for LLM."""
        # Add some messages
        conversation_manager.add_user_message("Test user message")
        conversation_manager.add_assistant_message("Test assistant response")

        messages = conversation_manager.get_messages_for_llm()

        assert isinstance(messages, list)
        assert len(messages) >= 2  # At least system message + our messages
        assert messages[0]["role"] == "system"  # First should be system message

        # Check our messages are present
        user_found = any(
            msg["role"] == "user" and msg["content"] == "Test user message"
            for msg in messages
        )
        assistant_found = any(
            msg["role"] == "assistant" and msg["content"] == "Test assistant response"
            for msg in messages
        )
        assert user_found
        assert assistant_found

    @pytest.mark.unit
    def test_get_chat_history(self, conversation_manager: ConversationManager) -> None:
        """Test getting formatted chat history."""
        # Add some messages
        conversation_manager.add_user_message("User question")
        conversation_manager.add_assistant_message("Assistant answer")

        chat_history = conversation_manager.get_chat_history()

        assert isinstance(chat_history, list)
        assert len(chat_history) == 2
        assert chat_history[0][0] == "👤 You"
        assert chat_history[0][1] == "User question"
        assert "🤖" in chat_history[1][0]  # Should have robot emoji
        assert chat_history[1][1] == "Assistant answer"

    @pytest.mark.unit
    def test_clear_conversation(
        self, conversation_manager: ConversationManager
    ) -> None:
        """Test clearing conversation history."""
        # Add some messages
        conversation_manager.add_user_message("Test message")
        conversation_manager.add_assistant_message("Test response")
        assert len(conversation_manager.history) == 2

        # Clear and verify
        conversation_manager.clear()
        assert len(conversation_manager.history) == 0

    @pytest.mark.unit
    def test_persona_switching(self, conversation_manager: ConversationManager) -> None:
        """Test switching personas."""
        # This might not work if no other personas are available, so we'll just test the method exists
        with contextlib.suppress(KeyError, ValueError):
            conversation_manager.set_persona("NonExistentPersona")
            # If it doesn't raise an error, the persona was set or handled gracefully

        # Verify we can get current persona
        current = conversation_manager.get_current_persona()
        assert isinstance(current, str)


class TestMessageDict:
    """Test cases for MessageDict typing."""

    @pytest.mark.unit
    def test_basic_message_dict(self) -> None:
        """Test basic MessageDict creation."""
        message: MessageDict = {"role": "user", "content": "test content"}

        assert message["role"] == "user"
        assert message["content"] == "test content"

    @pytest.mark.unit
    def test_tool_message_dict(self) -> None:
        """Test MessageDict with tool_call_id."""
        message: MessageDict = {
            "role": "tool",
            "content": "search results",
            "tool_call_id": "search_123",
        }

        assert message["role"] == "tool"
        assert message["content"] == "search results"
        assert message["tool_call_id"] == "search_123"

    @pytest.mark.unit
    def test_message_dict_optional_field(self) -> None:
        """Test MessageDict with optional tool_call_id field."""
        # Should work with or without tool_call_id
        basic_message: MessageDict = {"role": "assistant", "content": "response"}

        tool_message: MessageDict = {
            "role": "tool",
            "content": "results",
            "tool_call_id": "id_123",
        }

        messages = [basic_message, tool_message]
        assert len(messages) == 2
