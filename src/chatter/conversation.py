"""
Manages conversation history and persona context.
"""

from .persona import PersonaManager
from .types import MessageDict


class ConversationManager:
    """Manages conversation history."""

    def __init__(self, persona_manager: PersonaManager):
        self.history: list[MessageDict] = []
        self.persona_manager = persona_manager
        self.current_persona = persona_manager.get_default_persona()

    def set_persona(self, persona_name: str) -> None:
        """Set the current persona."""
        if persona_name in self.persona_manager.get_persona_names():
            self.current_persona = persona_name
        else:
            print(f"Warning: Persona '{persona_name}' not found, using default")
            self.current_persona = self.persona_manager.get_default_persona()

    def get_current_persona(self) -> str:
        """Get the current persona name."""
        return self.current_persona

    def add_user_message(self, text: str) -> None:
        """Add user message to history."""
        self.history.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        """Add assistant message to history with current persona."""
        self.history.append({
            "role": "assistant",
            "content": text,
            "persona": self.current_persona
        })

    def get_messages_for_llm(self) -> list[MessageDict]:
        """Get messages formatted for LLM with current persona."""
        system_prompt = self.persona_manager.get_persona_prompt(self.current_persona)
        system_message: MessageDict = {"role": "system", "content": system_prompt}
        return [system_message] + self.history

    def get_chat_history(self) -> list[list[str]]:
        """Get formatted chat history for display."""
        chat_history: list[list[str]] = []
        for message in self.history:
            if message["role"] == "user":
                chat_history.append(["👤 You", message["content"]])
            else:
                # Use original content for display (no cleaning needed)
                cleaned_content = message["content"]
                # Use the persona that generated this message, fallback to current persona
                message_persona = message.get("persona", self.current_persona)
                if message_persona == "Default":
                    prefix = "🤖 Assistant"
                else:
                    prefix = f"🤖 {message_persona}"
                chat_history.append([prefix, cleaned_content])
        return chat_history

    def clear(self) -> None:
        """Clear conversation history."""
        self.history = []
