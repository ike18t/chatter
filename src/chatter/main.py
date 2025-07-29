# Standard library imports
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
    load_dotenv(".env.local", override=True)  # Override with .env.local if it exists
except ImportError:
    pass  # python-dotenv not installed, skip

# Local imports
from .assistant import VoiceAssistant
from .config import Config
from .llm import ModelManager
from .ui import VoiceChatInterface


def main():
    """Main function to run the voice assistant."""
    print("Starting Chatter...")
    print("Loading models...")

    # Ensure all required models are available
    ModelManager.ensure_deepseek_model(Config.DEEPSEEK_MODEL)
    ModelManager.ensure_kokoro_model()

    assistant = VoiceAssistant()
    chat_interface = VoiceChatInterface(assistant)
    interface = chat_interface.create_interface()

    print("Interface created, launching server...")
    interface.launch(
        server_name=Config.SERVER_HOST, server_port=Config.SERVER_PORT, share=False
    )


if __name__ == "__main__":
    main()