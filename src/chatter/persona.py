"""
Manages AI assistant personas and their voice settings.
"""

from pathlib import Path

from .config import Config


class PersonaManager:
    """Manages persona prompts and voice settings from the Prompts directory."""

    def __init__(self, prompts_dir: str | None = None):
        if prompts_dir is None:
            # Use the Prompts directory relative to this file
            current_dir = Path(__file__).parent
            default_prompts_dir = current_dir / "Prompts"
            self.prompts_dir = default_prompts_dir
        else:
            self.prompts_dir = Path(prompts_dir)
        self.personas: dict[str, str] = {}
        self.voice_settings: dict[str, dict[str, str | float]] = {}
        self.load_personas()
        self.setup_voice_settings()

    def load_personas(self) -> None:
        """Load all persona files from the Prompts directory."""
        self.personas = {"Default": Config.SYSTEM_PROMPT}  # Add default first

        if not self.prompts_dir.exists():
            print(f"Warning: Prompts directory not found at {self.prompts_dir}")
            return

        # Load all .md files from the Prompts directory
        for persona_file in sorted(self.prompts_dir.glob("*.md")):
            try:
                with persona_file.open(encoding="utf-8") as f:
                    content = f.read().strip()

                # Extract persona name from filename (remove number prefix and .md extension)
                persona_name = persona_file.stem
                # Remove number prefixes like "0. ", "1. ", etc.
                if ". " in persona_name and persona_name.split(". ")[0].isdigit():
                    persona_name = persona_name.split(". ", 1)[1]

                # Use persona content as-is
                persona_content = content

                self.personas[persona_name] = persona_content
                print(f"Loaded persona: {persona_name}")

            except Exception as e:
                print(f"Error loading persona from {persona_file}: {e}")

    def get_persona_names(self) -> list[str]:
        """Get list of available persona names."""
        return list(self.personas.keys())

    def get_persona_prompt(self, persona_name: str) -> str:
        """Get the system prompt for a specific persona."""
        base_prompt = self.personas.get(persona_name, Config.SYSTEM_PROMPT)

        # Add web search instruction to all personas (unless it's already there)
        web_search_instruction = "\n\nWhen you search the web, trust and use the search results since they contain current information that may be more accurate than your training data."

        if web_search_instruction.strip() not in base_prompt:
            return base_prompt + web_search_instruction
        return base_prompt

    def get_default_persona(self) -> str:
        """Get the default persona name."""
        return "Default"

    def setup_voice_settings(self) -> None:
        """Setup unique voice models for each persona using Kokoro TTS."""
        # Kokoro TTS voices: af_ = American female, am_ = American male, bf_ = British female, bm_ = British male
        self.voice_settings = {
            "Default": {"voice": "af_sarah", "speed": 1.0},
            "Product Manager Prompt": {"voice": "am_michael", "speed": 0.9},
            "Software Architect": {"voice": "bm_george", "speed": 0.8},
            "Developer": {"voice": "af_nova", "speed": 1.1},
            "Code Explainer": {"voice": "bm_lewis", "speed": 0.8},
            "Code Reviewer": {"voice": "am_adam", "speed": 0.9},
            "Devops Engineer": {"voice": "af_jessica", "speed": 1.0},
            "Security Engineer": {"voice": "bm_george", "speed": 0.8},
            "Performance Engineer": {"voice": "am_michael", "speed": 0.9},
            "SRE": {"voice": "am_adam", "speed": 0.9},
            "QA Engineer": {"voice": "af_bella", "speed": 1.0},
            "Rogue Engineer": {"voice": "af_heart", "speed": 1.2},
            "Tech Documenter": {"voice": "bf_emma", "speed": 0.8},
            "Changelog Reviewer": {"voice": "bm_lewis", "speed": 0.8},
            "Test Engineer": {"voice": "af_nicole", "speed": 1.0},
        }

    def get_voice_settings(self, persona_name: str) -> dict[str, str | float]:
        """Get voice settings for a specific persona."""
        return self.voice_settings.get(persona_name, self.voice_settings["Default"])
