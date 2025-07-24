#!/usr/bin/env python3
"""Simple runner script for the voice assistant."""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from voice_ai_assistant.main import main

if __name__ == "__main__":
    main()