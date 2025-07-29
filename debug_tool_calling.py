#!/usr/bin/env python3
"""
Simple script to test tool calling functionality
"""

import sys
import os
sys.path.insert(0, 'src')

from chatter.main import LLMService

def test_tool_calling():
    """Test if tool calling is working"""
    print("=== Testing Tool Calling ===")

    llm = LLMService()
    print(f"Search available: {llm.search_available}")

    # Test with explicit request for search
    test_messages = [
        {"role": "user", "content": "Please search the web for React 19 new features and tell me about them"}
    ]

    print("\nSending request to model...")
    print("Message:", test_messages[0]["content"])
    print("\nLooking for tool calls...")

    response, status = llm.get_response(test_messages)

    print(f"\nFinal response length: {len(response)}")
    print(f"Status: {status}")
    print(f"Response preview: {response[:300]}...")

if __name__ == "__main__":
    test_tool_calling()