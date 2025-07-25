#!/usr/bin/env python3
"""
Test script to verify web search integration with DeepSeek
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatter.main import LLMService, ModelManager, Config

def test_search_integration():
    """Test the web search integration"""
    print("Testing Web Search Integration with Llama 3.1")
    print("=" * 50)
    
    # Ensure the model is available
    print(f"Ensuring model {Config.DEEPSEEK_MODEL} is available...")
    try:
        ModelManager.ensure_deepseek_model(Config.DEEPSEEK_MODEL)
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return
    
    # Initialize LLM service
    llm = LLMService()
    
    if not llm.search_available:
        print("❌ Search tool not available - cannot test integration")
        return
    
    print(f"✅ Search tool available: {len(llm.tools)} tools defined")
    print(f"Tool: {llm.tools[0]['function']['name']}")
    
    # Test messages that should trigger search
    test_cases = [
        {
            "query": "I need current information about React 19 new features. Please use web search to find the latest information.",
            "should_search": True
        },
        {
            "query": "How do Python lists work?", 
            "should_search": False
        },
        {
            "query": "Search the web for what's new in JavaScript 2024",
            "should_search": True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['query']} ---")
        
        messages = [
            {"role": "user", "content": test_case['query']}
        ]
        
        try:
            response, status = llm.get_response(messages)
            print(f"Status: {status}")
            print(f"Response length: {len(response) if response else 0} chars")
            if response:
                print(f"Response preview: {response[:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_search_integration()