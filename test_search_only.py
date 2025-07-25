#!/usr/bin/env python3
"""
Test web search functionality without voice interface dependencies
"""

import sys
import os
sys.path.insert(0, 'src')

def test_search_tool():
    """Test the search tool directly"""
    print("=== Testing Search Tool Directly ===")
    
    try:
        from chatter.search_tool import web_search
        print("✅ Search tool imported successfully")
        
        # Test search
        result = web_search("Python 3.12 new features")
        print("✅ Search executed successfully")
        print(f"Result length: {len(result)}")
        print(f"Preview: {result[:200]}...")
        
        if "Web search results for" in result:
            print("✅ Result format is correct")
            return True
        else:
            print("❌ Result format incorrect")  
            return False
            
    except Exception as e:
        print(f"❌ Search tool test failed: {e}")
        return False

def test_tool_definition():
    """Test that we can create tool definitions for the search"""
    print("\n=== Testing Tool Definition ===")
    
    try:
        # Manually test the tool definition logic that would be in LLMService
        from chatter.search_tool import web_search
        
        # This is the tool definition that should be created
        tool_definition = {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information when you don't know something or need recent data.",
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
        }
        
        print("✅ Tool definition created successfully")
        print(f"Tool name: {tool_definition['function']['name']}")
        print(f"Tool description: {tool_definition['function']['description'][:50]}...")
        
        # Test that we can call the function
        test_result = web_search("test query")
        print("✅ Tool function callable")
        return True
        
    except Exception as e:
        print(f"❌ Tool definition test failed: {e}")
        return False

def test_mock_tool_calling():
    """Test mock tool calling workflow"""
    print("\n=== Testing Mock Tool Calling Workflow ===")
    
    try:
        from chatter.search_tool import web_search
        
        # Simulate what happens in the LLM service when a tool call is made
        print("1. Mock LLM decides to call web_search tool...")
        
        # This would come from the LLM's tool_calls
        mock_tool_call = {
            "function": {
                "name": "web_search",
                "arguments": {"query": "React 19 features"}
            },
            "id": "test_call_id"
        }
        
        print(f"2. Executing tool call with query: {mock_tool_call['function']['arguments']['query']}")
        
        # Execute the tool call
        search_result = web_search(mock_tool_call['function']['arguments']['query'])
        
        print("3. Tool execution completed")
        print(f"   Result length: {len(search_result)}")
        
        # Create the enhanced result that would be sent back to LLM
        enhanced_result = f"Current web search results (use this information):\n\n{search_result}\n\nPlease base your response on the search results above, as they contain more current information than your training data."
        
        print("4. Enhanced result created for LLM")
        print(f"   Enhanced result length: {len(enhanced_result)}")
        
        # Create the tool message that would be added to conversation
        tool_message = {
            'role': 'tool',
            'content': enhanced_result,
            'tool_call_id': mock_tool_call['id']
        }
        
        print("5. Tool message created for conversation")
        print("✅ Mock tool calling workflow completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Mock tool calling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Web Search Integration (without voice dependencies)")
    print("=" * 60)
    
    results = []
    results.append(test_search_tool())
    results.append(test_tool_definition())
    results.append(test_mock_tool_calling())
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"✅ Passed: {sum(results)}/{len(results)} tests")
    if all(results):
        print("🎉 All web search integration tests passed!")
        print("The simplified search tool is working correctly.")
    else:
        print("❌ Some tests failed - search integration may have issues")