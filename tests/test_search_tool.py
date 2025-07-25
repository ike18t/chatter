"""
Unit tests for web search functionality.
"""

import pytest
from chatter.search_tool import web_search


class TestWebSearch:
    """Test cases for the web search tool."""

    @pytest.mark.unit
    def test_web_search_basic_functionality(self):
        """Test basic web search functionality."""
        # Test search with a simple query
        result = web_search("Python programming language")
        
        # Basic assertions
        assert isinstance(result, str)
        assert len(result) > 0
        assert "search" in result.lower() or "result" in result.lower()

    @pytest.mark.unit
    def test_web_search_current_events(self):
        """Test web search for current events."""
        result = web_search("current news today")
        
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.unit
    def test_web_search_technical_query(self):
        """Test web search for technical information."""
        result = web_search("artificial intelligence developments 2025")
        
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.unit
    def test_web_search_empty_query(self):
        """Test web search with empty query."""
        result = web_search("")
        
        # Should handle empty query gracefully
        assert isinstance(result, str)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_web_search_multiple_queries(self):
        """Test multiple consecutive web searches."""
        queries = [
            "weather forecast",
            "latest technology news", 
            "Python 3.13 features"
        ]
        
        for query in queries:
            result = web_search(query)
            assert isinstance(result, str)
            assert len(result) > 0