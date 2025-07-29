"""
Integration tests for LLM service with web search capabilities.
"""

from typing import cast

import pytest

from chatter.config import Config
from chatter.llm import LLMService
from chatter.types import MessageDict


class TestLLMIntegration:
    """Test cases for LLM service integration."""

    @pytest.fixture
    def llm_service(self) -> LLMService:
        """Create LLM service instance for testing."""
        return LLMService(Config.DEEPSEEK_MODEL)

    @pytest.mark.integration
    def test_llm_service_initialization(self, llm_service: LLMService) -> None:
        """Test LLM service initializes correctly."""
        assert llm_service.model_name == Config.DEEPSEEK_MODEL
        assert llm_service.tool_manager is not None
        assert llm_service.search_available is True
        assert len(llm_service.tools) > 0

    @pytest.mark.integration
    def test_simple_query_response(self, llm_service: LLMService) -> None:
        """Test simple query that shouldn't trigger web search."""
        messages = [
            cast(MessageDict, {"role": "user", "content": "Hello, how are you?"})
        ]

        response, status = llm_service.get_response(messages)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert isinstance(status, str)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_current_events_query_triggers_search(
        self, llm_service: LLMService
    ) -> None:
        """Test that current events queries trigger web search."""
        messages = [
            cast(
                MessageDict,
                {
                    "role": "user",
                    "content": "What is the latest news in artificial intelligence?",
                },
            )
        ]

        response, status = llm_service.get_response(messages)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert isinstance(status, str)
        # Should show evidence of web search usage
        assert any(
            keyword in response.lower()
            for keyword in ["search", "results", "web", "found"]
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_president_query_uses_search(self, llm_service: LLMService) -> None:
        """Test that president query triggers web search (addressing the reported issue)."""
        messages = [
            cast(
                MessageDict,
                {
                    "role": "user",
                    "content": "Who is the current President of the United States?",
                },
            )
        ]

        response, status = llm_service.get_response(messages)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert isinstance(status, str)
        # After our improvements, should use web search and be more cautious
        assert not response.startswith("The current President of the United States is")
        # Should show evidence of search or uncertainty
        assert any(
            keyword in response.lower()
            for keyword in ["search", "unable", "cannot", "results"]
        )

    @pytest.mark.integration
    def test_streaming_response(self, llm_service: LLMService) -> None:
        """Test streaming response functionality."""
        messages = [
            cast(
                MessageDict,
                {"role": "user", "content": "Tell me about Python programming"},
            )
        ]

        streaming_gen = llm_service.get_streaming_response(messages)
        chunks: list[str] = []

        # Collect first few chunks
        for i, (chunk_content, _) in enumerate(streaming_gen):
            if chunk_content is not None:
                chunks.append(chunk_content)
            if i >= 5:  # Limit to first 5 chunks
                break

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_streaming_with_web_search(self, llm_service: LLMService) -> None:
        """Test streaming response with web search."""
        messages = [
            cast(
                MessageDict,
                {"role": "user", "content": "What's the weather like today?"},
            )
        ]

        streaming_gen = llm_service.get_streaming_response(messages)
        search_detected = False
        chunks: list[str] = []

        # Look for search activity in streaming response
        for i, (chunk_content, _) in enumerate(streaming_gen):
            if chunk_content is not None:
                chunks.append(chunk_content)
                if "search" in chunk_content.lower():
                    search_detected = True
            if i >= 10:  # Limit to first 10 chunks
                break

        assert len(chunks) > 0
        # Should see search activity in streaming response
        assert search_detected or any(
            "search" in chunk.lower() for chunk in chunks if chunk is not None
        )
