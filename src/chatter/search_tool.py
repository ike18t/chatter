"""
Simple Web Search Tool

A tool that uses LLM analysis to extract information from web search results.
"""

from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import ollama
import requests


# Constants
ANALYSIS_PROMPT_TEMPLATE = """You are an expert at analyzing search result HTML and extracting specific, factual information.

Query: "{query}"

Your task: Extract SPECIFIC FACTUAL DATA from the search results. Focus on finding and clearly stating:
- Exact numbers, percentages, scores, ratings
- Specific dates, names, titles
- Direct facts that answer the query
- Current information that can be stated definitively

CRITICAL: If you find specific data (like "87%" or "top 5" or a person's name), STATE IT CLEARLY AND DIRECTLY. Don't be vague or cautious - extract and present the specific facts you find.

Examples of good responses:
- "Rotten Tomatoes score: 87%"
- "Current president: Donald Trump"
- "Release date: July 25, 2025"
- "Box office: $150 million"

HTML to analyze:
{html}

Extract and clearly state the specific facts that answer the query:"""


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for web search functionality.

    This dataclass is frozen to prevent accidental modification after creation.
    """
    # Search behavior
    temporal_prefix: str = "as of today:"
    request_timeout: int = 15
    html_truncation_limit: int = 8000

    # LLM settings
    llm_model: str = "llama3.1:8b"

    # HTTP settings
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # Search URLs
    search_base_url: str = "https://html.duckduckgo.com/html/"


def web_search(
    query: str,
    max_results: int = 3,
    config: Optional[SearchConfig] = None
) -> str:
    """
    Search the web for information when you don't know something.

    Args:
        query: What to search for (e.g., "React 19 new features")
        max_results: Maximum number of results to return (default: 3)
        config: Search configuration (uses default if not provided)

    Returns:
        Formatted search results as a string

    Example:
        >>> result = web_search("What's new in TypeScript 5.3")
        >>> print(result)
        Web search results for 'What's new in TypeScript 5.3':
        ...

    Raises:
        requests.exceptions.RequestException: For network-related errors
    """
    if config is None:
        config = SearchConfig()

    try:
        # Add temporal context to help find current information
        enhanced_query = f"{config.temporal_prefix} {query}"
        print(f"🔍 Searching for: {enhanced_query}")

        # Get search results using LLM analysis
        search_result = _search_with_llm_analysis(enhanced_query, config)

        if search_result:
            return f"Web search results for '{query}':\n\n{search_result}"
        else:
            return _get_fallback_response(query)

    except requests.exceptions.RequestException as e:
        return f"Search failed due to network error: {e}"
    except Exception as e:
        return f"Search failed: {e}"


def _search_with_llm_analysis(query: str, config: SearchConfig) -> str:
    """
    Perform web search and use LLM to analyze results.

    Args:
        query: The search query
        config: Search configuration

    Returns:
        LLM-analyzed search results or empty string if failed

    Raises:
        requests.exceptions.RequestException: For network errors
        Exception: For other errors during analysis
    """
    try:
        # Fetch HTML from DuckDuckGo search
        html = _fetch_search_html(query, config)
        if not html:
            return ""

        # Use LLM to analyze the HTML and extract information
        return _analyze_search_html_with_llm(html, query, config)

    except requests.exceptions.RequestException:
        # Re-raise network errors so they can be handled properly upstream
        raise
    except Exception as e:
        print(f"Search with LLM analysis failed: {e}")
        return ""


def _fetch_search_html(query: str, config: SearchConfig) -> str:
    """
    Fetch HTML from DuckDuckGo search.

    Args:
        query: The search query
        config: Search configuration

    Returns:
        HTML content or empty string if failed

    Raises:
        requests.exceptions.RequestException: For network-related errors
        requests.exceptions.HTTPError: For HTTP errors
        requests.exceptions.Timeout: For timeout errors
    """
    try:
        url = f"{config.search_base_url}?q={quote(query)}"
        headers = {
            'User-Agent': config.user_agent
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=config.request_timeout
        )
        response.raise_for_status()
        return response.text

    except (
        requests.exceptions.RequestException,
        requests.exceptions.HTTPError,
        requests.exceptions.Timeout
    ):
        # Re-raise specific network errors
        raise
    except Exception as e:
        print(f"Failed to fetch search HTML: {e}")
        return ""


def _analyze_search_html_with_llm(
    html: str,
    query: str,
    config: SearchConfig
) -> str:
    """
    Use LLM to analyze search HTML and return formatted search results.

    Args:
        html: The HTML content to analyze
        query: The original search query
        config: Search configuration

    Returns:
        LLM-analyzed search results or empty string if failed

    Note:
        This function truncates HTML content to avoid token limits
        based on config.html_truncation_limit.
    """
    try:
        # Truncate HTML to avoid token limits
        truncated_html = (
            html[:config.html_truncation_limit]
            if len(html) > config.html_truncation_limit
            else html
        )

        analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            query=query,
            html=truncated_html
        )

        response = ollama.chat(
            model=config.llm_model,
            messages=[{"role": "user", "content": analysis_prompt}],
            stream=False
        )

        analysis_result = response.message.content.strip()
        print(f"LLM analysis completed, length: {len(analysis_result)}")

        return analysis_result

    except Exception as e:
        print(f"LLM analysis failed: {e}")
        return ""


def _get_fallback_response(query: str) -> str:
    """
    Provide a helpful fallback response when no search results are found.

    Args:
        query: The original search query

    Returns:
        Fallback response message with helpful suggestions
    """
    return f"""No specific search results found for '{query}'.

Here are some suggestions:
• Try rephrasing your query with more specific terms
• Check official documentation for the technology you're asking about
• Search on Stack Overflow or GitHub for code-related questions
• For recent updates, check the project's official blog or release notes

This search tool works best with general topics and may not have the latest information about very recent developments."""


if __name__ == "__main__":
    # Test the simplified tool
    test_queries = [
        "Python 3.12 new features",
        "React 19 changes",
        "who is the current president of the US"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Testing: {query}")
        print('='*50)
        result = web_search(query)
        print(result)