"""
Simple Web Search Tool

A tool that can be called when you don't know something and need to search for current information.
"""

import requests
from typing import List, Dict, Optional
import json


def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web for information when you don't know something.

    Args:
        query: What to search for (e.g., "React 19 new features", "Python 3.12 changes")
        max_results: Maximum number of results to return (default: 3)

    Returns:
        Formatted search results as a string

    Example:
        result = web_search("What's new in TypeScript 5.3")
    """
    try:
        # Add temporal context to help find current information
        enhanced_query = f"as of today: {query}"
        print(f"🔍 Searching for: {enhanced_query}")

        # Try multiple search approaches
        results = []

        # First try DuckDuckGo instant answers
        instant_results = _search_duckduckgo_instant(enhanced_query, max_results)
        if instant_results:
            results.extend(instant_results)

        # If we need more results, try DuckDuckGo HTML search
        if len(results) < max_results:
            html_results = _search_duckduckgo_html(enhanced_query, max_results - len(results))
            if html_results:
                results.extend(html_results)

        # If still no good results, use suggestions
        if not results:
            results = _search_duckduckgo_suggestions(enhanced_query, max_results)

        # Format results
        if not results:
            return _get_fallback_response(query)

        formatted = f"Web search results for '{query}':\n\n"

        print(f"{formatted}")

        for i, result in enumerate(results[:max_results], 1):
            formatted += f"**{i}. {result['title']}**\n"
            if result['snippet']:
                # Limit snippet length
                snippet = result['snippet'][:400]
                if len(result['snippet']) > 400:
                    snippet += "..."
                formatted += f"{snippet}\n"
            if result['url']:
                formatted += f"Source: {result['url']}\n"
            formatted += "\n"

        return formatted.strip()

    except requests.exceptions.RequestException as e:
        return f"Search failed due to network error: {e}"
    except Exception as e:
        return f"Search failed: {e}"


def _search_duckduckgo_instant(query: str, max_results: int) -> List[Dict]:
    """Search using DuckDuckGo instant answer API"""
    url = "https://api.duckduckgo.com/"
    params = {
        'q': query,
        'format': 'json',
        'no_html': '1',
        'skip_disambig': '1'
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = []

    # Get instant answer if available
    if data.get('Answer'):
        results.append({
            'title': 'Direct Answer',
            'snippet': data.get('Answer'),
            'url': data.get('AnswerURL', '')
        })

    # Get definition if available
    if data.get('Definition'):
        results.append({
            'title': f"Definition: {data.get('DefinitionSource', 'Dictionary')}",
            'snippet': data.get('Definition'),
            'url': data.get('DefinitionURL', '')
        })

    # Get abstract if available
    if data.get('Abstract'):
        results.append({
            'title': data.get('Heading', 'Summary')[:100],
            'snippet': data.get('Abstract'),
            'url': data.get('AbstractURL', '')
        })

    # Get related topics
    for topic in data.get('RelatedTopics', [])[:max_results-len(results)]:
        if isinstance(topic, dict) and topic.get('Text'):
            results.append({
                'title': topic.get('Text', '')[:80] + "...",
                'snippet': topic.get('Text', ''),
                'url': topic.get('FirstURL', '')
            })

    return results


def _search_duckduckgo_html(query: str, max_results: int) -> List[Dict]:
    """Search using actual web search by scraping DuckDuckGo HTML and LLM analysis"""
    import re
    from urllib.parse import quote, unquote

    try:
        # Use DuckDuckGo HTML search
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        html = response.text

        # Use LLM to analyze the HTML and extract key information
        llm_analysis = _analyze_search_html_with_llm(html, query, max_results)

        if llm_analysis:
            # Return as a single result that contains the LLM analysis
            return [{"title": "Search Analysis", "snippet": llm_analysis, "url": ""}]

        # Fallback to regex parsing if LLM analysis fails
        results = []

        # Extract results from h2 tags containing links
        h2_pattern = r'<h2[^>]*>\s*<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="[^"]*uddg=([^&"]+)[^"]*"[^>]*>([^<]*)</a>\s*</h2>'
        h2_matches = re.findall(h2_pattern, html, re.DOTALL)

        # First, let's extract the complete result blocks to get better context
        result_blocks = re.findall(r'<div[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*</div>', html, re.DOTALL)

        # Extract snippets with multiple approaches
        snippet_matches = []

        # Try to extract text content from each result block
        for i, (encoded_url, title) in enumerate(h2_matches):
            snippet = ""

            # If we have result blocks, try to extract text from the corresponding block
            if i < len(result_blocks):
                block = result_blocks[i]
                # Remove HTML tags and extract meaningful text
                text_content = re.sub(r'<[^>]+>', ' ', block)
                text_content = re.sub(r'\s+', ' ', text_content).strip()

                # Look for text that comes after the title
                title_clean = re.sub(r'<[^>]+>', '', title).strip()
                if title_clean in text_content:
                    # Get text after the title
                    parts = text_content.split(title_clean, 1)
                    if len(parts) > 1:
                        snippet = parts[1].strip()[:300]  # First 300 chars after title

                # If that didn't work, just take meaningful text from the block
                if not snippet:
                    # Look for sentences with useful content
                    sentences = text_content.split('.')
                    meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20 and
                                          any(word in s.lower() for word in ['score', 'rating', 'percent', '%', 'review'])]
                    if meaningful_sentences:
                        snippet = meaningful_sentences[0][:300]
                    else:
                        snippet = text_content[:200] if text_content else ""

            snippet_matches.append(snippet)

        print(f"Extracted {len(snippet_matches)} enhanced snippets")

        # Combine results
        for i, (encoded_url, title) in enumerate(h2_matches[:max_results]):
            # Decode the URL
            try:
                clean_url = unquote(encoded_url)
                if not clean_url.startswith('http'):
                    continue
            except:
                continue

            # Get corresponding snippet if available
            snippet = snippet_matches[i] if i < len(snippet_matches) else ""

            # Clean up HTML and whitespace
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()

            if clean_title and clean_url:
                results.append({
                    'title': clean_title,
                    'snippet': clean_snippet,
                    'url': clean_url
                })

        # If the above pattern didn't work, try a simpler approach
        if not results:
            # Look for any external links with titles
            general_link_pattern = r'<a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>'
            general_links = re.findall(general_link_pattern, html)

            for url_match, title in general_links[:max_results]:
                if not any(skip in url_match for skip in ['duckduckgo.com', 'javascript:', 'mailto:']):
                    clean_title = re.sub(r'<[^>]+>', '', title).strip()
                    if clean_title and len(clean_title) > 5:  # Filter out very short titles
                        results.append({
                            'title': clean_title,
                            'snippet': f"Search result from {url_match.split('/')[2] if '/' in url_match else url_match}",
                            'url': url_match
                        })

        print(f"Found {len(results)} real search results")
        return results[:max_results]

    except Exception as e:
        print(f"HTML search failed: {e}")
        return []


def _search_duckduckgo_suggestions(query: str, max_results: int) -> List[Dict]:
    """Try to get search suggestions if instant answers don't work"""
    # This is a simplified approach - in practice you might want to use other APIs

    # For now, return some generic technology-related suggestions based on keywords
    results = []

    query_lower = query.lower()

    # Check for common programming topics and provide general guidance
    if any(term in query_lower for term in ['react', 'javascript', 'python', 'typescript', 'node']):
        results.append({
            'title': f"Programming Topic: {query}",
            'snippet': f"This appears to be a programming-related query about {query}. For the most current information, I recommend checking the official documentation or recent community discussions.",
            'url': ''
        })

        # Add common resource suggestions
        if 'react' in query_lower:
            results.append({
                'title': "React Official Documentation",
                'snippet': "The official React documentation is the best source for current React features and best practices.",
                'url': 'https://react.dev'
            })
        elif 'python' in query_lower:
            results.append({
                'title': "Python Official Documentation",
                'snippet': "The official Python documentation contains comprehensive information about Python features and changes.",
                'url': 'https://docs.python.org'
            })

    return results


def _get_fallback_response(query: str) -> str:
    """Provide a helpful fallback response when no search results are found"""
    return f"""No specific search results found for '{query}'.

Here are some suggestions:
• Try rephrasing your query with more specific terms
• Check official documentation for the technology you're asking about
• Search on Stack Overflow or GitHub for code-related questions
• For recent updates, check the project's official blog or release notes

This search tool works best with general topics and may not have the latest information about very recent developments."""


def _analyze_search_html_with_llm(html: str, query: str, max_results: int) -> str:
    """Use LLM to analyze search HTML and return formatted search results"""
    try:
        import ollama

        # Truncate HTML to avoid token limits (keep first 8000 chars which should include results)
        truncated_html = html[:8000] if len(html) > 8000 else html

        analysis_prompt = f"""You are an expert at analyzing search result HTML and extracting specific, factual information.

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
{truncated_html}

Extract and clearly state the specific facts that answer the query:"""

        response = ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": analysis_prompt}],
            stream=False
        )

        analysis_result = response.message.content.strip()
        print(f"LLM analysis completed, length: {len(analysis_result)}")
        print(f"LLM analysis result: {analysis_result}")

        # Format as search result 
        formatted_result = f"Web search results for '{query}':\n\n{analysis_result}"
        return formatted_result

    except Exception as e:
        print(f"LLM analysis failed: {e}")
        return ""


if __name__ == "__main__":
    # Test the tool with multiple queries
    test_queries = [
        "Python",
        "JavaScript async await",
        "React 19 new features"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Testing: {query}")
        print('='*50)
        result = web_search(query)
        print(result)
