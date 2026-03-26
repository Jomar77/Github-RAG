from __future__ import annotations

from typing import Any, Callable, List

from pydantic_ai import Agent


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant for a course.

Use the search tool to find relevant information from the course materials before answering questions.

If you can find specific information through search, use it to provide accurate answers.

Always include references by citing the filename of the source material you used.  
When citing the reference, replace "faq-main" by the full path to the GitHub repository: "https://github.com/DataTalksClub/faq/blob/main/"
Format: [LINK TITLE](FULL_GITHUB_LINK)

If the search doesn't return relevant results, let the user know and provide general guidance.
"""


SearchFn = Callable[..., List[Any]]


def _call_search_compat(search_fn: SearchFn, query: str) -> List[Any]:
    """Call different search function signatures with a fixed top-k of 5."""
    try:
        return search_fn(query, num_results=5)
    except TypeError:
        try:
            return search_fn(query, limit=5)
        except TypeError:
            return search_fn(query)


def build_text_search_tool(search_fn: SearchFn) -> Callable[[str], List[Any]]:
    """Create a pydantic-ai tool function backed by the provided search function."""

    def text_search(query: str) -> List[Any]:
        """Perform a text-based search on the FAQ index.

        Args:
            query (str): The search query string.

        Returns:
            List[Any]: A list of up to 5 search results returned by the FAQ index.
        """
        return _call_search_compat(search_fn, query)

    return text_search


def build_course_agent(
    search_fn: SearchFn,
    model: str = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Agent:
    """Create the FAQ/course agent using a provided search function."""
    text_search = build_text_search_tool(search_fn)
    return Agent(
        name="faq_agent_V2",
        instructions=system_prompt,
        tools=[text_search],
        model=model,
    )


def result_text(result: Any) -> str:
    """Extract final answer text from different pydantic-ai result versions."""
    if hasattr(result, "data"):
        return str(result.data)
    if hasattr(result, "output"):
        return str(result.output)
    if hasattr(result, "output_text"):
        return str(result.output_text)
    return str(result)


__all__ = ["build_course_agent", "build_text_search_tool", "result_text"]
