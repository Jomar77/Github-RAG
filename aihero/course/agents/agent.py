from __future__ import annotations

import asyncio
from typing import Any, Callable


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant for a course.



Use the search tool to find relevant information from the course materials before answering questions.

If you can find specific information through search, use it to provide accurate answers.
If the search doesn't return relevant results, let the user know and provide general guidance.
"""


SearchFn = Callable[[str], list[dict[str, Any]]]


class CourseFAQAgent:
    """LLM agent that can call a text-search tool for FAQ lookups."""

    def __init__(
        self,
        search_fn: SearchFn,
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.search_fn = search_fn
        self.model = model
        self.system_prompt = system_prompt

        def text_search(query: str) -> list[dict[str, Any]]:
            """Perform a text-based search on the FAQ index.

            Args:
                query: The search query string.

            Returns:
                A list of up to 5 search results returned by the FAQ index.
            """
            return self.search_fn(query)

        # Import lazily so this module can still be imported when pydantic_ai
        # is not installed yet in the current environment.
        from pydantic_ai import Agent

        self.agent = Agent(
            name="faq_agent",
            instructions=self.system_prompt,
            tools=[text_search],
            model=self.model,
        )

    async def run_async(self, question: str):
        """Run the agent asynchronously and return the full result object."""
        return await self.agent.run(user_prompt=question)

    def run(self, question: str):
        """Run the agent synchronously and return the full result object."""
        try:
            return asyncio.run(self.run_async(question))
        except RuntimeError as error:
            if "asyncio.run() cannot be called" in str(error):
                raise RuntimeError(
                    "A running event loop was detected. Use `await run_async(...)` instead."
                ) from error
            raise

    @staticmethod
    def result_text(result: Any) -> str:
        """Extract final answer text from different pydantic-ai result versions."""
        if hasattr(result, "data"):
            return str(result.data)
        if hasattr(result, "output"):
            return str(result.output)
        if hasattr(result, "output_text"):
            return str(result.output_text)
        return str(result)

    @staticmethod
    def result_messages(result: Any) -> list[Any]:
        """Extract message trace from different pydantic-ai result versions."""
        if hasattr(result, "new_messages"):
            return list(result.new_messages())
        if hasattr(result, "all_messages"):
            return list(result.all_messages())
        return []

    async def answer_async(self, question: str) -> str:
        """Return only the final answer text asynchronously."""
        result = await self.run_async(question)
        return self.result_text(result)

    def answer(self, question: str) -> str:
        """Return only the final answer text synchronously."""
        result = self.run(question)
        return self.result_text(result)


def answer_course_question(question: str, search_fn: SearchFn) -> str:
    """Convenience helper for one-off usage without managing class state."""
    return CourseFAQAgent(search_fn=search_fn).answer(question)


__all__ = ["CourseFAQAgent", "answer_course_question"]
