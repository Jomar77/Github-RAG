from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from tqdm.auto import tqdm

# Load .env before creating model clients.
load_dotenv()

try:
    from ..agents.agent import build_course_agent, result_text
    from ..data_loading import read_repo_data
    from ..evals.evals import log_interaction_to_file
    from ..search import FAQHybridSearch, filter_documents_by_filename
except ImportError:
    # Allow running directly: `python data_generator_agent.py`
    course_root = Path(__file__).resolve().parents[1]
    if str(course_root) not in sys.path:
        sys.path.insert(0, str(course_root))

    from agents.agent import build_course_agent, result_text
    from data_loading import read_repo_data
    from evals import log_interaction_to_file
    from search import FAQHybridSearch, filter_documents_by_filename


DEFAULT_TRACK = "data-engineering"
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_ROUNDS = 1

question_generation_prompt = """
You are helping to create test questions for an AI agent that answers questions about a data engineering course.

Based on the provided FAQ content, generate realistic questions that students might ask.

The questions should:

- Be natural and varied in style
- Range from simple to complex
- Include both specific technical questions and general course questions

Generate one question for each record.
""".strip()


class QuestionsList(BaseModel):
    questions: list[str]


def build_question_generator(model: str = "gpt-4o-mini") -> Agent:
    return Agent(
        name="question_generator",
        instructions=question_generation_prompt,
        model=model,
        output_type=QuestionsList,
    )


def sample_prompt_docs(
    documents: list[dict[str, Any]],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int | None = None,
    content_key: str = "content",
) -> list[str]:
    if not documents:
        raise ValueError("Cannot sample from an empty documents list")

    k = min(sample_size, len(documents))
    sampler = random.Random(seed)
    sampled_docs = sampler.sample(documents, k)
    return [str(doc.get(content_key, "")) for doc in sampled_docs]


async def generate_questions(
    question_generator: Agent,
    prompt_docs: list[str],
) -> list[str]:
    prompt = json.dumps(prompt_docs, ensure_ascii=False)
    result = await question_generator.run(prompt)

    output = result.output
    if isinstance(output, QuestionsList):
        return output.questions

    # Compatibility path for other output wrappers.
    if hasattr(output, "questions"):
        return list(output.questions)
    return []


def _result_messages(result: Any) -> list[Any]:
    if hasattr(result, "new_messages"):
        return list(result.new_messages())
    if hasattr(result, "all_messages"):
        return list(result.all_messages())
    return []


async def generate_and_log_ai_questions(
    course_agent: Agent,
    source_documents: list[dict[str, Any]],
    question_generator: Agent | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    rounds: int = DEFAULT_ROUNDS,
    source: str = "ai-generated",
    seed: int | None = None,
    verbose: bool = True,
) -> list[str]:
    generator = question_generator or build_question_generator()
    all_questions: list[str] = []

    for round_idx in range(rounds):
        round_seed = None if seed is None else seed + round_idx
        prompt_docs = sample_prompt_docs(
            source_documents,
            sample_size=sample_size,
            seed=round_seed,
        )

        questions = await generate_questions(generator, prompt_docs)
        all_questions.extend(questions)

        iterator = tqdm(questions, desc=f"Round {round_idx + 1}/{rounds}", leave=False)
        for question in iterator:
            result = await course_agent.run(question)
            messages = _result_messages(result)
            log_interaction_to_file(course_agent, messages, source=source)

            if verbose:
                print(question)
                print(result_text(result))
                print()

    return all_questions


async def main() -> None:
    dtc_faq = read_repo_data("DataTalksClub", "faq")
    dtc_faq_track = filter_documents_by_filename(dtc_faq, DEFAULT_TRACK)

    faq_hybrid_search = FAQHybridSearch.from_documents(
        dtc_faq,
        track=DEFAULT_TRACK,
        show_progress=True,
    )

    course_agent = build_course_agent(search_fn=faq_hybrid_search.lexical)

    questions = await generate_and_log_ai_questions(
        course_agent=course_agent,
        source_documents=dtc_faq_track,
        sample_size=DEFAULT_SAMPLE_SIZE,
        rounds=DEFAULT_ROUNDS,
        source="ai-generated",
        verbose=True,
    )
    print(f"Generated and logged {len(questions)} questions.")


if __name__ == "__main__":
    asyncio.run(main())
    


__all__ = [
    
    "QuestionsList",
    "build_question_generator",
    "generate_and_log_ai_questions",
    "generate_questions",
    "main",
    "question_generation_prompt",
    "sample_prompt_docs",
]
