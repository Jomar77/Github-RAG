from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from tqdm.auto import tqdm


COURSE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = COURSE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
DEFAULT_EVAL_MODEL = "gpt-5-nano"

EVALUATION_PROMPT = """
Use this checklist to evaluate the quality of an AI agent's answer (<ANSWER>) to a user question (<QUESTION>).
We also include the entire log (<LOG>) for analysis.

For each item, check if the condition is met.

Checklist:

- instructions_follow: The agent followed the user's instructions (in <INSTRUCTIONS>)
- instructions_avoid: The agent avoided doing things it was told not to do
- answer_relevant: The response directly addresses the user's question
- answer_clear: The answer is clear and correct
- answer_citations: The response includes proper citations or sources when required
- completeness: The response is complete and covers all key aspects of the request
- tool_call_search: Is the search tool invoked?

Output true/false for each check and provide a short explanation for your judgment.
""".strip()

USER_PROMPT_FORMAT = """
<INSTRUCTIONS>{instructions}</INSTRUCTIONS>
<QUESTION>{question}</QUESTION>
<ANSWER>{answer}</ANSWER>
<LOG>{log}</LOG>
""".strip()

# Backward-compatible aliases used in notebook/tutorial snippets.
evaluation_prompt = EVALUATION_PROMPT
user_prompt_format = USER_PROMPT_FORMAT


def _get_agent_instructions(agent: Any) -> str:
    instructions = getattr(agent, "_instructions", None)
    if instructions is None:
        instructions = getattr(agent, "instructions", "")
    return str(instructions)


def _get_model_info(agent: Any) -> tuple[str, str]:
    model = getattr(agent, "model", None)
    provider = str(getattr(model, "system", "unknown"))
    model_name = str(getattr(model, "model_name", model or "unknown"))
    return provider, model_name


def log_entry(agent, messages, source: str = "user") -> dict[str, Any]:
    tools: list[str] = []

    for toolset in getattr(agent, "toolsets", []):
        tools.extend(toolset.tools.keys())

    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)
    provider, model_name = _get_model_info(agent)

    return {
        "agent_name": getattr(agent, "name", "unknown"),
        "system_prompt": _get_agent_instructions(agent),
        "provider": provider,
        "model": model_name,
        "tools": tools,
        "messages": dict_messages,
        "source": source,
    }


def serializer(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def _last_message_timestamp(messages: list[dict[str, Any]]) -> datetime:
    if messages:
        timestamp = messages[-1].get("timestamp")
        if isinstance(timestamp, str) and timestamp:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return datetime.utcnow()


def log_interaction_to_file(
    agent,
    messages,
    source: str = "user",
    log_dir: Path = LOG_DIR,
) -> Path:
    entry = log_entry(agent, messages, source)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts_obj = _last_message_timestamp(entry["messages"])
    ts_str = ts_obj.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)

    filename = f"{entry['agent_name']}_{ts_str}_{rand_hex}.json"
    filepath = log_dir / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=serializer)

    return filepath


def load_log_file(log_file: str | Path) -> dict[str, Any]:
    path = Path(log_file)
    with path.open("r", encoding="utf-8") as f_in:
        log_data = json.load(f_in)
    log_data["log_file"] = str(path)
    return log_data


def _extract_message_content(message: dict[str, Any]) -> str:
    parts = message.get("parts", [])
    if not isinstance(parts, list):
        return ""

    for part in parts:
        if not isinstance(part, dict):
            continue
        content = part.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_items = [item for item in content if isinstance(item, str)]
            if text_items:
                return "\n".join(text_items)

    return ""


def extract_question_answer(messages: list[dict[str, Any]]) -> tuple[str, str]:
    question = ""
    answer = ""

    if not messages:
        return question, answer

    for message in messages:
        kind = str(message.get("kind", "")).lower()
        content = _extract_message_content(message)
        if not content:
            continue

        if not question and "request" in kind:
            question = content
        if "response" in kind:
            answer = content

    if not question:
        question = _extract_message_content(messages[0])
    if not answer:
        answer = _extract_message_content(messages[-1])

    return question, answer


def simplify_log_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only fields commonly useful for evaluation prompts."""
    simplified: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        item: dict[str, Any] = {
            "kind": message.get("kind"),
            "timestamp": message.get("timestamp"),
            "parts": message.get("parts", []),
        }
        simplified.append(item)
    return simplified


class EvaluationCheck(BaseModel):
    check_name: str
    justification: str
    check_pass: bool


class EvaluationChecklist(BaseModel):
    checklist: list[EvaluationCheck]
    summary: str


def build_eval_agent(model: str = DEFAULT_EVAL_MODEL) -> Agent:
    return Agent(
        name="eval_agent",
        model=model,
        instructions=EVALUATION_PROMPT,
        output_type=EvaluationChecklist,
    )


def build_eval_user_prompt(log_record: dict[str, Any]) -> str:
    instructions = str(log_record.get("system_prompt", ""))
    messages = log_record.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    question, answer = extract_question_answer(messages)
    log_str = json.dumps(messages, ensure_ascii=False)

    return USER_PROMPT_FORMAT.format(
        instructions=instructions,
        question=question,
        answer=answer,
        log=log_str,
    )


async def evaluate_log_record(
    eval_agent: Agent,
    log_record: dict[str, Any],
) -> EvaluationChecklist:
    messages = log_record.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    instructions = str(log_record.get("system_prompt", ""))
    question, answer = extract_question_answer(messages)

    log_simplified = simplify_log_messages(messages)
    log = json.dumps(log_simplified, ensure_ascii=False)

    user_prompt = user_prompt_format.format(
        instructions=instructions,
        question=question,
        answer=answer,
        log=log,
    )

    try:
        result = await eval_agent.run(user_prompt, output_type=EvaluationChecklist)
    except TypeError:
        result = await eval_agent.run(user_prompt)
    return result.output


def evaluate_log_record_sync(
    log_record: dict[str, Any],
    eval_agent: Agent | None = None,
) -> EvaluationChecklist:
    agent = eval_agent or build_eval_agent()
    messages = log_record.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    instructions = str(log_record.get("system_prompt", ""))
    question, answer = extract_question_answer(messages)
    log_simplified = simplify_log_messages(messages)
    log = json.dumps(log_simplified, ensure_ascii=False)

    user_prompt = user_prompt_format.format(
        instructions=instructions,
        question=question,
        answer=answer,
        log=log,
    )

    try:
        result = agent.run_sync(user_prompt, output_type=EvaluationChecklist)
    except TypeError:
        result = agent.run_sync(user_prompt)
    return result.output


def evaluate_log_file_sync(
    log_file: str | Path,
    eval_agent: Agent | None = None,
) -> EvaluationChecklist:
    log_record = load_log_file(log_file)
    return evaluate_log_record_sync(log_record, eval_agent=eval_agent)


def collect_log_records(
    log_dir: Path = LOG_DIR,
    agent_name_contains: str | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    eval_set: list[dict[str, Any]] = []

    # Backward compatibility: older runs wrote logs relative to different
    # working directories, including evals/logs and cwd/logs.
    candidate_dirs: list[Path] = [
        log_dir,
        Path(__file__).resolve().parent / "logs",
        Path.cwd() / "logs",
    ]

    seen_files: set[Path] = set()
    name_filter = agent_name_contains.lower() if agent_name_contains else None
    source_filter = source.lower() if source is not None else None

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue

        for log_file in candidate_dir.glob("*.json"):
            resolved = log_file.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)

            if name_filter and name_filter not in log_file.name.lower():
                continue

            log_record = load_log_file(log_file)
            if source_filter is not None and str(log_record.get("source", "")).lower() != source_filter:
                continue

            eval_set.append(log_record)

    return eval_set


def collect_ai_generated_v2_logs(log_dir: Path = LOG_DIR) -> list[dict[str, Any]]:
    return collect_log_records(
        log_dir=log_dir,
        agent_name_contains="faq_agent_v2",
        source="ai-generated",
    )


async def evaluate_log_set(
    eval_agent: Agent,
    eval_set: list[dict[str, Any]],
    show_progress: bool = True,
) -> list[tuple[dict[str, Any], EvaluationChecklist]]:
    eval_results: list[tuple[dict[str, Any], EvaluationChecklist]] = []

    iterator = tqdm(eval_set) if show_progress else eval_set
    for log_record in iterator:
        eval_result = await evaluate_log_record(eval_agent, log_record)
        eval_results.append((log_record, eval_result))

    return eval_results


def eval_results_to_rows(
    eval_results: list[tuple[dict[str, Any], EvaluationChecklist]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for log_record, eval_result in eval_results:
        messages = log_record.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        question, answer = extract_question_answer(messages)
        log_file = Path(str(log_record.get("log_file", ""))).name

        row: dict[str, Any] = {
            "file": log_file,
            "question": question,
            "answer": answer,
        }

        checks = {c.check_name: c.check_pass for c in eval_result.checklist}
        row.update(checks)
        rows.append(row)

    return rows


def eval_results_to_dataframe(
    eval_results: list[tuple[dict[str, Any], EvaluationChecklist]],
):
    import pandas as pd

    rows = eval_results_to_rows(eval_results)
    return pd.DataFrame(rows)


def eval_dataframe_stats(df_evals):
    return df_evals.mean(numeric_only=True)





__all__ = [
    "DEFAULT_EVAL_MODEL",
    "EVALUATION_PROMPT",
    "EvaluationCheck",
    "EvaluationChecklist",
    "LOG_DIR",
    "USER_PROMPT_FORMAT",
    "build_eval_agent",
    "build_eval_user_prompt",
    "collect_ai_generated_v2_logs",
    "collect_log_records",
    "eval_dataframe_stats",
    "eval_results_to_dataframe",
    "eval_results_to_rows",
    "evaluate_log_set",
    "evaluate_log_file_sync",
    "evaluate_log_record",
    "evaluate_log_record_sync",
    "evaluation_prompt",
    "extract_question_answer",
    "load_log_file",
    "log_entry",
    "log_interaction_to_file",
    "simplify_log_messages",
    "serializer",
    "user_prompt_format",
]
