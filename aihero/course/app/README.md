# GitHub FAQ RAG App

This app builds a lightweight retrieval-augmented assistant over a GitHub repository's markdown documentation.

## What It Does

1. Downloads repository docs from GitHub as a ZIP file.
2. Parses markdown frontmatter/content.
3. Builds a local search index.
4. Runs a pydantic-ai agent that uses a search tool before answering.
5. Logs each interaction to JSON.

## Components

- `ingest.py`: Reads repository docs, chunks text, and builds the search index.
- `search_tools.py`: Exposes index search as an agent tool.
- `search_agent.py`: Configures the agent system prompt and tool wiring.
- `logs.py`: Serializes and writes interaction logs.
- `streamlit_app.py`: Streamlit web chat entrypoint.
- `main.py`: CLI entrypoint and orchestration loop.

## Prerequisites

- Python 3.11+
- `uv` (recommended) or another Python environment manager

## Setup

From this directory (`aihero/course/app`):

```bash
uv sync
```

Install test dependencies:

```bash
uv sync --group dev
```

## Run the App

CLI:

```bash
uv run python main.py
```

Then ask questions in the prompt. Type `stop` to exit.

Streamlit web app:

```bash
uv run streamlit run streamlit_app.py
```

## Logs

- Default log folder: `logs/` (relative to this app directory)
- Override with environment variable: `LOGS_DIRECTORY`

## Testing

See `TESTING.md` for smoke-test coverage, scope, and commands.

## CI/CD

See `CI_CD.md` for an app-local CI blueprint and root workflow handoff steps.

## Current Limitations

- `main.py` hardcodes repository target values.
- Unit tests avoid live API/network calls.
- Full GitHub Actions workflows must live in repository root and are intentionally out of scope for app-only edits.
