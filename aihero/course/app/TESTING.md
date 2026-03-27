# Testing Guide

## Scope

This app currently uses a minimal smoke-test strategy.

Covered:
- Core pure utility behavior in ingestion and logging.
- Basic tool delegation behavior.
- One mocked index orchestration path.

Not covered in this phase:
- Live OpenAI calls.
- Live network integration with GitHub downloads.
- Full CLI loop integration in `main.py`.

## Run Tests

From `aihero/course/app`:

```bash
uv sync --group dev
uv run pytest
```

If you run from a different active virtual environment and see a `VIRTUAL_ENV` mismatch warning, either:

```bash
uv run --active pytest
```

or run the command from this app directory so `uv` targets the local project environment.

## Test Design

- Use deterministic local data for utility tests.
- Use monkeypatch/mocks for external dependencies and side effects.
- Keep tests independent from API keys and internet access.

## Acceptance Criteria

- Test run succeeds locally without network/API credentials.
- Smoke tests validate key app behavior does not regress.
- All changes remain inside the app folder.
