# CI/CD Blueprint (App Scope)

This file defines the desired CI behavior for this app module while keeping edits limited to the app folder.

## Goal

Run smoke tests automatically on every push and pull request.

## Why This Is a Blueprint

GitHub Actions workflow files must be created under repository root:

- `.github/workflows/*.yml`

That path is outside the app-only editing scope, so this document provides the handoff content.

## Recommended Pipeline Steps

1. Checkout repository.
2. Install Python 3.11.
3. Install `uv`.
4. Change directory to `aihero/course/app`.
5. Install dependencies including dev group.
6. Run smoke tests.

## Suggested Workflow Snippet

```yaml
name: App Smoke Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  app-tests:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        working-directory: aihero/course/app
        run: uv sync --group dev

      - name: Run smoke tests
        working-directory: aihero/course/app
        run: uv run pytest
```

## Handoff Step (Out of Scope Here)

When root-level edits are allowed, copy the snippet into:

- `.github/workflows/app-tests.yml`
