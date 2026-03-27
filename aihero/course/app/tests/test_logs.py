import json
from datetime import datetime

import logs


class DummyToolset:
    def __init__(self):
        self.tools = {"search": object()}


class DummyModel:
    system = "openai"
    model_name = "gpt-4o-mini"


class DummyAgent:
    name = "gh_agent"
    _instructions = "Use search first"
    model = DummyModel()
    toolsets = [DummyToolset()]


def test_serializer_formats_datetime():
    dt = datetime(2026, 3, 27, 10, 30, 0)
    assert logs.serializer(dt) == "2026-03-27T10:30:00"


def test_log_entry_contains_expected_fields(monkeypatch):
    class DummyAdapter:
        @staticmethod
        def dump_python(messages):
            return messages

    monkeypatch.setattr(logs, "ModelMessagesTypeAdapter", DummyAdapter)

    messages = [{"timestamp": datetime(2026, 3, 27, 10, 0, 0), "parts": []}]
    entry = logs.log_entry(DummyAgent(), messages)

    assert entry["agent_name"] == "gh_agent"
    assert entry["provider"] == "openai"
    assert entry["model"] == "gpt-4o-mini"
    assert entry["tools"] == ["search"]
    assert entry["messages"] == messages


def test_log_interaction_to_file_writes_json(monkeypatch, tmp_path):
    monkeypatch.setattr(logs, "LOG_DIR", tmp_path)

    class DummyAdapter:
        @staticmethod
        def dump_python(messages):
            return messages

    monkeypatch.setattr(logs, "ModelMessagesTypeAdapter", DummyAdapter)

    messages = [{"timestamp": datetime(2026, 3, 27, 10, 0, 0), "parts": []}]
    filepath = logs.log_interaction_to_file(DummyAgent(), messages)

    assert filepath.exists()
    assert filepath.name.startswith("gh_agent_20260327_100000_")
    assert filepath.suffix == ".json"

    payload = json.loads(filepath.read_text(encoding="utf-8"))
    assert payload["agent_name"] == "gh_agent"
    assert payload["source"] == "user"
