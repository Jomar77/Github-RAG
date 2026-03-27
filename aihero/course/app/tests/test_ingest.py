import pytest

import ingest


def test_sliding_window_returns_expected_windows():
    data = "abcdefghij"

    result = ingest.sliding_window(data, size=4, step=3)

    assert result == [
        {"start": 0, "content": "abcd"},
        {"start": 3, "content": "defg"},
        {"start": 6, "content": "ghij"},
        {"start": 9, "content": "j"},
    ]


def test_sliding_window_rejects_non_positive_values():
    with pytest.raises(ValueError):
        ingest.sliding_window("abc", size=0, step=1)

    with pytest.raises(ValueError):
        ingest.sliding_window("abc", size=1, step=0)


def test_chunk_documents_keeps_metadata_and_chunks_content():
    docs = [
        {
            "filename": "docs/a.md",
            "title": "A",
            "content": "abcdefgh",
        }
    ]

    chunks = ingest.chunk_documents(docs, size=4, step=4)

    assert chunks == [
        {"start": 0, "content": "abcd", "filename": "docs/a.md", "title": "A"},
        {"start": 4, "content": "efgh", "filename": "docs/a.md", "title": "A"},
    ]


def test_index_data_filters_and_fits_docs(monkeypatch):
    source_docs = [
        {"filename": "keep.md", "content": "keep this"},
        {"filename": "drop.md", "content": "drop this"},
    ]

    class FakeIndex:
        def __init__(self, text_fields):
            self.text_fields = text_fields
            self.fitted_docs = None

        def fit(self, docs):
            self.fitted_docs = docs

    monkeypatch.setattr(ingest, "read_repo_data", lambda owner, name: source_docs)
    monkeypatch.setattr(ingest, "Index", FakeIndex)

    index = ingest.index_data(
        "owner",
        "repo",
        filter=lambda doc: doc["filename"] == "keep.md",
        chunk=False,
    )

    assert isinstance(index, FakeIndex)
    assert index.text_fields == ["content", "filename"]
    assert index.fitted_docs == [{"filename": "keep.md", "content": "keep this"}]
