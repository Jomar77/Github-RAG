import search_tools


class DummyIndex:
    def __init__(self):
        self.calls = []

    def search(self, query, num_results):
        self.calls.append((query, num_results))
        return [{"filename": "docs/a.md", "content": "result"}]


def test_search_tool_delegates_query_with_fixed_result_size():
    index = DummyIndex()
    tool = search_tools.SearchTool(index=index)

    result = tool.search("what is RAG?")

    assert result == [{"filename": "docs/a.md", "content": "result"}]
    assert index.calls == [("what is RAG?", 5)]
