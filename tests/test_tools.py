from research_scout.tools.base import BaseTool, ToolRegistry
from research_scout.tools.search import SearchTool
from research_scout.tools.fetch import FetchPageTool


def test_tool_registry_registration_and_dispatch():
    registry = ToolRegistry()
    search = SearchTool(force_mock=True)
    registry.register(search)

    assert registry.get_tool("web_search") is not None
    assert len(registry.list_tools()) == 1

    # Schema generation
    schemas = registry.get_function_declarations()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "web_search"
    assert "parameters" in schemas[0]

    # Execution dispatch
    result = registry.dispatch("web_search", {"query": "agent evaluation"})
    assert "HELM" in result or "Evaluation" in result

    # Non-existent tool error boundary
    missing_result = registry.dispatch("non_existent_tool", {})
    assert "Error: Tool 'non_existent_tool' not found" in missing_result


def test_fetch_tool_mock():
    fetch = FetchPageTool(force_mock=True)
    assert fetch.name == "fetch_webpage"

    result = fetch.execute(url="https://crfm.stanford.edu/helm/latest/")
    assert "HELM" in result
    assert "toxicity" in result or "accuracy" in result


def test_search_empty_query():
    search = SearchTool(force_mock=True)
    res = search.execute(query="")
    assert "Error: Search query cannot be empty" in res
