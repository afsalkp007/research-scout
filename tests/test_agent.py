from research_scout.agent.orchestrator import ResearchAgent
from research_scout.agent.providers import MockProvider
from research_scout.tools.base import ToolRegistry
from research_scout.tools.search import SearchTool
from research_scout.tools.fetch import FetchPageTool
from research_scout.models import ResearchReport, AgentStep


def test_research_agent_react_cycle():
    registry = ToolRegistry()
    registry.register(SearchTool(force_mock=True))
    registry.register(FetchPageTool(force_mock=True))

    agent = ResearchAgent(
        provider=MockProvider(),
        tool_registry=registry,
        max_steps=3
    )

    steps_logged = []
    def on_step(s: AgentStep):
        steps_logged.append(s)

    report = agent.run("LLM agent evaluation benchmarks", on_step_callback=on_step)

    assert isinstance(report, ResearchReport)
    assert len(steps_logged) >= 2
    assert len(report.citations) >= 1
    assert len(report.key_findings) >= 1
    assert report.confidence_rating in ["High", "Medium", "Low"]
    assert report.citations[0].index == 1
    assert report.citations[0].quote_snippet != ""


def test_research_agent_max_step_termination():
    registry = ToolRegistry()
    registry.register(SearchTool(force_mock=True))

    # Test that max_steps=1 terminates promptly
    agent = ResearchAgent(
        provider=MockProvider(),
        tool_registry=registry,
        max_steps=1
    )

    report = agent.run("KV Cache compression")
    assert isinstance(report, ResearchReport)
