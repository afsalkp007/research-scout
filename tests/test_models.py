import pytest
from pydantic import ValidationError
from research_scout.models import (
    SearchResult,
    Citation,
    ResearchFinding,
    ResearchReport,
    AgentStep,
    ToolCall
)


def test_citation_model_validation():
    c = Citation(
        index=1,
        title="HELM Benchmark",
        url="https://crfm.stanford.edu/helm/latest/",
        quote_snippet="Standardizing multi-metric evaluations"
    )
    assert c.index == 1
    assert c.title == "HELM Benchmark"

    # Negative or zero index should raise ValidationError
    with pytest.raises(ValidationError):
        Citation(
            index=0,
            title="Invalid",
            url="https://example.com",
            quote_snippet="Quote"
        )


def test_research_report_serialization():
    report = ResearchReport(
        topic="Vector DB Benchmarks",
        executive_summary="Vector databases trade memory for QPS.",
        key_findings=[
            ResearchFinding(
                claim="Qdrant uses HNSW with scalar quantization",
                explanation="Reduces memory 75% while keeping high recall",
                citation_indices=[1]
            )
        ],
        citations=[
            Citation(
                index=1,
                title="Qdrant Benchmarks",
                url="https://qdrant.tech",
                quote_snippet="Reduces memory footprint by 75%"
            )
        ],
        confidence_rating="High",
        unresolved_questions=[]
    )
    data = report.model_dump()
    assert data["topic"] == "Vector DB Benchmarks"
    assert len(data["key_findings"]) == 1
    assert data["citations"][0]["index"] == 1


def test_agent_step_creation():
    step = AgentStep(
        step_number=1,
        thought="Need to search for benchmarks",
        tool_call=ToolCall(
            tool_name="web_search",
            arguments={"query": "agent benchmarks"},
            thought="Search benchmarks"
        ),
        observation="Found 3 results"
    )
    assert step.step_number == 1
    assert step.tool_call.tool_name == "web_search"
    assert step.observation == "Found 3 results"
