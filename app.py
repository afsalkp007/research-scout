"""
ResearchScout - Interactive Research Cockpit (Streamlit)
Visualizes multi-step agent reasoning, tool dispatching, live observations, and grounded reports.
"""

import streamlit as st
from pathlib import Path

from research_scout.models import AgentStep, ResearchReport
from research_scout.agent.orchestrator import ResearchAgent
from research_scout.agent.providers import GeminiProvider, OpenAIProvider, MockProvider
from research_scout.tools.base import ToolRegistry
from research_scout.tools.search import SearchTool
from research_scout.tools.fetch import FetchPageTool
from research_scout.export import export_to_markdown, export_to_json

st.set_page_config(
    page_title="ResearchScout | Autonomous Research Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 ResearchScout: Autonomous AI Research Assistant")
st.markdown(
    "Investigate complex technical and market questions with multi-turn tool calling, "
    "grounded evidence quotes, and academic-grade citations."
)

# Sidebar setup
st.sidebar.header("⚙️ Agent Configuration")
provider_choice = st.sidebar.selectbox(
    "LLM Provider",
    ["Mock (Offline Demo)", "Gemini 2.5 Flash", "OpenAI (GPT-4o-mini)"],
    index=0
)

api_key = ""
if "Gemini" in provider_choice:
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get free key at aistudio.google.com")
elif "OpenAI" in provider_choice:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

max_steps = st.sidebar.slider("Max Research Steps", min_value=2, max_value=6, value=3)

st.sidebar.divider()
st.sidebar.subheader("💡 Preset Research Topics")
preset = st.sidebar.selectbox(
    "Choose a preset topic:",
    [
        "Custom Topic",
        "LLM Multi-Step Agent Evaluation & Reliability",
        "Vector Database Benchmark: Qdrant vs Chroma vs Pinecone",
        "KV Cache Compression & PagedAttention in LLM Serving"
    ]
)

default_query = ""
if preset != "Custom Topic":
    default_query = preset

# Main Query Input
query_input = st.text_input(
    "Enter your research query or hypothesis:",
    value=default_query,
    placeholder="e.g. State of context window compression techniques in 2026"
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    start_btn = st.button("🚀 Start Investigation", type="primary")

if start_btn:
    if not query_input.strip():
        st.warning("Please enter a research topic first.")
    else:
        # Determine provider
        is_mock = "Mock" in provider_choice
        if "Gemini" in provider_choice:
            provider = GeminiProvider(api_key=api_key) if api_key else MockProvider()
        elif "OpenAI" in provider_choice:
            provider = OpenAIProvider(api_key=api_key) if api_key else MockProvider()
        else:
            provider = MockProvider()

        # Tools
        registry = ToolRegistry()
        registry.register(SearchTool(force_mock=is_mock))
        registry.register(FetchPageTool(force_mock=is_mock))

        agent = ResearchAgent(
            provider=provider,
            tool_registry=registry,
            max_steps=max_steps
        )

        st.divider()
        st.subheader("🤖 Live Research Trajectory (ReAct Loop)")
        progress_placeholder = st.container()

        steps_recorded = []

        def step_callback(step: AgentStep):
            steps_recorded.append(step)
            with progress_placeholder:
                with st.expander(
                    f"Step {step.step_number}: {step.tool_call.tool_name if step.tool_call else 'Concluded'}",
                    expanded=True
                ):
                    st.write(f"💭 **Thought:** *{step.thought}*")
                    if step.tool_call:
                        st.code(f"{step.tool_call.tool_name}({step.tool_call.arguments})", language="python")
                        st.caption("Observation:")
                        st.info(step.observation[:400] + ("..." if len(step.observation) > 400 else ""))

        with st.spinner("Agent researching, cross-referencing, and synthesizing..."):
            try:
                report = agent.run(query_input, on_step_callback=step_callback)
                st.session_state["report"] = report
                st.success("Research completed successfully!")
            except Exception as e:
                st.error(f"Research failed: {str(e)}")

# Display Results
if "report" in st.session_state:
    report: ResearchReport = st.session_state["report"]

    st.divider()
    st.header(f"📋 Research Briefing: {report.topic}")

    # KPI Metrics
    k1, k2, k3 = st.columns(3)
    k1.metric("Confidence Rating", report.confidence_rating)
    k2.metric("Verified Sources", len(report.citations))
    k3.metric("Research Steps", len(report.steps_taken))

    tab_summary, tab_findings, tab_sources, tab_export = st.tabs([
        "📌 Executive Summary",
        "💡 Key Findings & Evidence",
        "📚 Verified Citations",
        "📥 Exports & Downloads"
    ])

    with tab_summary:
        st.markdown(report.executive_summary)
        if report.unresolved_questions:
            st.divider()
            st.subheader("⚠️ Unresolved / Open Questions")
            for q in report.unresolved_questions:
                st.markdown(f"- {q}")

    with tab_findings:
        for idx, f in enumerate(report.key_findings, 1):
            cite_badges = " ".join([f"`[{i}]`" for i in f.citation_indices])
            with st.container():
                st.subheader(f"{idx}. {f.claim} {cite_badges}")
                st.write(f.explanation)
                st.divider()

    with tab_sources:
        for c in report.citations:
            st.markdown(f"#### `[{c.index}]` [{c.title}]({c.url})")
            st.caption(f"Canonical URL: {c.url}")
            st.info(f"**Evidence Quote:** \"{c.quote_snippet}\"")
            st.divider()

    with tab_export:
        st.subheader("Download Research Deliverables")
        col_md, col_json = st.columns(2)
        with col_md:
            md_text = export_to_markdown(report)
            st.download_button(
                "📥 Download Markdown Briefing",
                data=md_text,
                file_name="research_briefing.md",
                mime="text/markdown",
                use_container_width=True
            )
            st.caption("Ready to publish in Notion, GitHub Discussions, or Confluence.")
        with col_json:
            json_text = export_to_json(report)
            st.download_button(
                "📥 Download Structured JSON",
                data=json_text,
                file_name="research_report.json",
                mime="application/json",
                use_container_width=True
            )
            st.caption("Full audit trail and citations in JSON.")
