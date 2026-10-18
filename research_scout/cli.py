"""
CLI interface for ResearchScout.
Renders real-time agent thoughts, tool dispatch actions, observations, and structured briefing.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .models import AgentStep, ResearchReport
from .agent.orchestrator import ResearchAgent
from .agent.providers import GeminiProvider, OpenAIProvider, MockProvider
from .tools.base import ToolRegistry
from .tools.search import SearchTool
from .tools.fetch import FetchPageTool
from .export import export_to_markdown, export_to_json

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


def render_step_live(step: AgentStep):
    """Callback rendering each step of agent reasoning and action."""
    if HAS_RICH and console:
        console.print(f"[bold cyan]─── Step {step.step_number} ───[/bold cyan]")
        console.print(f"[bold yellow]💭 Thought:[/bold yellow] [italic]{step.thought}[/italic]")
        if step.tool_call:
            args_formatted = ", ".join(f"{k}='{v}'" for k, v in step.tool_call.arguments.items())
            console.print(f"[bold green]⚡ Action:[/bold green] [bold white]{step.tool_call.tool_name}[/bold white]({args_formatted})")
            if step.observation:
                # Preview observation snippet
                preview = step.observation[:200].replace("\n", " ").strip()
                console.print(f"[bold blue]👁️ Observation:[/bold blue] [dim]{preview}...[/dim]\n")
        else:
            console.print(f"[bold green]✓ Status:[/bold green] [bold magenta]Research Concluded[/bold magenta]\n")
    else:
        print(f"--- Step {step.step_number} ---")
        print(f"Thought: {step.thought}")
        if step.tool_call:
            print(f"Action: {step.tool_call.tool_name}({step.tool_call.arguments})")
            if step.observation:
                print(f"Observation preview: {step.observation[:120]}...\n")
        else:
            print("Research Concluded.\n")


def render_final_report(report: ResearchReport):
    """Render structured report in terminal."""
    if HAS_RICH and console:
        console.print()
        console.print(Panel(
            Text(report.executive_summary, style="white"),
            title=f"[bold green]🔬 Research Briefing: {report.topic}[/bold green]",
            subtitle=f"Confidence: {report.confidence_rating} | Sources: {len(report.citations)}",
            border_style="green"
        ))
        console.print()

        # Key findings
        if report.key_findings:
            table = Table(title="💡 Key Findings & Evidence", show_header=True, header_style="bold magenta")
            table.add_column("#", width=4, justify="center")
            table.add_column("Claim & Technical Explanation", min_width=45)
            table.add_column("Citations", justify="center", width=12)

            for idx, finding in enumerate(report.key_findings, 1):
                cite_str = ", ".join(f"[{i}]" for i in finding.citation_indices) or "N/A"
                text = f"[bold]{finding.claim}[/bold]\n[dim]{finding.explanation}[/dim]"
                table.add_row(str(idx), text, f"[cyan]{cite_str}[/cyan]")
            console.print(table)
            console.print()

        # Sources & Footnotes
        if report.citations:
            cite_table = Table(title="📚 Verified Citations", show_header=True, header_style="bold blue")
            cite_table.add_column("Ref", width=6, justify="center")
            cite_table.add_column("Source & Evidence Quote", min_width=50)

            for c in report.citations:
                content = f"[bold underline cyan]{c.title}[/bold underline cyan]\n[dim]{c.url}[/dim]\n[italic]\"{c.quote_snippet}\"[/italic]"
                cite_table.add_row(f"[{c.index}]", content)
            console.print(cite_table)
            console.print()

        # Unresolved
        if report.unresolved_questions:
            for q in report.unresolved_questions:
                console.print(Panel(
                    f"[yellow]{q}[/yellow]",
                    title="[bold yellow]⚠️ Open / Unresolved Question[/bold yellow]",
                    border_style="yellow"
                ))
            console.print()
    else:
        print("=" * 70)
        print(f"REPORT: {report.topic}")
        print(f"Summary: {report.executive_summary}\n")
        print("FINDINGS:")
        for idx, f in enumerate(report.key_findings, 1):
            print(f"{idx}. {f.claim} (Citations: {f.citation_indices})")
            print(f"   {f.explanation}")
        print("\nCITATIONS:")
        for c in report.citations:
            print(f"[{c.index}] {c.title} ({c.url})")
            print(f"    Evidence: \"{c.quote_snippet}\"")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="ResearchScout: Autonomous research assistant with function calling & grounded citations."
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Research topic or question (e.g. 'LLM agent evaluation benchmarks 2026')"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in deterministic offline mock mode (no API keys required)"
    )
    parser.add_argument(
        "-p", "--provider",
        choices=["gemini", "openai", "mock"],
        default=None,
        help="LLM provider backend (default: env or gemini)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Maximum tool execution steps before forcing synthesis (default: 4)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated markdown briefing and JSON export"
    )

    args = parser.parse_args()

    topic = args.query or "LLM multi-step agent evaluation benchmarks"

    # Select provider
    force_mock = args.mock or (args.provider == "mock")
    if force_mock:
        provider = MockProvider()
        tools_mock = True
    elif args.provider == "gemini":
        provider = GeminiProvider()
        tools_mock = False
    elif args.provider == "openai":
        provider = OpenAIProvider()
        tools_mock = False
    else:
        # Auto detect or mock
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if gemini_key:
            provider = GeminiProvider(api_key=gemini_key)
            tools_mock = False
        elif openai_key:
            provider = OpenAIProvider(api_key=openai_key)
            tools_mock = False
        else:
            provider = MockProvider()
            tools_mock = True

    # Build tools registry
    registry = ToolRegistry()
    registry.register(SearchTool(force_mock=tools_mock))
    registry.register(FetchPageTool(force_mock=tools_mock))

    agent = ResearchAgent(
        provider=provider,
        tool_registry=registry,
        max_steps=args.max_steps
    )

    if HAS_RICH and console:
        console.print(Panel(
            f"[bold white]Objective:[/bold white] [cyan]{topic}[/cyan]\n"
            f"[bold white]Backend Provider:[/bold white] [magenta]{provider.__class__.__name__}[/magenta]\n"
            f"[bold white]Tools Registered:[/bold white] [green]{', '.join(t.name for t in registry.list_tools())}[/green]",
            title="[bold green]🚀 ResearchScout Engine Initialized[/bold green]",
            border_style="green"
        ))
        console.print()

    # Run agent
    report = agent.run(topic, on_step_callback=render_step_live)

    # Render report
    render_final_report(report)

    # Export if requested
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md_file = out_dir / "research_briefing.md"
        json_file = out_dir / "research_report.json"
        
        md_file.write_text(export_to_markdown(report), encoding="utf-8")
        json_file.write_text(export_to_json(report), encoding="utf-8")
        
        print(f"✓ Saved Markdown Briefing: {md_file}")
        print(f"✓ Saved JSON Report: {json_file}")


if __name__ == "__main__":
    main()
