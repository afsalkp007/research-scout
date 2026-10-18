"""
Export utilities for ResearchScout reports.
Produces academic-grade Markdown briefings with footnote citations and JSON exports.
"""

import json
from .models import ResearchReport


def export_to_markdown(report: ResearchReport) -> str:
    """Format research report into clean, publication-ready Markdown."""
    lines = []
    lines.append(f"# 🔬 Research Briefing: {report.topic}")
    lines.append("")
    lines.append(f"**Confidence Rating:** `{report.confidence_rating}`  |  "
                 f"**Sources Consulted:** `{len(report.citations)}`  |  "
                 f"**Research Steps:** `{len(report.steps_taken)}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📌 Executive Summary")
    lines.append(report.executive_summary)
    lines.append("")

    if report.key_findings:
        lines.append("## 💡 Key Findings & Evidence")
        for idx, finding in enumerate(report.key_findings, 1):
            citation_tags = " ".join([f"[[{i}]]" for i in finding.citation_indices]) if finding.citation_indices else ""
            lines.append(f"### {idx}. {finding.claim} {citation_tags}")
            lines.append(finding.explanation)
            lines.append("")

    if report.citations:
        lines.append("## 📚 Verified Sources & Citations")
        for cite in report.citations:
            lines.append(f"**[{cite.index}] [{cite.title}]({cite.url})**")
            lines.append(f"> \"{cite.quote_snippet}\"")
            lines.append("")

    if report.unresolved_questions:
        lines.append("## ⚠️ Unresolved Questions & Conflicting Data")
        for q in report.unresolved_questions:
            lines.append(f"- {q}")
        lines.append("")

    if report.steps_taken:
        lines.append("## 🔍 Research Trajectory (Audit Trail)")
        lines.append("| Step | Action / Tool | Query / Target | Key Observation |")
        lines.append("|:---:|:---|:---|:---|")
        for s in report.steps_taken:
            tool_name = s.tool_call.tool_name if s.tool_call else "Conclude"
            args_str = str(s.tool_call.arguments) if s.tool_call else "-"
            # Truncate observation for table
            obs_preview = (s.observation[:60].replace("\n", " ") + "...") if s.observation else "-"
            lines.append(f"| {s.step_number} | `{tool_name}` | `{args_str}` | {obs_preview} |")
        lines.append("")

    return "\n".join(lines)


def export_to_json(report: ResearchReport, indent: int = 2) -> str:
    """Serialize research report to clean JSON."""
    return report.model_dump_json(indent=indent)
