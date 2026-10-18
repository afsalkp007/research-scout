"""
LLM Provider implementations for ResearchScout.
Includes Gemini, OpenAI, and a Mock provider for offline testing and fast evaluations.
"""

from abc import ABC, abstractmethod
import json
import os
import re
import urllib.request
import urllib.error
from typing import List, Optional, Tuple, Dict, Any

from ..models import AgentStep, ToolCall, ResearchReport, ResearchFinding, Citation
from .prompts import AGENT_SYSTEM_PROMPT, SYNTHESIS_PROMPT_TEMPLATE


def clean_json_response(text: str) -> str:
    """Clean markdown fences from LLM responses."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


class BaseLLMProvider(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def plan_next_step(
        self,
        topic: str,
        steps: List[AgentStep],
        tools_schema: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[ToolCall]]:
        """
        Return (thought, tool_call).
        If tool_call is None, the agent has finished research and is ready to synthesize.
        """
        pass

    @abstractmethod
    def synthesize_report(self, topic: str, steps: List[AgentStep]) -> ResearchReport:
        """Synthesize accumulated findings into a final ResearchReport."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Gemini 2.5 Flash implementation via REST API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", model)
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiProvider.")

    def _call_gemini(self, prompt: str, system_instruction: str, json_mode: bool = False) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {"temperature": 0.2}
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]

    def plan_next_step(
        self,
        topic: str,
        steps: List[AgentStep],
        tools_schema: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[ToolCall]]:
        history_text = "\n\n".join([
            f"Step {s.step_number}:\nThought: {s.thought}\nAction: {s.tool_call.model_dump_json() if s.tool_call else 'None'}\nObservation: {s.observation}"
            for s in steps
        ]) if steps else "No previous steps taken yet."

        tools_desc = json.dumps(tools_schema, indent=2)

        prompt = f"""Research Objective: "{topic}"

Available Tools:
{tools_desc}

Previous Research Steps:
{history_text}

Analyze what information has been gathered.
If you have gathered at least 2-3 reliable sources with sufficient evidence to answer the research objective, decide to FINISH.
Otherwise, choose the next tool call to make.

Return JSON in this format:
{{
  "thought": "Your internal reasoning on what to investigate next or why research is complete",
  "action": "call_tool" or "FINISH",
  "tool_name": "web_search" or "fetch_webpage" (or null if FINISH),
  "arguments": {{ ... }} (or null if FINISH)
}}
"""
        raw = self._call_gemini(prompt, AGENT_SYSTEM_PROMPT, json_mode=True)
        data = json.loads(clean_json_response(raw))
        thought = data.get("thought", "Analyzing findings.")
        action = data.get("action", "FINISH")

        if action == "FINISH" or not data.get("tool_name"):
            return thought, None

        tool_call = ToolCall(
            tool_name=data["tool_name"],
            arguments=data.get("arguments", {}),
            thought=thought
        )
        return thought, tool_call

    def synthesize_report(self, topic: str, steps: List[AgentStep]) -> ResearchReport:
        obs_history = "\n\n".join([
            f"--- Step {s.step_number} ({s.tool_call.tool_name if s.tool_call else 'Note'}) ---\n"
            f"Observation:\n{s.observation}"
            for s in steps if s.observation
        ])
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(topic=topic, observation_history=obs_history)
        raw = self._call_gemini(prompt, AGENT_SYSTEM_PROMPT, json_mode=True)
        data = json.loads(clean_json_response(raw))
        report = ResearchReport.model_validate(data)
        report.steps_taken = steps
        return report


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API implementation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", model)
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIProvider.")

    def _call_openai(self, prompt: str, system_instruction: str, json_mode: bool = False) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def plan_next_step(
        self,
        topic: str,
        steps: List[AgentStep],
        tools_schema: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[ToolCall]]:
        history_text = "\n\n".join([
            f"Step {s.step_number}:\nThought: {s.thought}\nAction: {s.tool_call.model_dump_json() if s.tool_call else 'None'}\nObservation: {s.observation}"
            for s in steps
        ]) if steps else "No previous steps taken yet."

        prompt = f"""Research Objective: "{topic}"
Available Tools: {json.dumps(tools_schema, indent=2)}
Previous Steps: {history_text}

Return JSON with format:
{{
  "thought": "Your reasoning",
  "action": "call_tool" or "FINISH",
  "tool_name": "web_search" or "fetch_webpage" or null,
  "arguments": {{ ... }} or null
}}
"""
        raw = self._call_openai(prompt, AGENT_SYSTEM_PROMPT, json_mode=True)
        data = json.loads(clean_json_response(raw))
        thought = data.get("thought", "")
        if data.get("action") == "FINISH" or not data.get("tool_name"):
            return thought, None
        return thought, ToolCall(
            tool_name=data["tool_name"],
            arguments=data.get("arguments", {}),
            thought=thought
        )

    def synthesize_report(self, topic: str, steps: List[AgentStep]) -> ResearchReport:
        obs_history = "\n\n".join([
            f"Step {s.step_number}:\nObservation:\n{s.observation}"
            for s in steps if s.observation
        ])
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(topic=topic, observation_history=obs_history)
        raw = self._call_openai(prompt, AGENT_SYSTEM_PROMPT, json_mode=True)
        data = json.loads(clean_json_response(raw))
        report = ResearchReport.model_validate(data)
        report.steps_taken = steps
        return report


class MockProvider(BaseLLMProvider):
    """
    Deterministic Mock Provider for instant CLI demos, CI/CD testing,
    and offline evaluation without an API key.
    """

    def plan_next_step(
        self,
        topic: str,
        steps: List[AgentStep],
        tools_schema: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[ToolCall]]:
        step_count = len(steps)

        # Step 0: Execute initial web search
        if step_count == 0:
            thought = f"I will perform an initial web search to discover primary benchmark data and sources on: '{topic}'."
            return thought, ToolCall(
                tool_name="web_search",
                arguments={"query": f"{topic} benchmarks evaluation 2026"},
                thought=thought
            )

        # Step 1: Follow up by reading a relevant webpage URL
        elif step_count == 1:
            thought = "The initial search returned high-relevance sources. I will fetch and read the top technical document for deep empirical data."
            # Extract URL if available in first observation
            obs = steps[0].observation or ""
            target_url = "https://crfm.stanford.edu/helm/latest/"
            match = re.search(r"https?://[^\s\)]+", obs)
            if match:
                target_url = match.group(0).rstrip(",")

            return thought, ToolCall(
                tool_name="fetch_webpage",
                arguments={"url": target_url},
                thought=thought
            )

        # Step 2: Conclude research and finalize
        else:
            thought = "Sufficient verified technical data and empirical quotes have been gathered. Ready to synthesize final report."
            return thought, None

    def synthesize_report(self, topic: str, steps: List[AgentStep]) -> ResearchReport:
        t_lower = topic.lower()

        if "agent" in t_lower or "eval" in t_lower:
            return ResearchReport(
                topic=topic,
                executive_summary=(
                    "Evaluating autonomous multi-step agentic workflows requires a departure from single-metric static benchmarks [1]. "
                    "Modern evaluation frameworks combine trajectory testing (step-level tool precision and error recovery) "
                    "with holistic multi-axis scoring across accuracy, robustness, and cost per query [1][2]. "
                    "Unit-testing tools in sandboxed environments enables reproducible CI/CD pipelines before deployment [2]."
                ),
                key_findings=[
                    ResearchFinding(
                        claim="Single-metric benchmarks fail to capture multi-step agent failure modes",
                        explanation="Agents fail predominantly in tool dispatch parameter drift and loop termination, which static accuracy scores overlook.",
                        citation_indices=[1]
                    ),
                    ResearchFinding(
                        claim="Step-level trajectory evaluation is standard in production engineering",
                        explanation="Industry frameworks like HELM and DeepEval advocate unit testing individual tool calling steps in deterministic sandbox fixtures.",
                        citation_indices=[1, 2]
                    )
                ],
                citations=[
                    Citation(
                        index=1,
                        title="HELM: Holistic Evaluation of Language Models",
                        url="https://crfm.stanford.edu/helm/latest/",
                        quote_snippet="The benchmark emphasizes that single-metric evaluations fail to capture safety and robustness degradation across complex multi-step scenarios."
                    ),
                    Citation(
                        index=2,
                        title="Evaluating Agentic Workflows: Reliability, Precision, and Evals",
                        url="https://www.deeplearning.ai/the-batch/evaluating-agentic-workflows/",
                        quote_snippet="Evaluating multi-step agents requires step-level tool precision metrics, trajectory evaluation, and unit-testing tool calling loops."
                    )
                ],
                confidence_rating="High",
                unresolved_questions=[
                    "Standardized automated metrics for grading natural language agentic apologies vs corrective actions.",
                    "Long-term benchmark drift under continuous web scraping updates."
                ],
                steps_taken=steps
            )

        # Generic report for any topic
        return ResearchReport(
            topic=topic,
            executive_summary=(
                f"Comprehensive technical investigation into '{topic}' reveals significant advancements in efficiency and architectural modularity [1]. "
                "Empirical evaluations emphasize the importance of strict schema validation, deterministic fallback mechanisms, "
                "and granular latency optimization across production deployments [1][2]."
            ),
            key_findings=[
                ResearchFinding(
                    claim=f"Modern architectures for {topic} prioritize reliability over raw model scale",
                    explanation="Production systems combine fast lightweight inference with strict function-calling boundaries rather than monolithic prompting.",
                    citation_indices=[1]
                ),
                ResearchFinding(
                    claim="Standardized evaluation suites are required to prevent regression",
                    explanation="Teams that introduce automated trajectory testing catch edge-case hallucinations before production rollout.",
                    citation_indices=[2]
                )
            ],
            citations=[
                Citation(
                    index=1,
                    title=f"Technical Overview of {topic.title()}",
                    url=f"https://en.wikipedia.org/wiki/{urllib.parse.quote(topic)}",
                    quote_snippet=f"Comprehensive documentation and analysis covering {topic}, including modern architectural patterns, industry benchmarks, and implementation best practices."
                ),
                Citation(
                    index=2,
                    title="Recent Advances in AI Engineering & Systems Design",
                    url="https://arxiv.org/abs/cs/recent",
                    quote_snippet="Empirical case studies analyzing reliability, cost trade-offs, and scalability in production systems."
                )
            ],
            confidence_rating="High",
            unresolved_questions=[
                f"Longitudinal impact of continuous updates on {topic} benchmarks.",
                "Cross-cloud deployment performance variance."
            ],
            steps_taken=steps
        )
