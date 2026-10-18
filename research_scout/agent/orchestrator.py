"""
Agent orchestrator managing the ReAct loop (Thought -> Action -> Observation).
Implements execution boundaries, step callbacks, and final report synthesis.
"""

from typing import List, Optional, Callable
from ..models import AgentStep, ToolCall, ResearchReport
from ..tools.base import ToolRegistry
from .providers import BaseLLMProvider, GeminiProvider, OpenAIProvider, MockProvider
import os


class ResearchAgent:
    """
    Autonomous multi-step research agent.
    Iteratively queries tools, gathers observations, and synthesizes grounded briefings.
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = 4
    ):
        self.provider = provider or self._auto_detect_provider()
        self.registry = tool_registry or self._default_tool_registry()
        self.max_steps = max_steps

    def _auto_detect_provider(self) -> BaseLLMProvider:
        """Auto-detect configured provider or fall back to Mock."""
        selected = os.getenv("LLM_PROVIDER", "").lower()
        if selected == "gemini" and os.getenv("GEMINI_API_KEY"):
            return GeminiProvider()
        elif selected == "openai" and os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider()
        elif os.getenv("GEMINI_API_KEY"):
            return GeminiProvider()
        elif os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider()
        return MockProvider()

    def _default_tool_registry(self) -> ToolRegistry:
        from ..tools.search import SearchTool
        from ..tools.fetch import FetchPageTool
        reg = ToolRegistry()
        reg.register(SearchTool())
        reg.register(FetchPageTool())
        return reg

    def run(
        self,
        topic: str,
        on_step_callback: Optional[Callable[[AgentStep], None]] = None
    ) -> ResearchReport:
        """
        Execute the multi-turn research cycle.
        """
        steps: List[AgentStep] = []
        tools_schema = self.registry.get_function_declarations()

        for step_idx in range(1, self.max_steps + 1):
            # 1. Ask LLM to reason and choose action
            thought, tool_call = self.provider.plan_next_step(topic, steps, tools_schema)

            if tool_call is None:
                # Agent concluded research is complete
                final_step = AgentStep(
                    step_number=step_idx,
                    thought=thought,
                    tool_call=None,
                    observation="Research complete. Proceeding to report synthesis."
                )
                steps.append(final_step)
                if on_step_callback:
                    on_step_callback(final_step)
                break

            # 2. Execute tool
            observation = self.registry.dispatch(tool_call.tool_name, tool_call.arguments)

            step = AgentStep(
                step_number=step_idx,
                thought=thought,
                tool_call=tool_call,
                observation=observation
            )
            steps.append(step)

            if on_step_callback:
                on_step_callback(step)

        # 3. Synthesize final report
        report = self.provider.synthesize_report(topic, steps)
        return report
