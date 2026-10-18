"""
Base interfaces and registry for agent tools.
Follows the Open/Closed Principle: new tools can be added without modifying the agent loop.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json


class BaseTool(ABC):
    """Abstract Base Class for all ResearchScout tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed instruction of what the tool does and when to call it."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema defining the expected parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool action and return text observation."""
        pass

    def to_function_declaration(self) -> Dict[str, Any]:
        """Format as a standard function declaration for LLM tool calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema
        }


class ToolRegistry:
    """Manages registration, schema discovery, and dispatching of tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_function_declarations(self) -> List[Dict[str, Any]]:
        """Return OpenAPI/JSON-schema function declarations for all tools."""
        return [tool.to_function_declaration() for tool in self._tools.values()]

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Safely execute a tool by name with error boundary."""
        tool = self.get_tool(tool_name)
        if not tool:
            available = ", ".join(self._tools.keys())
            return f"Error: Tool '{tool_name}' not found. Available tools: [{available}]"

        try:
            return tool.execute(**arguments)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
