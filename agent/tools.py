"""
Tool registry for the AI Agent.
"""

from typing import Callable, Any
from dataclasses import dataclass


@dataclass
class Tool:
    """Definition of an agent tool."""
    name: str
    description: str
    function: Callable[..., Any]
    parameters: dict[str, Any]


class ToolRegistry:
    """Registry for managing agent tools."""
    
    def __init__(self):
        self.tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> list[dict]:
        """List all registered tools in OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]
    
    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return {"error": f"Tool '{name}' not found"}
        
        try:
            return tool.function(**kwargs)
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}


# Global registry instance
registry = ToolRegistry()
