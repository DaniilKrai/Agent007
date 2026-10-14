
from dataclasses import dataclass
from typing import Any, Callable

from jsonschema import Draft202012Validator

from agent_my.models import ToolCall, ToolResult


@dataclass
class RegisteredTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., dict],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        Draft202012Validator.check_schema(parameters)

        if parameters.get("type") != "object":
            raise ValueError("Tool parameters must be an object schema")

        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def openai_tools(self) -> list[dict]:
        result = []

        for tool in self._tools.values():
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })

        return result

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)

        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=f"Unknown tool: {call.name}",
            )

        validator = Draft202012Validator(tool.parameters)
        errors = list(validator.iter_errors(call.arguments))

        if errors:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=f"Invalid arguments: {errors[0].message}",
            )

        try:
            data = tool.handler(**call.arguments)

            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=True,
                data=data,
            )

        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
