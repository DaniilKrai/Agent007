
import json

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | str

@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    truncated: bool = False

    def to_message(self) -> dict[str, str]:
        if self.success:
            content = {
                "success": True,
                "data": self.data,
            }
        else:
            content = {
                "success": False,
                "error": self.error,
            }

        if self.truncated:
            content["truncated"] = True
            if self.data is not None:
                content["data"] = self.data

        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": json.dumps(content, ensure_ascii=False),
        }

@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None

    def to_assistant_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }

        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": (
                            call.arguments
                            if isinstance(call.arguments, str)
                            else json.dumps(call.arguments, ensure_ascii=False)
                        ),
                    },
                }
                for call in self.tool_calls
            ]

        return message
