
import json

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None

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

        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": json.dumps(content, ensure_ascii=False),
        }
