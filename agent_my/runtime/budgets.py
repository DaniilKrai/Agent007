from dataclasses import dataclass, replace

from agent_my.models import ToolResult


SIDE_EFFECT_TOOLS = frozenset({"write_file", "delete_file", "Bash"})
MIN_RESULT_BYTES = 128
MAX_RESULT_BYTES = 65536


@dataclass(frozen=True)
class BudgetLimits:
    tool_calls: int = 100
    side_effects: int = 20
    result_bytes: int = 1048576

    def __post_init__(self):
        for value in (self.tool_calls, self.side_effects, self.result_bytes):
            if type(value) is not int or value < 0:
                raise ValueError("лимиты бюджета должны быть неотрицательными целыми числами")


def result_size(result: ToolResult) -> int:
    return len(result.to_message()["content"].encode("utf-8"))


def limit_result(result: ToolResult, available: int) -> ToolResult:
    if available < MIN_RESULT_BYTES:
        raise ValueError("недостаточно бюджета для результата инструмента")
    if result_size(result) <= available:
        return result

    original = result.to_message()["content"]
    limited = replace(result, data={"preview": ""}, error=None, truncated=True)
    low, high = 0, min(len(original), available)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = replace(limited, data={"preview": original[:middle]})
        if result_size(candidate) <= available:
            low = middle
        else:
            high = middle - 1
    return replace(limited, data={"preview": original[:low]})
