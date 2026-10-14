import json

from dataclasses import dataclass
from jsonschema import Draft202012Validator

from agent_my.models import ToolCall, ToolResult
from agent_my.tools.registry import ToolRegistry

from .policy import Policy


@dataclass(frozen=True)
class ExecutionDecision:
    error: ToolResult | None = None
    requires_approval: bool = False


class Executor:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: Policy,
    ):
        self.registry = registry
        self.policy = policy

    def _error(
        self,
        call: ToolCall,
        message: str,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=False,
            error=message,
        )

    def validate(
        self,
        call: ToolCall,
    ) -> ToolResult | None:
        if not isinstance(call.name, str) or not call.name:
            return self._error(call, "некорректное имя инструмента")

        if isinstance(call.arguments, str):
            try:
                json.loads(call.arguments)
            except (json.JSONDecodeError, RecursionError):
                return self._error(call, "некорректный json аргументов инструмента")

            return self._error(call, "аргументы инструмента должны быть объектом")

        definitions = {
            item["function"]["name"]: item["function"]
            for item in self.registry.openai_tools()
        }

        definition = definitions.get(call.name)

        if definition is None:
            return self._error(
                call,
                f"неизвестный инструмент: {call.name}",
            )

        validator = Draft202012Validator(
            definition["parameters"]
        )

        if not validator.is_valid(call.arguments):
            return self._error(
                call,
                "некорректные аргументы инструмента",
            )

        try:
            self.policy.reload()
            if call.name == "Bash":
                self.policy.check_bash(
                    command=call.arguments["command"],
                    cwd=call.arguments["cwd"],
                )

            elif call.name in (
                "list_files",
                "read_file",
                "write_file",
                "delete_file",
            ):
                self.policy.check_file(
                    tool_name=call.name,
                    path=call.arguments["path"],
                )

            elif call.name != "submit_plan":
                raise PermissionError(
                    f"инструмент не разрешён: {call.name}"
                )

        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as exc:
            return self._error(call, str(exc))

        return None

    def check(
        self,
        call: ToolCall,
        mode: str,
        approved: bool = False,
        phase: str = "execution",
    ) -> ExecutionDecision:
        if mode not in ("safe", "auto"):
            return ExecutionDecision(
                error=self._error(call, "неизвестный режим выполнения"),
            )

        if phase not in ("planning", "execution"):
            return ExecutionDecision(error=self._error(call, "неизвестная фаза запуска"))

        error = self.validate(call)

        if error is not None:
            return ExecutionDecision(error=error)

        if phase == "planning" and call.name not in ("list_files", "read_file", "submit_plan"):
            return ExecutionDecision(error=self._error(
                call, "до утверждения плана разрешены только чтение файлов и отправка плана",
            ))

        if phase == "execution" and call.name == "submit_plan":
            return ExecutionDecision(error=self._error(call, "запуск не находится в режиме планирования"))

        requires_approval = (
            call.name == "delete_file"
            or mode == "safe" and call.name in ("write_file", "Bash")
        )

        return ExecutionDecision(
            requires_approval=requires_approval and approved is not True,
        )

    def execute(
        self,
        call: ToolCall,
        mode: str = "safe",
        approved: bool = False,
        phase: str = "execution",
        result_limit: int | None = None,
    ) -> ToolResult:
        decision = self.check(call, mode, approved, phase)

        if decision.error is not None:
            return decision.error

        if decision.requires_approval:
            return self._error(call, "для выполнения инструмента требуется подтверждение")

        return self.registry.execute(call, result_limit=result_limit)
