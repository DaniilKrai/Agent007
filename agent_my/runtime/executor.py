from jsonschema import Draft202012Validator

from agent_my.models import ToolCall, ToolResult
from agent_my.tools.registry import ToolRegistry

from .policy import Policy


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

            else:
                raise PermissionError(
                    f"инструмент не разрешён: {call.name}"
                )

        except (
            PermissionError,
            ValueError,
            FileNotFoundError,
            NotADirectoryError,
        ) as exc:
            return self._error(call, str(exc))

        return None

    def execute(self, call: ToolCall) -> ToolResult:
        error = self.validate(call)

        if error is not None:
            return error

        return self.registry.execute(call)