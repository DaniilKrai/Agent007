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

    def execute(self, call: ToolCall) -> ToolResult:
        definitions = {
            item["function"]["name"]: item["function"]
            for item in self.registry.openai_tools()
        }

        definition = definitions.get(call.name)

        if definition is None:
            return self.registry.execute(call)

        validator = Draft202012Validator(
            definition["parameters"]
        )

        if not validator.is_valid(call.arguments):
            return self.registry.execute(call)

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
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=str(exc),
            )

        return self.registry.execute(call)