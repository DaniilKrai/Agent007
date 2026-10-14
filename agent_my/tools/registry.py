from dataclasses import dataclass
from typing import Any, Callable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

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
            raise ValueError(f"инструмент уже зарегистрирован: {name}")

        try:
            Draft202012Validator.check_schema(parameters)
        except SchemaError as exc:
            raise ValueError(
                "некорректная схема аргументов инструмента"
            ) from exc

        if parameters.get("type") != "object":
            raise ValueError(
                "параметры инструмента должны описываться схемой объекта"
            )

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

    @staticmethod
    def _validation_message(error: ValidationError) -> str:
        argument = ".".join(str(part) for part in error.path)

        if error.validator == "required":
            missing = [
                name
                for name in error.validator_value
                if name not in error.instance
            ]

            if missing:
                return f"отсутствует обязательный аргумент: {missing[0]}"

        if error.validator == "additionalProperties":
            extra = sorted(
                set(error.instance)
                - set(error.schema.get("properties", {}))
            )

            if extra:
                return f"неизвестные аргументы: {', '.join(extra)}"

        if error.validator == "type":
            types = {
                "string": "строка",
                "integer": "целое число",
                "number": "число",
                "boolean": "логическое значение",
                "object": "объект",
                "array": "массив",
                "null": "null",
            }

            expected = error.validator_value

            if isinstance(expected, list):
                expected = ", ".join(
                    types.get(value, value)
                    for value in expected
                )
            else:
                expected = types.get(expected, expected)

            if argument:
                return (
                    f"неверный тип аргумента {argument}: "
                    f"ожидается {expected}"
                )

            return f"неверный тип аргументов: ожидается {expected}"

        if error.validator == "minLength":
            return f"слишком короткое значение аргумента: {argument}"

        return "аргументы не соответствуют схеме"

    def execute(self, call: ToolCall, result_limit: int | None = None) -> ToolResult:
        tool = self._tools.get(call.name)

        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=f"неизвестный инструмент: {call.name}",
            )

        validator = Draft202012Validator(tool.parameters)
        errors = list(validator.iter_errors(call.arguments))

        if errors:
            message = self._validation_message(errors[0])

            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=f"некорректные аргументы: {message}",
            )

        try:
            arguments = dict(call.arguments)
            if result_limit is not None:
                arguments["result_limit"] = result_limit
            data = tool.handler(**arguments)

            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=True,
                data=data,
                truncated=bool(data.get("truncated")),
            )

        except (
            PermissionError,
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            ValueError,
        ) as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=str(exc) or "операция не выполнена",
            )

        except UnicodeError:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error="ошибка обработки текста в кодировке utf-8",
            )

        except OSError:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error="ошибка файловой системы",
            )

        except Exception:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error="внутренняя ошибка при выполнении инструмента",
            )
