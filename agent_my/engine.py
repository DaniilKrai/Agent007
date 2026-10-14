from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .llm import LLMClient
from .models import ToolCall, ToolResult
from .runtime.executor import Executor
from .tools.registry import ToolRegistry


SYSTEM_PROMPT = (
    "ты coding agent, который самостоятельно работает с программным проектом "
    "твоя задача изучать файлы, находить ошибки, исправлять код и проверять результат "
    "для взаимодействия с проектом используй предоставленные инструменты "
    "не придумывай содержимое файлов, сначала прочитай их "
    "не изменяй тесты и не пытайся обходить ограничения безопасности "
    "перед завершением задачи обязательно попробуй запустить тесты через Bash "
    "если тесты не прошли, изучи ошибки и попробуй исправить проблему "
    "не утверждай, что тесты прошли, если не получил подтверждение от инструмента "
    "если инструмент вернул ошибку, учитывай её при выборе следующего действия "
    "отвечай пользователю на русском языке "
    "начинай сообщения со строчной буквы и не ставь точку в конце сообщения"
)


class AgentError(Exception):
    pass


@dataclass(frozen=True)
class AgentRunResult:
    status: str
    final_answer: str | None
    messages: list[dict[str, Any]]
    model_calls: int
    tool_calls: int


class AgentEngine:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        executor: Executor,
        max_steps: int = 20,
        on_tool_result: Callable[
            [ToolCall, ToolResult], None
        ] | None = None,
    ):
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps <= 0
        ):
            raise ValueError(
                "количество шагов должно быть положительным целым числом"
            )

        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.max_steps = max_steps
        self.on_tool_result = on_tool_result

    def _execute_tool(self, call: ToolCall) -> ToolResult:
        try:
            return self.executor.execute(call)

        except Exception:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error="внутренняя ошибка при выполнении инструмента",
            )

    def run(self, prompt: str) -> AgentRunResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise AgentError(
                "задача для агента не должна быть пустой"
            )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        model_calls = 0
        tool_calls = 0

        definitions = self.registry.openai_tools()

        for _ in range(self.max_steps):
            response = self.llm.complete(
                messages=messages,
                tools=definitions,
            )

            model_calls += 1

            messages.append(
                response.to_assistant_message()
            )

            if response.tool_calls:
                for call in response.tool_calls:
                    result = self._execute_tool(call)

                    messages.append(
                        result.to_message()
                    )

                    tool_calls += 1

                    if self.on_tool_result is not None:
                        self.on_tool_result(call, result)

                continue

            answer = (response.content or "").strip()

            if not answer:
                raise AgentError(
                    "модель не вернула итоговый ответ"
                )

            return AgentRunResult(
                status="completed",
                final_answer=answer,
                messages=deepcopy(messages),
                model_calls=model_calls,
                tool_calls=tool_calls,
            )

        return AgentRunResult(
            status="limit_reached",
            final_answer=None,
            messages=deepcopy(messages),
            model_calls=model_calls,
            tool_calls=tool_calls,
        )