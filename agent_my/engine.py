import json
import uuid

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .llm import LLMClient
from .models import ToolCall, ToolResult
from .runtime.executor import Executor
from .runtime.store import RunStore, StoreError
from .tools.registry import ToolRegistry


SYSTEM_PROMPT = """
ты coding agent, который самостоятельно работает с программным проектом

язык общения:
- все ответы пользователю пиши строго на русском языке
- не используй украинский и другие языки для объяснений
- используй русскую лексику, грамматику и орфографию
- исключение составляют названия файлов, переменных, функций и программный код
- начинай сообщения со строчной буквы
- не ставь точку в конце сообщения
- пример правильного ответа: функция calculate_total вычисляет итоговую стоимость заказа с учётом скидки

правила работы:
- самостоятельно изучай файлы, находи ошибки, исправляй код и проверяй результат
- для взаимодействия с проектом используй предоставленные инструменты
- не придумывай содержимое файлов, сначала прочитай их
- не изменяй тесты и не пытайся обходить ограничения безопасности
- для изменения файла действительно вызывай write_file, а не просто предлагай код
- перед завершением задачи обязательно попробуй запустить тесты через Bash
- если тесты не прошли, изучи ошибки и попробуй исправить проблему
- не утверждай, что тесты прошли, если не получил подтверждение от инструмента
- если инструмент вернул ошибку, учитывай её при выборе следующего действия

важно:
- пользователь общается исключительно на русском языке
- итоговый ответ должен быть написан на русском языке
"""

class AgentError(Exception):
    pass


@dataclass(frozen=True)
class AgentRunResult:
    status: str
    final_answer: str | None
    messages: list[dict[str, Any]]
    model_calls: int
    tool_calls: int
    run_id: str


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
        store: RunStore | None = None,
        after_tool_execution: Callable[
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
        self.after_tool_execution = after_tool_execution

        self.store = store or RunStore(":memory:")

    def _snapshot(self, run_id: str) -> AgentRunResult:
        run = self.store.get_run(run_id)

        return AgentRunResult(
            status=run["status"],
            final_answer=run["final_answer"],
            messages=deepcopy(
                self.store.get_messages(run_id)
            ),
            model_calls=run["model_calls"],
            tool_calls=run["tool_calls"],
            run_id=run_id,
        )

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

    def run(
        self,
        prompt: str,
        run_id: str | None = None,
    ) -> AgentRunResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise AgentError(
                "задача для агента не должна быть пустой"
            )

        if run_id is None:
            run_id = str(uuid.uuid4())

        self.store.create_run(
            run_id=run_id,
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            max_steps=self.max_steps,
        )

        return self._drive(run_id)

    def resume(self, run_id: str) -> AgentRunResult:
        run = self.store.get_run(run_id)

        if run["status"] != "running":
            return self._snapshot(run_id)

        self.store.mark_uncertain(run_id)

        run = self.store.get_run(run_id)

        if run["status"] == "waiting_resolution":
            return self._snapshot(run_id)

        return self._drive(run_id)

    def resolve(
        self,
        run_id: str,
        action: str,
    ) -> AgentRunResult:
        self.store.resolve(run_id, action)

        if action == "abort":
            return self._snapshot(run_id)

        return self._drive(run_id)

    def _drive(self, run_id: str) -> AgentRunResult:
        while True:
            run = self.store.get_run(run_id)

            if run["status"] != "running":
                return self._snapshot(run_id)

            pending = self.store.next_pending(run_id)

            if pending is not None:
                position = pending["position"]

                call = ToolCall(
                    id=pending["call_id"],
                    name=pending["name"],
                    arguments=json.loads(
                        pending["arguments"]
                    ),
                )

                if pending["override_result"] is not None:
                    data = json.loads(
                        pending["override_result"]
                    )

                    result = ToolResult(**data)

                    self.store.deliver_result(
                        run_id,
                        position,
                        result,
                        executed=False,
                    )

                    if self.on_tool_result is not None:
                        self.on_tool_result(call, result)

                    continue

                if pending["status"] == "completed":
                    result = ToolResult(
                        **json.loads(pending["result"])
                    )

                    self.store.deliver_result(
                        run_id,
                        position,
                        result,
                        executed=False,
                    )

                    continue

                if pending["status"] in (
                    "started",
                    "uncertain",
                ):
                    self.store.mark_uncertain(run_id)
                    return self._snapshot(run_id)

                if pending["status"] != "queued":
                    raise StoreError(
                        "неизвестное состояние вызова инструмента"
                    )

                self.store.begin_attempt(
                    run_id,
                    call.id,
                )

                result = self._execute_tool(call)

                if self.after_tool_execution is not None:
                    self.after_tool_execution(
                        call,
                        result,
                    )

                self.store.deliver_result(
                    run_id,
                    position,
                    result,
                    executed=True,
                )

                if self.on_tool_result is not None:
                    self.on_tool_result(call, result)

                continue

            if run["model_calls"] >= run["max_steps"]:
                self.store.mark_limit(run_id)
                return self._snapshot(run_id)

            response = self.llm.complete(
                messages=self.store.get_messages(run_id),
                tools=self.registry.openai_tools(),
            )

            if (
                not response.tool_calls
                and not (response.content or "").strip()
            ):
                raise AgentError(
                    "модель не вернула итоговый ответ"
                )

            self.store.record_response(
                run_id,
                response,
            )