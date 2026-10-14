import ast
import json
import uuid

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .llm import LLMClient
from .models import LLMResponse, ToolCall, ToolResult
from .runtime.executor import Executor
from .runtime.store import RunStore, StoreError
from .runtime.planning import PLANNING_PROMPT, PLAN_RETRY_MESSAGE, PLAN_SCHEMA
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

правила работы:
- самостоятельно изучай файлы, находи ошибки, исправляй код и проверяй результат
- для взаимодействия с проектом используй предоставленные инструменты
- не придумывай содержимое файлов, сначала прочитай их
- перед исправлением прочитай и исходный файл, и относящиеся к нему тесты
- исправляй все случаи, проверяемые тестами, включая проверки ошибочных аргументов
- не изменяй тесты и не пытайся обходить ограничения безопасности
- для изменения файла действительно вызывай write_file, а не просто предлагай код
- перед завершением задачи обязательно попробуй запустить тесты через Bash
- если тесты не прошли, изучи ошибки и попробуй исправить проблему
- не утверждай, что тесты прошли, если не получил подтверждение от инструмента
- если инструмент вернул ошибку, учитывай её при выборе следующего действия
- если пользователь отклонил действие, учитывай отказ и выбирай дальнейшие шаги

важно:
- пользователь общается исключительно на русском языке
- итоговый ответ должен быть написан на русском языке
"""

TOOL_CALL_RETRY_MESSAGE = (
    "предыдущий ответ содержит вызов инструмента обычным текстом, "
    "поэтому этот вызов не был выполнен\n"
    "для действия используй настоящий tool call из предоставленного списка, "
    "не записывай вызов в content или в блок кода\n"
    "после изменения файла проверь результат через Bash\n"
    "если действие невозможно или больше не требуется, объясни это обычным текстом"
)
MAX_TOOL_CALL_RETRIES = 2
VERIFY_TESTS_PROMPT = (
    "успешное завершение этого запуска требует проверки тестов через Bash\n"
    "после последнего изменения файлов выполни python3 -m unittest -v "
    "в рабочей директории проекта и получи код завершения 0\n"
    "вызывай инструменты через tool_calls, не выводи код вместо изменения файла"
)
VERIFY_TESTS_RETRY_MESSAGE = (
    "задача ещё не подтверждена: после последних изменений нет успешного "
    "запуска тестов через Bash\n"
    "сейчас вызови нужный инструмент через tool_calls: если исправление ещё "
    "не записано, используй write_file с полным содержимым файла, "
    "затем Bash с командой python3 -m unittest -v и cwd='.'\n"
    "если тесты не пройдут, исправь причину и запусти их снова\n"
    "не отвечай блоком кода или описанием предполагаемых изменений"
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
        mode: str = "auto",
        verify_tests: bool = False,
    ):
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps <= 0
        ):
            raise ValueError(
                "количество шагов должно быть положительным целым числом"
            )

        if mode not in ("safe", "auto"):
            raise ValueError(
                "неизвестный режим выполнения"
            )

        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.max_steps = max_steps
        self.on_tool_result = on_tool_result
        self.after_tool_execution = after_tool_execution
        self.mode = mode
        self.verify_tests = verify_tests

        self.store = (
            store
            if store is not None
            else RunStore(":memory:")
        )

    def _snapshot(
        self,
        run_id: str,
    ) -> AgentRunResult:
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

    def _execute_tool(
        self,
        call: ToolCall,
        mode: str,
        approved: bool,
        phase: str,
    ) -> ToolResult:
        try:
            return self.executor.execute(call, mode=mode, approved=approved, phase=phase)

        except Exception:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error="внутренняя ошибка при выполнении инструмента",
            )

    def _is_text_tool_call(self, content: str | None) -> bool:
        text = (content or "").strip()
        lines = text.splitlines()

        if (
            len(lines) >= 3
            and lines[0] in ("```", "```python", "```py")
            and lines[-1] == "```"
        ):
            text = "\n".join(lines[1:-1])

        try:
            statements = ast.parse(text).body
        except (SyntaxError, ValueError, RecursionError):
            return False

        names = {
            tool["function"]["name"]
            for tool in self.registry.openai_tools()
        }

        return bool(statements) and all(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id in names
            for statement in statements
        )

    @staticmethod
    def _tests_verified(messages: list[dict[str, Any]]) -> bool:
        calls = {}
        verified = False
        seen = set()

        for message in messages:
            if message["role"] == "assistant":
                for call in message.get("tool_calls", []):
                    calls[call["id"]] = call["function"]
            elif message["role"] == "tool":
                call_id = message["tool_call_id"]
                if call_id in seen:
                    continue
                seen.add(call_id)
                call = calls.get(call_id, {})
                name = call.get("name")
                result = json.loads(message["content"])
                if name in ("write_file", "delete_file", "Bash"):
                    verified = False
                if name == "Bash" and result.get("success"):
                    arguments = json.loads(call["arguments"])
                    data = result.get("data") or {}
                    verified = (
                        arguments.get("command") in (
                            "python3 -m unittest -v",
                            "python -m unittest -v",
                        )
                        and arguments.get("cwd") == "."
                        and data.get("exit_code") == 0
                        and not data.get("timed_out")
                        and not data.get("truncated")
                    )

        return verified

    def run(
        self,
        prompt: str,
        run_id: str | None = None,
        plan: bool = False,
    ) -> AgentRunResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise AgentError(
                "задача для агента не должна быть пустой"
            )

        if run_id is None:
            run_id = str(uuid.uuid4())

        system_prompt = SYSTEM_PROMPT
        if self.verify_tests:
            system_prompt += "\n" + VERIFY_TESTS_PROMPT
        if plan:
            system_prompt += "\n" + PLANNING_PROMPT

        self.store.create_run(
            run_id=run_id,
            prompt=prompt,
            system_prompt=system_prompt,
            max_steps=self.max_steps,
            mode=self.mode,
            phase="planning" if plan else "execution",
        )

        return self._drive(run_id)

    def resume(
        self,
        run_id: str,
    ) -> AgentRunResult:
        run = self.store.get_run(run_id)

        if run["status"] == "waiting_approval":
            return self._snapshot(run_id)

        if run["status"] != "running":
            return self._snapshot(run_id)

        self.store.mark_uncertain(run_id)

        run = self.store.get_run(run_id)

        if run["status"] == "waiting_resolution":
            return self._snapshot(run_id)

        return self._drive(run_id)

    def approve_plan(self, run_id: str) -> AgentRunResult:
        self.store.decide_plan(run_id, "approve")
        return self._drive(run_id)

    def reject_plan(self, run_id: str) -> AgentRunResult:
        self.store.decide_plan(run_id, "reject")
        return self._snapshot(run_id)

    def approve(
        self,
        run_id: str,
    ) -> AgentRunResult:
        self.store.decide_approval(
            run_id,
            "approve",
        )

        return self._drive(run_id)

    def reject(
        self,
        run_id: str,
    ) -> AgentRunResult:
        approval = self.store.get_pending_approval(
            run_id
        )

        result = self.store.decide_approval(
            run_id,
            "reject",
        )

        if (
            approval is not None
            and result is not None
            and self.on_tool_result is not None
        ):
            call = ToolCall(
                id=approval["call_id"],
                name=approval["name"],
                arguments=json.loads(
                    approval["arguments"]
                ),
            )

            self.on_tool_result(call, result)

        return self._drive(run_id)

    def resolve(
        self,
        run_id: str,
        action: str,
    ) -> AgentRunResult:
        self.store.resolve(
            run_id,
            action,
        )

        if action == "abort":
            return self._snapshot(run_id)

        return self._drive(run_id)

    def _drive(
        self,
        run_id: str,
    ) -> AgentRunResult:
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

                approval = self.store.get_approval(run_id, call.id)
                approved = approval == "approved"
                decision = self.executor.check(
                    call, mode=run["mode"], approved=approved, phase=run["phase"],
                )

                validation_error = decision.error
                if validation_error is None and call.name == "submit_plan" and self.store.batch_size(run_id) != 1:
                    validation_error = ToolResult(
                        call.id, call.name, False,
                        error="отправь план отдельным вызовом submit_plan, без других инструментов в ответе",
                    )

                if validation_error is not None:
                    self.store.complete_without_execution(
                        run_id,
                        position,
                        validation_error,
                    )

                    if self.on_tool_result is not None:
                        self.on_tool_result(
                            call,
                            validation_error,
                        )

                    continue

                if decision.requires_approval:
                    if approval is None:
                        self.store.request_approval(
                            run_id,
                            call.id,
                        )

                        return self._snapshot(run_id)

                    if approval == "pending":
                        raise StoreError(
                            "запуск ожидает подтверждения действия"
                        )

                    if approval != "approved":
                        raise StoreError(
                            "некорректное состояние подтверждения"
                        )

                self.store.begin_attempt(
                    run_id,
                    call.id,
                )

                result = self._execute_tool(call, run["mode"], approved, run["phase"])

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
                    self.on_tool_result(
                        call,
                        result,
                    )

                continue

            if run["model_calls"] >= run["max_steps"]:
                self.store.mark_limit(run_id)
                return self._snapshot(run_id)

            messages = self.store.get_messages(run_id)
            tools = [
                tool for tool in self.registry.openai_tools()
                if (
                    tool["function"]["name"] in ("list_files", "read_file", "submit_plan")
                    if run["phase"] == "planning"
                    else tool["function"]["name"] != "submit_plan"
                )
            ]
            structured_plan = run["phase"] == "planning" and any(
                message.get("role") == "system"
                and message.get("content") == PLAN_RETRY_MESSAGE
                for message in messages
            )
            request_messages = deepcopy(messages)
            if run["phase"] == "execution":
                request_messages[0]["content"] = request_messages[0]["content"].replace(
                    "\n" + PLANNING_PROMPT, "",
                )
                request_messages = [
                    message for message in request_messages
                    if not (
                        message.get("role") == "system"
                        and message.get("content") == PLAN_RETRY_MESSAGE
                    )
                ]
            response = self.llm.complete(
                messages=request_messages,
                tools=None if structured_plan else tools,
                response_schema=PLAN_SCHEMA if structured_plan else None,
                tool_choice=(
                    "required"
                    if messages[-1].get("role") == "system"
                    and messages[-1].get("content") in (
                        TOOL_CALL_RETRY_MESSAGE,
                        VERIFY_TESTS_RETRY_MESSAGE,
                        PLAN_RETRY_MESSAGE,
                    )
                    else "auto"
                ),
            )

            if structured_plan and not response.tool_calls:
                try:
                    plan = json.loads(response.content or "")
                except (json.JSONDecodeError, RecursionError):
                    pass
                else:
                    response = LLMResponse(
                        content=response.content,
                        tool_calls=(ToolCall(
                            id="plan_" + uuid.uuid4().hex,
                            name="submit_plan",
                            arguments=plan if isinstance(plan, dict) else response.content,
                        ),),
                        finish_reason=response.finish_reason,
                    )

            retry_message = None
            error = None
            verification_missing = (
                run["phase"] == "execution"
                and VERIFY_TESTS_PROMPT in messages[0]["content"]
                and not self._tests_verified(messages)
            )

            if not response.tool_calls and run["phase"] == "planning":
                retries = sum(
                    message.get("role") == "system"
                    and message.get("content") == PLAN_RETRY_MESSAGE
                    for message in messages
                )
                if retries < MAX_TOOL_CALL_RETRIES:
                    retry_message = PLAN_RETRY_MESSAGE
                else:
                    error = "модель не представила структурированный план, исполнение не начато"

            elif not response.tool_calls and verification_missing:
                retries = sum(
                    message.get("role") == "system"
                    and message.get("content") == VERIFY_TESTS_RETRY_MESSAGE
                    for message in messages
                )
                if retries < MAX_TOOL_CALL_RETRIES:
                    retry_message = VERIFY_TESTS_RETRY_MESSAGE
                else:
                    error = (
                        "задача не завершена: модель не подтвердила "
                        "исправление успешным запуском тестов через Bash"
                    )

            elif (
                not response.tool_calls
                and not (response.content or "").strip()
            ):
                raise AgentError(
                    "модель не вернула итоговый ответ"
                )

            elif (
                not response.tool_calls
                and self._is_text_tool_call(response.content)
            ):
                retries = sum(
                    message.get("role") == "system"
                    and message.get("content") == TOOL_CALL_RETRY_MESSAGE
                    for message in messages
                )

                if retries < MAX_TOOL_CALL_RETRIES:
                    retry_message = TOOL_CALL_RETRY_MESSAGE
                else:
                    error = (
                        "модель повторно вернула вызов инструмента текстом, "
                        "задача не завершена, текстовый вызов не выполнен"
                    )

            self.store.record_response(
                run_id,
                response,
                retry_message=retry_message,
                error=error,
            )
