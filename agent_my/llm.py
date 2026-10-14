
import json

from collections import deque
from copy import deepcopy
from typing import Any, Protocol, Sequence

from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    OpenAI,
    OpenAIError,
)

from .config import LLMSettings
from .models import LLMResponse, ToolCall

class LLMError(Exception):
    pass

class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        ...

class OpenAICompatibleClient:
    def __init__(
        self,
        settings: LLMSettings | None = None,
        sdk_client: Any | None = None,
    ):
        self.settings = settings or LLMSettings.from_env()

        if sdk_client is not None:
            self._client = sdk_client
        else:
            self._client = OpenAI(
                base_url=self.settings.base_url,
                api_key=self.settings.api_key,
                timeout=180.0,
                max_retries=0,
            )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if not messages:
            raise LLMError("нельзя отправить пустую историю сообщений")

        request: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
        }

        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(
                **request
            )

        except APITimeoutError as exc:
            raise LLMError(
                "превышено время ожидания ответа модели"
            ) from exc

        except APIConnectionError as exc:
            raise LLMError(
                "не удалось подключиться к серверу модели, проверьте запуск ollama"
            ) from exc

        except APIStatusError as exc:
            if exc.status_code == 404:
                raise LLMError(
                    "модель не найдена, проверьте OPENAI_MODEL и команду ollama list"
                ) from exc

            raise LLMError(
                "сервер модели отклонил запрос"
            ) from exc

        except OpenAIError as exc:
            raise LLMError(
                "ошибка при обращении к языковой модели"
            ) from exc

        if not response.choices:
            raise LLMError("модель вернула ответ без вариантов")

        choice = response.choices[0]
        message = choice.message
        content = message.content

        if content is not None and not isinstance(content, str):
            raise LLMError(
                "модель вернула текст в неподдерживаемом формате"
            )

        calls = []
        used_ids = set()

        for raw_call in message.tool_calls or []:
            if getattr(raw_call, "type", "function") != "function":
                raise LLMError(
                    "модель запросила неподдерживаемый тип инструмента"
                )

            call_id = getattr(raw_call, "id", None)
            function = getattr(raw_call, "function", None)

            name = getattr(function, "name", None)
            raw_arguments = getattr(function, "arguments", None)

            if not isinstance(call_id, str) or not call_id:
                raise LLMError(
                    "модель вернула вызов без идентификатора"
                )

            if call_id in used_ids:
                raise LLMError(
                    "модель повторила идентификатор инструмента"
                )

            used_ids.add(call_id)

            if not isinstance(name, str) or not name:
                raise LLMError(
                    "модель вернула вызов без имени инструмента"
                )

            if isinstance(raw_arguments, dict):
                arguments = raw_arguments

            elif isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise LLMError(
                        "модель вернула некорректный json аргументов инструмента"
                    ) from exc

            else:
                raise LLMError(
                    "модель вернула аргументы в неподдерживаемом формате"
                )

            if not isinstance(arguments, dict):
                raise LLMError(
                    "аргументы инструмента должны быть объектом"
                )

            calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )

        if not calls and (not content or not content.strip()):
            raise LLMError("модель вернула пустой ответ")

        return LLMResponse(
            content=content,
            tool_calls=tuple(calls),
            finish_reason=choice.finish_reason,
        )

class FakeLLM:
    def __init__(self, responses: Sequence[LLMResponse]):
        self._responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append({
            "messages": deepcopy(messages),
            "tools": deepcopy(tools),
        })

        if not self._responses:
            raise LLMError(
                "у тестовой модели закончились ответы"
            )

        return self._responses.popleft()