
import json
import os
import unittest

from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_my.config import LLMSettings
from agent_my.llm import (
    FakeLLM,
    LLMError,
    OpenAICompatibleClient,
)
from agent_my.models import LLMResponse, ToolCall

def make_sdk_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )

def make_sdk_response(
    content=None,
    tool_calls=None,
    finish_reason="stop",
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
    )

class OpenAICompatibleClientTest(unittest.TestCase):
    def setUp(self):
        self.sdk = Mock()

        self.settings = LLMSettings(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
            model="qwen3:4b",
        )

        self.client = OpenAICompatibleClient(
            settings=self.settings,
            sdk_client=self.sdk,
        )

        self.messages = [
            {"role": "user", "content": "привет"}
        ]

    def test_text_response(self):
        self.sdk.chat.completions.create.return_value = (
            make_sdk_response(content="привет")
        )

        response = self.client.complete(self.messages)

        self.assertEqual(response.content, "привет")
        self.assertEqual(response.tool_calls, ())
        self.assertEqual(response.finish_reason, "stop")

        self.sdk.chat.completions.create.assert_called_once_with(
            model="qwen3:4b",
            messages=self.messages,
        )

    def test_single_tool_call(self):
        call = make_sdk_call(
            "call_1",
            "read_file",
            '{"path": "order.py"}',
        )

        self.sdk.chat.completions.create.return_value = (
            make_sdk_response(
                tool_calls=[call],
                finish_reason="tool_calls",
            )
        )

        tools = [{"type": "function", "function": {
            "name": "read_file",
            "description": "прочитать файл",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }}]

        response = self.client.complete(
            self.messages,
            tools=tools,
        )

        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].id, "call_1")
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(
            response.tool_calls[0].arguments,
            {"path": "order.py"},
        )

        kwargs = self.sdk.chat.completions.create.call_args.kwargs

        self.assertEqual(kwargs["tools"], tools)
        self.assertEqual(kwargs["tool_choice"], "auto")

    def test_required_tool_choice(self):
        self.sdk.chat.completions.create.return_value = make_sdk_response(
            tool_calls=[make_sdk_call("read", "read_file", '{"path":"order.py"}')],
            finish_reason="tool_calls",
        )
        tools = [{"type": "function", "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {}},
        }}]

        response = self.client.complete(self.messages, tools=tools, tool_choice="required")

        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(
            self.sdk.chat.completions.create.call_args.kwargs["tool_choice"],
            "required",
        )

    def test_multiple_tool_calls(self):
        calls = [
            make_sdk_call(
                "call_1",
                "read_file",
                '{"path": "order.py"}',
            ),
            make_sdk_call(
                "call_2",
                "read_file",
                '{"path": "test_order.py"}',
            ),
        ]

        self.sdk.chat.completions.create.return_value = (
            make_sdk_response(
                tool_calls=calls,
                finish_reason="tool_calls",
            )
        )

        response = self.client.complete(self.messages)

        self.assertEqual(len(response.tool_calls), 2)
        self.assertEqual(response.tool_calls[0].id, "call_1")
        self.assertEqual(response.tool_calls[1].id, "call_2")

    def test_invalid_json_arguments(self):
        call = make_sdk_call(
            "call_1",
            "read_file",
            "{invalid",
        )

        self.sdk.chat.completions.create.return_value = (
            make_sdk_response(tool_calls=[call])
        )

        with self.assertRaisesRegex(
            LLMError,
            "некорректный json",
        ):
            self.client.complete(self.messages)

    def test_arguments_must_be_object(self):
        call = make_sdk_call(
            "call_1",
            "read_file",
            '["order.py"]',
        )

        self.sdk.chat.completions.create.return_value = (
            make_sdk_response(tool_calls=[call])
        )

        with self.assertRaisesRegex(
            LLMError,
            "должны быть объектом",
        ):
            self.client.complete(self.messages)

    def test_empty_response(self):
        self.sdk.chat.completions.create.return_value = (
            make_sdk_response(content="")
        )

        with self.assertRaisesRegex(
            LLMError,
            "пустой ответ",
        ):
            self.client.complete(self.messages)

    def test_duplicate_call_ids(self):
        calls = [
            make_sdk_call("call_1", "read_file", '{"path":"a.py"}'),
            make_sdk_call("call_1", "read_file", '{"path":"b.py"}'),
        ]

        self.sdk.chat.completions.create.return_value = (
            make_sdk_response(tool_calls=calls)
        )

        with self.assertRaisesRegex(
            LLMError,
            "повторила идентификатор",
        ):
            self.client.complete(self.messages)

    def test_empty_choices(self):
        self.sdk.chat.completions.create.return_value = (
            SimpleNamespace(choices=[])
        )

        with self.assertRaisesRegex(
            LLMError,
            "без вариантов",
        ):
            self.client.complete(self.messages)

    def test_assistant_message(self):
        response = LLMResponse(
            content=None,
            tool_calls=(
                ToolCall(
                    id="call_1",
                    name="read_file",
                    arguments={"path": "order.py"},
                ),
            ),
            finish_reason="tool_calls",
        )

        message = response.to_assistant_message()

        self.assertEqual(message["role"], "assistant")

        call = message["tool_calls"][0]

        self.assertEqual(call["id"], "call_1")
        self.assertEqual(call["function"]["name"], "read_file")

        arguments = json.loads(
            call["function"]["arguments"]
        )

        self.assertEqual(arguments, {"path": "order.py"})

class FakeLLMTest(unittest.TestCase):
    def test_predefined_responses(self):
        fake = FakeLLM([
            LLMResponse(content="первый ответ"),
            LLMResponse(content="второй ответ"),
        ])

        self.assertEqual(
            fake.complete([]).content,
            "первый ответ",
        )

        self.assertEqual(
            fake.complete([]).content,
            "второй ответ",
        )

    def test_history_is_copied(self):
        fake = FakeLLM([
            LLMResponse(content="готово"),
        ])

        messages = [
            {"role": "user", "content": "привет"}
        ]

        fake.complete(messages)
        messages[0]["content"] = "изменённый текст"

        self.assertEqual(
            fake.calls[0]["messages"][0]["content"],
            "привет",
        )

    def test_exhausted_responses(self):
        fake = FakeLLM([])

        with self.assertRaisesRegex(
            LLMError,
            "закончились ответы",
        ):
            fake.complete([])

class LLMSettingsTest(unittest.TestCase):
    def test_default_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = LLMSettings.from_env()

        self.assertEqual(
            settings.base_url,
            "http://localhost:11434/v1/",
        )
        self.assertEqual(settings.api_key, "ollama")
        self.assertEqual(settings.model, "qwen3:4b")

if __name__ == "__main__":
    unittest.main()
