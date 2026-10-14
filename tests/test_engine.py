import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from agent_my.engine import (
    AgentEngine,
    AgentError,
)
from agent_my.llm import FakeLLM
from agent_my.models import LLMResponse, ToolCall
from agent_my.runtime.executor import Executor
from agent_my.runtime.policy import Policy
from agent_my.tools import build_tool_registry


class AgentEngineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.root = Path(self.temp.name)

        self.workspace = self.root / "creating_space"
        self.workspace.mkdir()

        self.original_code = (
            "def calculate_total("
            "prices: list[float], "
            "discount_percent: float) -> float:\n"
            "    return sum(prices) - discount_percent\n"
        )

        self.corrected_code = (
            "def calculate_total("
            "prices: list[float], "
            "discount_percent: float) -> float:\n"
            "    if not 0 <= discount_percent <= 100:\n"
            "        raise ValueError('некорректная скидка')\n"
            "    return sum(prices) * "
            "(1 - discount_percent / 100)\n"
        )

        self.test_code = (
            "import unittest\n"
            "from order import calculate_total\n"
            "\n"
            "class OrderTest(unittest.TestCase):\n"
            "    def test_discount(self):\n"
            "        self.assertEqual("
            "calculate_total([100, 50], 10), 135)\n"
            "\n"
            "    def test_invalid_discount(self):\n"
            "        for value in (-1, 101):\n"
            "            with self.assertRaises(ValueError):\n"
            "                calculate_total([100], value)\n"
        )

        (self.workspace / "order.py").write_text(
            self.original_code,
            encoding="utf-8",
        )

        (self.workspace / "test_order.py").write_text(
            self.test_code,
            encoding="utf-8",
        )

        (self.workspace / "private").mkdir()

        self.policy_path = self.root / "policy.json"

        self.policy_path.write_text(
            json.dumps({
                "commands": {
                    "allow": [
                        "pwd",
                        "python3 -m unittest -v",
                    ],
                    "deny": [
                        "rm -rf .",
                    ],
                },
                "directories": {
                    "allow": ["."],
                    "deny": ["private"],
                },
                "files": {
                    "deny": ["test_order.py"],
                },
            }),
            encoding="utf-8",
        )

        self.policy = Policy(
            self.policy_path,
            self.workspace,
        )

        self.registry = build_tool_registry(
            self.workspace,
            policy=self.policy,
        )

        self.executor = Executor(
            self.registry,
            self.policy,
        )

        self.events = []

    def make_engine(
        self,
        responses,
        max_steps=20,
    ):
        fake = FakeLLM(responses)

        engine = AgentEngine(
            llm=fake,
            registry=self.registry,
            executor=self.executor,
            max_steps=max_steps,
            on_tool_result=lambda call, result: (
                self.events.append((call, result))
            ),
        )

        return fake, engine

    def test_complete_task(self):
        responses = [
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="list_files",
                        arguments={"path": "."},
                    ),
                ),
            ),

            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_2",
                        name="read_file",
                        arguments={"path": "order.py"},
                    ),
                    ToolCall(
                        id="call_3",
                        name="read_file",
                        arguments={"path": "test_order.py"},
                    ),
                ),
            ),

            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_4",
                        name="write_file",
                        arguments={
                            "path": "order.py",
                            "content": self.corrected_code,
                        },
                    ),
                ),
            ),

            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_5",
                        name="Bash",
                        arguments={
                            "command": "python3 -m unittest -v",
                            "cwd": ".",
                        },
                    ),
                ),
            ),

            LLMResponse(
                content="исправление выполнено, тесты проходят",
            ),
        ]

        fake, engine = self.make_engine(responses)

        result = engine.run(
            "исправь расчёт стоимости заказа"
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.model_calls, 5)
        self.assertEqual(result.tool_calls, 5)

        self.assertEqual(
            result.final_answer,
            "исправление выполнено, тесты проходят",
        )

        actual_code = (
            self.workspace / "order.py"
        ).read_text(encoding="utf-8")

        self.assertEqual(
            actual_code,
            self.corrected_code,
        )

        actual_tests = (
            self.workspace / "test_order.py"
        ).read_text(encoding="utf-8")

        self.assertEqual(
            actual_tests,
            self.test_code,
        )

        bash_calls = [
            item
            for item in self.events
            if item[0].name == "Bash"
        ]

        self.assertEqual(len(bash_calls), 1)

        bash_result = bash_calls[0][1]

        self.assertTrue(bash_result.success)

        self.assertEqual(
            bash_result.data["exit_code"],
            0,
        )

        self.assertIn(
            "OK",
            bash_result.data["stderr"],
        )

        self.assertEqual(
            len(fake.calls),
            5,
        )

    def test_multiple_tool_calls(self):
        responses = [
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="read_file",
                        arguments={"path": "order.py"},
                    ),
                    ToolCall(
                        id="call_2",
                        name="read_file",
                        arguments={"path": "test_order.py"},
                    ),
                ),
            ),

            LLMResponse(
                content="файлы прочитаны",
            ),
        ]

        fake, engine = self.make_engine(responses)

        result = engine.run("прочитай файлы")

        self.assertEqual(result.model_calls, 2)
        self.assertEqual(result.tool_calls, 2)

        second_request = fake.calls[1]["messages"]

        tool_messages = [
            message
            for message in second_request
            if message["role"] == "tool"
        ]

        self.assertEqual(len(tool_messages), 2)

        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages],
            ["call_1", "call_2"],
        )

    def test_denied_file_change(self):
        responses = [
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="write_file",
                        arguments={
                            "path": "test_order.py",
                            "content": "changed",
                        },
                    ),
                ),
            ),

            LLMResponse(
                content="изменение тестов запрещено",
            ),
        ]

        fake, engine = self.make_engine(responses)

        result = engine.run(
            "измени test_order.py"
        )

        self.assertEqual(result.status, "completed")

        actual = (
            self.workspace / "test_order.py"
        ).read_text(encoding="utf-8")

        self.assertEqual(actual, self.test_code)

        tool_message = fake.calls[1]["messages"][-1]

        content = json.loads(
            tool_message["content"]
        )

        self.assertFalse(content["success"])

        self.assertIn(
            "изменение файла запрещено",
            content["error"],
        )

    def test_unknown_tool(self):
        responses = [
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="unknown_tool",
                        arguments={},
                    ),
                ),
            ),

            LLMResponse(
                content="инструмент недоступен",
            ),
        ]

        fake, engine = self.make_engine(responses)

        result = engine.run("вызови инструмент")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_calls, 1)

        tool_message = fake.calls[1]["messages"][-1]

        content = json.loads(tool_message["content"])

        self.assertFalse(content["success"])

    def test_invalid_path(self):
        responses = [
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="read_file",
                        arguments={"path": "../policy.json"},
                    ),
                ),
            ),

            LLMResponse(
                content="доступ к файлу запрещён",
            ),
        ]

        fake, engine = self.make_engine(responses)

        result = engine.run("прочитай файл")

        self.assertEqual(result.status, "completed")

        content = json.loads(
            fake.calls[1]["messages"][-1]["content"]
        )

        self.assertFalse(content["success"])

        self.assertIn(
            "переход в родительскую директорию запрещён",
            content["error"],
        )

    def test_max_steps(self):
        responses = [
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="read_file",
                        arguments={"path": "order.py"},
                    ),
                ),
            ),
        ]

        fake, engine = self.make_engine(
            responses,
            max_steps=1,
        )

        result = engine.run("прочитай файл")

        self.assertEqual(
            result.status,
            "limit_reached",
        )

        self.assertIsNone(result.final_answer)
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(len(fake.calls), 1)

    def test_empty_model_response(self):
        fake, engine = self.make_engine([
            LLMResponse(content=""),
        ])

        with self.assertRaisesRegex(
            AgentError,
            "модель не вернула итоговый ответ",
        ):
            engine.run("привет")

    def test_empty_prompt(self):
        fake, engine = self.make_engine([])

        with self.assertRaisesRegex(
            AgentError,
            "задача для агента не должна быть пустой",
        ):
            engine.run("   ")

        self.assertEqual(len(fake.calls), 0)

    def test_invalid_max_steps(self):
        for value in (0, -1, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    AgentEngine(
                        llm=FakeLLM([]),
                        registry=self.registry,
                        executor=self.executor,
                        max_steps=value,
                    )

    def test_executor_exception(self):
        responses = [
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="read_file",
                        arguments={"path": "order.py"},
                    ),
                ),
            ),

            LLMResponse(
                content="произошла ошибка инструмента",
            ),
        ]

        fake, engine = self.make_engine(responses)

        with patch.object(
            self.executor,
            "execute",
            side_effect=RuntimeError("test error"),
        ):
            result = engine.run("прочитай файл")

        self.assertEqual(result.status, "completed")

        content = json.loads(
            fake.calls[1]["messages"][-1]["content"]
        )

        self.assertFalse(content["success"])

        self.assertEqual(
            content["error"],
            "внутренняя ошибка при выполнении инструмента",
        )

    def test_tool_callback(self):
        responses = [
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="read_file",
                        arguments={"path": "order.py"},
                    ),
                ),
            ),

            LLMResponse(content="готово"),
        ]

        fake, engine = self.make_engine(responses)

        engine.run("прочитай файл")

        self.assertEqual(len(self.events), 1)

        call, result = self.events[0]

        self.assertEqual(call.name, "read_file")
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()