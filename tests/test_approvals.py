import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from agent_my.engine import AgentEngine
from agent_my.llm import FakeLLM
from agent_my.models import LLMResponse, ToolCall
from agent_my.runtime.executor import Executor
from agent_my.runtime.policy import Policy
from agent_my.runtime.store import RunStore, StoreError
from agent_my.tools import build_tool_registry


class SimulatedCrash(BaseException):
    pass


class ApprovalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.root = Path(self.temp.name)

        self.workspace = self.root / "creating_space"
        self.workspace.mkdir()

        (self.workspace / "order.py").write_text(
            "value = 1\n",
            encoding="utf-8",
        )

        (self.workspace / "test_order.py").write_text(
            "value = 2\n",
            encoding="utf-8",
        )

        (self.workspace / "temp.txt").write_text(
            "temporary\n",
            encoding="utf-8",
        )

        self.policy_path = self.root / "policy.json"

        self.policy_path.write_text(
            json.dumps({
                "commands": {
                    "allow": ["pwd"],
                    "deny": ["rm -rf ."],
                },
                "directories": {
                    "allow": ["."],
                    "deny": [],
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

        self.db_path = self.root / "runs.sqlite3"
        self.store = RunStore(self.db_path)

        self.addCleanup(
            lambda: self.store.close()
        )

    @staticmethod
    def tool_response(
        call_id,
        name,
        **arguments,
    ):
        return LLMResponse(
            content=None,
            tool_calls=(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                ),
            ),
        )

    def make_engine(
        self,
        responses,
        mode="safe",
        after_tool_execution=None,
    ):
        fake = FakeLLM(responses)

        engine = AgentEngine(
            llm=fake,
            registry=self.registry,
            executor=self.executor,
            store=self.store,
            mode=mode,
            after_tool_execution=after_tool_execution,
        )

        return fake, engine

    def restart_store(self):
        self.store.close()
        self.store = RunStore(self.db_path)

    def test_safe_write_requires_approval(self):
        fake, engine = self.make_engine([
            self.tool_response(
                "call_1",
                "write_file",
                path="order.py",
                content="value = 100\n",
            ),
            LLMResponse(content="готово"),
        ])

        result = engine.run("измени файл")

        self.assertEqual(
            result.status,
            "waiting_approval",
        )

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 1\n",
        )

        self.assertEqual(len(fake.calls), 1)

        approval = self.store.get_pending_approval(
            result.run_id
        )

        self.assertEqual(
            approval["name"],
            "write_file",
        )

        self.assertEqual(
            self.store.get_approval(
                result.run_id,
                "call_1",
            ),
            "pending",
        )

        completed = engine.approve(result.run_id)

        self.assertEqual(
            completed.status,
            "completed",
        )

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 100\n",
        )

        self.assertEqual(
            self.store.get_approval(
                result.run_id,
                "call_1",
            ),
            "approved",
        )

    def test_reject_sends_error_to_model(self):
        fake, engine = self.make_engine([
            self.tool_response(
                "call_1",
                "write_file",
                path="order.py",
                content="value = 200\n",
            ),
            LLMResponse(
                content="изменение отклонено пользователем"
            ),
        ])

        result = engine.run("измени файл")

        completed = engine.reject(
            result.run_id
        )

        self.assertEqual(
            completed.status,
            "completed",
        )

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 1\n",
        )

        messages = fake.calls[1]["messages"]

        tool_message = messages[-1]

        self.assertEqual(
            tool_message["role"],
            "tool",
        )

        self.assertEqual(
            tool_message["tool_call_id"],
            "call_1",
        )

        content = json.loads(
            tool_message["content"]
        )

        self.assertFalse(content["success"])

        self.assertIn(
            "пользователь отклонил",
            content["error"],
        )

        self.assertEqual(
            self.store.get_approval(
                result.run_id,
                "call_1",
            ),
            "rejected",
        )

    def test_auto_write_without_approval(self):
        fake, engine = self.make_engine(
            [
                self.tool_response(
                    "call_1",
                    "write_file",
                    path="order.py",
                    content="value = 300\n",
                ),
                LLMResponse(content="готово"),
            ],
            mode="auto",
        )

        result = engine.run("измени файл")

        self.assertEqual(
            result.status,
            "completed",
        )

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 300\n",
        )

        self.assertIsNone(
            self.store.get_approval(
                result.run_id,
                "call_1",
            )
        )

    def test_safe_bash_requires_approval(self):
        fake, engine = self.make_engine([
            self.tool_response(
                "call_1",
                "Bash",
                command="pwd",
                cwd=".",
            ),
            LLMResponse(content="команда выполнена"),
        ])

        with patch.object(
            self.executor,
            "execute",
            wraps=self.executor.execute,
        ) as execute:
            result = engine.run("выполни pwd")

            execute.assert_not_called()

            self.assertEqual(
                result.status,
                "waiting_approval",
            )

            completed = engine.approve(
                result.run_id
            )

            self.assertEqual(
                execute.call_count,
                1,
            )

        self.assertEqual(
            completed.status,
            "completed",
        )

        tool_messages = [
            message
            for message in completed.messages
            if message["role"] == "tool"
        ]

        content = json.loads(
            tool_messages[0]["content"]
        )

        self.assertTrue(content["success"])

        self.assertEqual(
            content["data"]["exit_code"],
            0,
        )

    def test_auto_bash_without_approval(self):
        fake, engine = self.make_engine(
            [
                self.tool_response(
                    "call_1",
                    "Bash",
                    command="pwd",
                    cwd=".",
                ),
                LLMResponse(content="готово"),
            ],
            mode="auto",
        )

        result = engine.run("выполни pwd")

        self.assertEqual(
            result.status,
            "completed",
        )

        self.assertIsNone(
            self.store.get_approval(
                result.run_id,
                "call_1",
            )
        )

    def test_delete_requires_approval_in_auto(self):
        fake, engine = self.make_engine(
            [
                self.tool_response(
                    "call_1",
                    "delete_file",
                    path="temp.txt",
                ),
                LLMResponse(content="файл удалён"),
            ],
            mode="auto",
        )

        result = engine.run("удали файл")

        self.assertEqual(
            result.status,
            "waiting_approval",
        )

        self.assertTrue(
            (self.workspace / "temp.txt").exists()
        )

        completed = engine.approve(
            result.run_id
        )

        self.assertEqual(
            completed.status,
            "completed",
        )

        self.assertFalse(
            (self.workspace / "temp.txt").exists()
        )

    def test_delete_requires_approval_in_safe(self):
        fake, engine = self.make_engine([
            self.tool_response(
                "call_1",
                "delete_file",
                path="temp.txt",
            ),
            LLMResponse(content="файл удалён"),
        ])

        result = engine.run("удали файл")

        self.assertEqual(
            result.status,
            "waiting_approval",
        )

        self.assertTrue(
            (self.workspace / "temp.txt").exists()
        )

        completed = engine.approve(
            result.run_id
        )

        self.assertEqual(
            completed.status,
            "completed",
        )

        self.assertFalse(
            (self.workspace / "temp.txt").exists()
        )

    def test_policy_denial_needs_no_approval(self):
        fake, engine = self.make_engine([
            self.tool_response(
                "call_1",
                "write_file",
                path="test_order.py",
                content="changed",
            ),
            LLMResponse(content="изменение запрещено"),
        ])

        result = engine.run(
            "измени защищённый файл"
        )

        self.assertEqual(
            result.status,
            "completed",
        )

        self.assertEqual(
            (self.workspace / "test_order.py").read_text(),
            "value = 2\n",
        )

        self.assertIsNone(
            self.store.get_approval(
                result.run_id,
                "call_1",
            )
        )

        messages = fake.calls[1]["messages"]

        content = json.loads(
            messages[-1]["content"]
        )

        self.assertFalse(content["success"])

        self.assertIn(
            "изменение файла запрещено",
            content["error"],
        )

    def test_approval_survives_restart(self):
        fake, engine = self.make_engine([
            self.tool_response(
                "call_1",
                "write_file",
                path="order.py",
                content="value = 400\n",
            ),
        ])

        result = engine.run("измени файл")

        self.assertEqual(
            result.status,
            "waiting_approval",
        )

        run_id = result.run_id

        self.restart_store()

        new_fake, new_engine = self.make_engine([
            LLMResponse(content="готово"),
        ])

        restored = new_engine.resume(run_id)

        self.assertEqual(
            restored.status,
            "waiting_approval",
        )

        self.assertEqual(len(new_fake.calls), 0)

        completed = new_engine.approve(run_id)

        self.assertEqual(
            completed.status,
            "completed",
        )

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 400\n",
        )

    def test_rejection_survives_restart(self):
        fake, engine = self.make_engine([
            self.tool_response(
                "call_1",
                "write_file",
                path="order.py",
                content="value = 500\n",
            ),
        ])

        result = engine.run("измени файл")

        run_id = result.run_id

        self.restart_store()

        new_fake, new_engine = self.make_engine([
            LLMResponse(content="отказ принят"),
        ])

        completed = new_engine.reject(run_id)

        self.assertEqual(
            completed.status,
            "completed",
        )

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 1\n",
        )

        self.assertEqual(
            self.store.get_approval(
                run_id,
                "call_1",
            ),
            "rejected",
        )

        decisions = self.store.get_decisions(run_id)

        self.assertEqual(
            decisions[0]["action"],
            "reject",
        )

    def test_multiple_tool_calls_with_approvals(self):
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
                        name="write_file",
                        arguments={
                            "path": "order.py",
                            "content": "value = 600\n",
                        },
                    ),
                    ToolCall(
                        id="call_3",
                        name="Bash",
                        arguments={
                            "command": "pwd",
                            "cwd": ".",
                        },
                    ),
                ),
            ),
            LLMResponse(content="готово"),
        ]

        fake, engine = self.make_engine(responses)

        result = engine.run("выполни задачу")

        self.assertEqual(
            result.status,
            "waiting_approval",
        )

        self.assertEqual(
            result.tool_calls,
            1,
        )

        first_approval = (
            self.store.get_pending_approval(
                result.run_id
            )
        )

        self.assertEqual(
            first_approval["name"],
            "write_file",
        )

        result = engine.approve(result.run_id)

        self.assertEqual(
            result.status,
            "waiting_approval",
        )

        second_approval = (
            self.store.get_pending_approval(
                result.run_id
            )
        )

        self.assertEqual(
            second_approval["name"],
            "Bash",
        )

        result = engine.approve(result.run_id)

        self.assertEqual(
            result.status,
            "completed",
        )

        self.assertEqual(
            result.tool_calls,
            3,
        )

        tool_messages = [
            message
            for message in result.messages
            if message["role"] == "tool"
        ]

        self.assertEqual(
            [
                message["tool_call_id"]
                for message in tool_messages
            ],
            ["call_1", "call_2", "call_3"],
        )

    def test_approve_without_pending_action(self):
        fake, engine = self.make_engine([
            LLMResponse(content="готово"),
        ])

        result = engine.run("привет")

        with self.assertRaisesRegex(
            StoreError,
            "не ожидает подтверждения",
        ):
            engine.approve(result.run_id)

    def test_mode_is_persisted(self):
        fake, engine = self.make_engine([
            self.tool_response(
                "call_1",
                "write_file",
                path="order.py",
                content="value = 700\n",
            ),
        ])

        result = engine.run("измени файл")

        self.assertEqual(
            self.store.get_run(result.run_id)["mode"],
            "safe",
        )

        self.restart_store()

        new_fake, new_engine = self.make_engine(
            [LLMResponse(content="готово")],
            mode="auto",
        )

        restored = new_engine.resume(
            result.run_id
        )

        self.assertEqual(
            restored.status,
            "waiting_approval",
        )

        completed = new_engine.approve(
            result.run_id
        )

        self.assertEqual(
            completed.status,
            "completed",
        )

    def test_crash_after_approval_is_uncertain(self):
        def crash(call, result):
            raise SimulatedCrash()

        fake, engine = self.make_engine(
            [
                self.tool_response(
                    "call_1",
                    "write_file",
                    path="order.py",
                    content="value = 800\n",
                ),
            ],
            after_tool_execution=crash,
        )

        result = engine.run("измени файл")

        self.assertEqual(
            result.status,
            "waiting_approval",
        )

        with self.assertRaises(SimulatedCrash):
            engine.approve(result.run_id)

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 800\n",
        )

        self.restart_store()

        new_fake, new_engine = self.make_engine([
            LLMResponse(content="работа продолжена"),
        ])

        restored = new_engine.resume(
            result.run_id
        )

        self.assertEqual(
            restored.status,
            "waiting_resolution",
        )

        self.assertEqual(len(new_fake.calls), 0)

        call = self.store.get_tool_call(
            result.run_id,
            "call_1",
        )

        self.assertEqual(
            call["status"],
            "uncertain",
        )


if __name__ == "__main__":
    unittest.main()