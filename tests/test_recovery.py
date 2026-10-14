import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from agent_my.engine import AgentEngine
from agent_my.llm import FakeLLM, LLMError
from agent_my.models import LLMResponse, ToolCall
from agent_my.runtime.executor import Executor
from agent_my.runtime.policy import Policy
from agent_my.runtime.store import RunStore, StoreError
from agent_my.tools import build_tool_registry


class SimulatedCrash(BaseException):
    pass


class RecoveryTest(unittest.TestCase):
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

        self.addCleanup(lambda: self.store.close())

    def make_engine(
        self,
        responses,
        after_tool_execution=None,
        max_steps=20,
    ):
        fake = FakeLLM(responses)

        engine = AgentEngine(
            llm=fake,
            registry=self.registry,
            executor=self.executor,
            store=self.store,
            max_steps=max_steps,
            after_tool_execution=after_tool_execution,
        )

        return fake, engine

    def restart_store(self):
        self.store.close()
        self.store = RunStore(self.db_path)

    @staticmethod
    def make_call(call_id, name, **arguments):
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

    @staticmethod
    def crash(call, result):
        raise SimulatedCrash()

    def test_text_call_retry_survives_restart(self):
        response = LLMResponse(content='write_file("order.py", "value = 3")')
        fake, engine = self.make_engine([response])

        with self.assertRaises(LLMError):
            engine.run("измени файл", run_id="text_retry")

        self.assertEqual(self.store.get_run("text_retry")["status"], "running")
        self.assertEqual(self.store.get_run("text_retry")["model_calls"], 2)
        self.restart_store()
        fake, engine = self.make_engine([response, response])

        result = engine.resume("text_retry")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.model_calls, 4)
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["messages"][-1]["role"], "system")
        self.assertEqual(
            (self.workspace / "order.py").read_text(encoding="utf-8"),
            "value = 1\n",
        )

        self.restart_store()
        fake, engine = self.make_engine([])
        restored = engine.resume("text_retry")

        self.assertEqual(restored.status, "failed")
        self.assertEqual(restored.final_answer, result.final_answer)
        self.assertEqual(fake.calls, [])

    def test_text_call_retry_continues_with_real_call_after_restart(self):
        fake, engine = self.make_engine([
            LLMResponse(content='read_file("order.py")'),
        ])

        with self.assertRaises(LLMError):
            engine.run("прочитай файл", run_id="text_retry_success")

        self.restart_store()
        fake, engine = self.make_engine([
            self.make_call("read", "read_file", path="order.py"),
            LLMResponse(content="файл прочитан"),
        ])

        result = engine.resume("text_retry_success")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.model_calls, 4)
        self.assertEqual(result.tool_calls, 1)
        self.assertIn("value = 1", fake.calls[1]["messages"][-1]["content"])

    def test_test_verification_requirement_survives_restart(self):
        fake, engine = self.make_engine([LLMResponse(content="готово")])
        engine.verify_tests = True

        with self.assertRaises(LLMError):
            engine.run("исправь файл", run_id="verify_tests")

        self.restart_store()
        fake, engine = self.make_engine([LLMResponse(content="готово")] * 2)

        result = engine.resume("verify_tests")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.model_calls, 4)
        self.assertEqual(fake.calls[0]["tool_choice"], "required")

    def test_completed_run_is_persisted(self):
        fake, engine = self.make_engine([
            LLMResponse(content="задача завершена"),
        ])

        result = engine.run("проверь проект")

        run_id = result.run_id

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.model_calls, 1)

        self.restart_store()

        new_fake, new_engine = self.make_engine([])

        restored = new_engine.resume(run_id)

        self.assertEqual(
            restored.status,
            "completed",
        )

        self.assertEqual(
            restored.final_answer,
            "задача завершена",
        )

        self.assertEqual(len(new_fake.calls), 0)

        self.assertEqual(
            restored.messages[1]["content"],
            "проверь проект",
        )

    def test_completed_tool_is_not_repeated(self):
        responses = [
            self.make_call(
                "call_1",
                "write_file",
                path="order.py",
                content="value = 100\n",
            ),
        ]

        fake, engine = self.make_engine(responses)

        with self.assertRaises(LLMError):
            engine.run(
                "измени файл",
                run_id="test_completed",
            )

        call = self.store.get_tool_call(
            "test_completed",
            "call_1",
        )

        self.assertEqual(
            call["status"],
            "completed",
        )

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 100\n",
        )

        self.restart_store()

        new_fake, new_engine = self.make_engine([
            LLMResponse(content="изменение завершено"),
        ])

        with patch.object(
            self.executor,
            "execute",
            wraps=self.executor.execute,
        ) as execute:
            result = new_engine.resume(
                "test_completed"
            )

            execute.assert_not_called()

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.model_calls, 3)

        tool_messages = [
            item
            for item in result.messages
            if item["role"] == "tool"
        ]

        self.assertEqual(len(tool_messages), 1)

    def test_uncertain_write_requires_decision(self):
        responses = [
            self.make_call(
                "call_1",
                "write_file",
                path="order.py",
                content="value = 200\n",
            ),
        ]

        fake, engine = self.make_engine(
            responses,
            after_tool_execution=self.crash,
        )

        with self.assertRaises(SimulatedCrash):
            engine.run(
                "измени файл",
                run_id="test_uncertain",
            )

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 200\n",
        )

        call = self.store.get_tool_call(
            "test_uncertain",
            "call_1",
        )

        self.assertEqual(call["status"], "started")

        self.restart_store()

        new_fake, new_engine = self.make_engine([
            LLMResponse(content="работа продолжена"),
        ])

        with patch.object(
            self.executor,
            "execute",
            wraps=self.executor.execute,
        ) as execute:
            restored = new_engine.resume(
                "test_uncertain"
            )

            execute.assert_not_called()

        self.assertEqual(
            restored.status,
            "waiting_resolution",
        )

        call = self.store.get_tool_call(
            "test_uncertain",
            "call_1",
        )

        self.assertEqual(call["status"], "uncertain")
        self.assertEqual(len(new_fake.calls), 0)

        with self.assertRaisesRegex(
            StoreError,
            "небезопасно",
        ):
            new_engine.resolve(
                "test_uncertain",
                "retry",
            )

        result = new_engine.resolve(
            "test_uncertain",
            "skip",
        )

        self.assertEqual(result.status, "completed")

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 200\n",
        )

        self.assertEqual(
            result.tool_calls,
            1,
        )

        decisions = self.store.get_decisions(
            "test_uncertain"
        )

        self.assertEqual(len(decisions), 1)
        self.assertEqual(
            decisions[0]["action"],
            "skip",
        )

        tool_messages = [
            message
            for message in result.messages
            if message["role"] == "tool"
        ]

        self.assertEqual(len(tool_messages), 1)

        content = json.loads(
            tool_messages[0]["content"]
        )

        self.assertFalse(content["success"])

    def test_retry_read_creates_new_attempt(self):
        responses = [
            self.make_call(
                "call_1",
                "read_file",
                path="order.py",
            ),
        ]

        fake, engine = self.make_engine(
            responses,
            after_tool_execution=self.crash,
        )

        with self.assertRaises(SimulatedCrash):
            engine.run(
                "прочитай файл",
                run_id="test_retry",
            )

        self.restart_store()

        new_fake, new_engine = self.make_engine([
            LLMResponse(content="файл прочитан"),
        ])

        restored = new_engine.resume(
            "test_retry"
        )

        self.assertEqual(
            restored.status,
            "waiting_resolution",
        )

        result = new_engine.resolve(
            "test_retry",
            "retry",
        )

        self.assertEqual(result.status, "completed")

        self.assertEqual(
            result.tool_calls,
            2,
        )

        attempts = self.store.get_attempts(
            "test_retry",
            "call_1",
        )

        self.assertEqual(len(attempts), 2)

        self.assertEqual(
            attempts[0]["status"],
            "uncertain",
        )

        self.assertEqual(
            attempts[1]["status"],
            "completed",
        )

        decisions = self.store.get_decisions(
            "test_retry"
        )

        self.assertEqual(
            decisions[0]["action"],
            "retry",
        )

    def test_abort_run(self):
        fake, engine = self.make_engine(
            [
                self.make_call(
                    "call_1",
                    "delete_file",
                    path="order.py",
                ),
            ],
            after_tool_execution=self.crash,
        )

        result = engine.run(
            "удали файл",
            run_id="test_abort",
        )

        self.assertEqual(
            result.status,
            "waiting_approval",
        )

        self.assertTrue(
            (self.workspace / "order.py").exists()
        )

        with self.assertRaises(SimulatedCrash):
            engine.approve("test_abort")

        self.restart_store()

        new_fake, new_engine = self.make_engine([])

        restored = new_engine.resume(
            "test_abort"
        )

        self.assertEqual(
            restored.status,
            "waiting_resolution",
        )

        result = new_engine.resolve(
            "test_abort",
            "abort",
        )

        self.assertEqual(
            result.status,
            "failed",
        )

        self.assertEqual(len(new_fake.calls), 0)

        decisions = self.store.get_decisions(
            "test_abort"
        )

        self.assertEqual(
            decisions[-1]["action"],
            "abort",
        )

    def test_repeated_tool_call_id(self):
        call = self.make_call(
            "call_1",
            "write_file",
            path="order.py",
            content="value = 300\n",
        )

        fake, engine = self.make_engine([
            call,
            call,
            LLMResponse(content="готово"),
        ])

        with patch.object(
            self.executor,
            "execute",
            wraps=self.executor.execute,
        ) as execute:
            result = engine.run(
                "измени файл",
                run_id="test_duplicate",
            )

            self.assertEqual(
                execute.call_count,
                1,
            )

        self.assertEqual(
            result.status,
            "completed",
        )

        self.assertEqual(
            result.tool_calls,
            1,
        )

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 300\n",
        )

        messages = [
            item
            for item in result.messages
            if item["role"] == "tool"
        ]

        self.assertEqual(len(messages), 2)

        self.assertEqual(
            messages[0]["tool_call_id"],
            "call_1",
        )

        self.assertEqual(
            messages[1]["tool_call_id"],
            "call_1",
        )

    def test_repeated_id_with_other_arguments(self):
        responses = [
            self.make_call(
                "call_1",
                "write_file",
                path="order.py",
                content="value = 400\n",
            ),

            self.make_call(
                "call_1",
                "write_file",
                path="order.py",
                content="value = 500\n",
            ),

            LLMResponse(content="готово"),
        ]

        fake, engine = self.make_engine(responses)

        result = engine.run(
            "измени файл",
            run_id="test_protocol",
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_calls, 1)

        self.assertEqual(
            (self.workspace / "order.py").read_text(),
            "value = 400\n",
        )

        messages = [
            item
            for item in result.messages
            if item["role"] == "tool"
        ]

        content = json.loads(
            messages[-1]["content"]
        )

        self.assertFalse(content["success"])

        self.assertIn(
            "ошибка протокола",
            content["error"],
        )

    def test_repeated_id_with_other_name(self):
        fake, engine = self.make_engine([
            self.make_call("same", "read_file", path="order.py"),
            self.make_call("same", "delete_file", path="order.py"),
            LLMResponse(content="готово"),
        ])

        result = engine.run("прочитай файл")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual((self.workspace / "order.py").read_text(), "value = 1\n")
        self.assertIn("ошибка протокола", fake.calls[-1]["messages"][-1]["content"])

    def test_invalid_arguments_survive_restart(self):
        response = LLMResponse(content=None, tool_calls=(
            ToolCall("bad", "write_file", '{"path":"order.py",'),
        ))
        self.store.create_run("bad_arguments", "измени файл", "агент", 20, mode="auto")
        self.store.begin_model_request("bad_arguments")
        self.store.record_response("bad_arguments", response)
        self.restart_store()
        fake, engine = self.make_engine([
            self.make_call("fixed", "write_file", path="order.py", content="value = 2\n"),
            LLMResponse(content="готово"),
        ])

        result = engine.resume("bad_arguments")

        self.assertEqual(result.status, "completed")
        self.assertEqual((self.workspace / "order.py").read_text(), "value = 2\n")
        self.assertEqual(fake.calls[0]["messages"][-1]["tool_call_id"], "bad")
        self.assertIn("некорректный json", fake.calls[0]["messages"][-1]["content"])
        self.assertEqual(self.store.get_attempts("bad_arguments", "bad"), [])
        self.assertEqual(
            json.loads(self.store.get_tool_call("bad_arguments", "bad")["arguments"]),
            '{"path":"order.py",',
        )

    def test_batch_recovery(self):
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
                ),
            ),
        ]

        def crash_on_write(call, result):
            if call.name == "write_file":
                raise SimulatedCrash()

        fake, engine = self.make_engine(
            responses,
            after_tool_execution=crash_on_write,
        )

        with self.assertRaises(SimulatedCrash):
            engine.run(
                "прочитай и измени файл",
                run_id="test_batch",
            )

        first = self.store.get_tool_call(
            "test_batch",
            "call_1",
        )

        second = self.store.get_tool_call(
            "test_batch",
            "call_2",
        )

        self.assertEqual(
            first["status"],
            "completed",
        )

        self.assertEqual(
            second["status"],
            "started",
        )

        self.restart_store()

        new_fake, new_engine = self.make_engine([
            LLMResponse(content="работа продолжена"),
        ])

        restored = new_engine.resume("test_batch")

        self.assertEqual(
            restored.status,
            "waiting_resolution",
        )

        result = new_engine.resolve(
            "test_batch",
            "skip",
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_calls, 2)

        tool_messages = [
            message
            for message in result.messages
            if message["role"] == "tool"
        ]

        self.assertEqual(len(tool_messages), 2)

        self.assertEqual(
            [
                message["tool_call_id"]
                for message in tool_messages
            ],
            ["call_1", "call_2"],
        )

    def test_limit_survives_resume(self):
        responses = [
            self.make_call(
                "call_1",
                "read_file",
                path="order.py",
            ),
        ]

        fake, engine = self.make_engine(
            responses,
            max_steps=1,
        )

        result = engine.run(
            "прочитай файл",
            run_id="test_limit",
        )

        self.assertEqual(
            result.status,
            "limit_reached",
        )

        self.restart_store()

        new_fake, new_engine = self.make_engine(
            [],
            max_steps=100,
        )

        restored = new_engine.resume(
            "test_limit"
        )

        self.assertEqual(
            restored.status,
            "limit_reached",
        )

        self.assertEqual(
            restored.model_calls,
            1,
        )

        self.assertEqual(
            restored.tool_calls,
            1,
        )

        self.assertEqual(len(new_fake.calls), 0)

    def test_queued_call_after_restart(self):
        self.store.create_run(
            run_id="test_queued",
            prompt="прочитай файл",
            system_prompt="ты агент",
            max_steps=10,
        )

        self.store.begin_model_request("test_queued")
        self.store.record_response(
            "test_queued",
            self.make_call(
                "call_1",
                "read_file",
                path="order.py",
            ),
        )

        self.restart_store()

        fake, engine = self.make_engine([
            LLMResponse(content="готово"),
        ])

        result = engine.resume(
            "test_queued"
        )

        self.assertEqual(
            result.status,
            "completed",
        )

        self.assertEqual(
            result.tool_calls,
            1,
        )

        call = self.store.get_tool_call(
            "test_queued",
            "call_1",
        )

        self.assertEqual(
            call["status"],
            "completed",
        )


if __name__ == "__main__":
    unittest.main()
