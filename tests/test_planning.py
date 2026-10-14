import io
import json
import sqlite3
import tempfile
import unittest

from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from agent_my.cli import main
from agent_my.engine import AgentEngine
from agent_my.llm import FakeLLM
from agent_my.models import LLMResponse, ToolCall
from agent_my.runtime.executor import Executor
from agent_my.runtime.policy import Policy
from agent_my.runtime.planning import PLAN_SCHEMA, PLANNING_PROMPT, PLAN_RETRY_MESSAGE
from agent_my.runtime.store import RunStore, StoreError
from agent_my.tools import build_tool_registry


class SimulatedCrash(BaseException):
    pass


class PlanningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.original = "def calculate_total(prices, discount):\n    return sum(prices) - discount\n"
        self.corrected = (
            "def calculate_total(prices, discount):\n"
            "    if not 0 <= discount <= 100:\n"
            "        raise ValueError('некорректная скидка')\n"
            "    return sum(prices) * (1 - discount / 100)\n"
        )
        (self.workspace / "order.py").write_text(self.original, encoding="utf-8")
        (self.workspace / "test_order.py").write_text(
            "import unittest\nfrom order import calculate_total\n"
            "class OrderTest(unittest.TestCase):\n"
            "    def test_discount(self):\n"
            "        self.assertEqual(calculate_total([100, 50], 10), 135)\n"
            "    def test_invalid_discount(self):\n"
            "        for value in (-1, 101):\n"
            "            with self.assertRaises(ValueError):\n"
            "                calculate_total([100], value)\n",
            encoding="utf-8",
        )
        self.policy_path = self.root / "policy.json"
        self.policy_path.write_text(json.dumps({
            "commands": {"allow": ["pwd", "python3 -m unittest -v"], "deny": []},
            "directories": {"allow": ["."], "deny": []},
            "files": {"deny": ["test_order.py"]},
        }), encoding="utf-8")
        self.policy = Policy(self.policy_path, self.workspace)
        self.registry = build_tool_registry(self.workspace, self.policy)
        self.executor = Executor(self.registry, self.policy)
        self.db_path = self.root / "runs.sqlite3"
        self.store = RunStore(self.db_path)
        self.addCleanup(lambda: self.store.close())
        self.plan = {
            "goal": "исправить расчёт стоимости",
            "steps": [
                {"description": "прочитать исходник и тесты", "tools": ["read_file"]},
                {"description": "исправить расчёт и проверку скидки", "tools": ["write_file"]},
                {"description": "проверить результат", "tools": ["Bash"]},
            ],
            "risks": ["можно пропустить граничный случай"],
            "assumptions": ["скидка задаётся в процентах"],
            "done_when": "оба теста проходят, тестовый файл не изменён",
        }

    def engine(self, responses, mode="safe", max_steps=20, after_tool_execution=None):
        fake = FakeLLM(responses)
        engine = AgentEngine(
            fake, self.registry, self.executor, store=self.store,
            mode=mode, max_steps=max_steps, verify_tests=True,
            after_tool_execution=after_tool_execution,
        )
        return fake, engine

    def restart(self):
        self.store.close()
        self.store = RunStore(self.db_path)

    def plan_response(self, call_id="plan", plan=None):
        return LLMResponse(None, (ToolCall(call_id, "submit_plan", self.plan if plan is None else plan),))

    def execution_responses(self):
        return [
            LLMResponse(None, (ToolCall("execution_read", "read_file", {"path": "order.py"}),)),
            LLMResponse(None, (ToolCall("done_read", "complete_task", {
                "task_id": "task_1", "evidence": ["execution_read"],
            }),)),
            LLMResponse(None, (ToolCall("write", "write_file", {
                "path": "order.py", "content": self.corrected,
            }),)),
            LLMResponse(None, (ToolCall("done_write", "complete_task", {
                "task_id": "task_2", "evidence": ["write"],
            }),)),
            LLMResponse(None, (ToolCall("test", "Bash", {
                "command": "python3 -m unittest -v", "cwd": ".",
            }),)),
            LLMResponse(None, (ToolCall("done_test", "complete_task", {
                "task_id": "task_3", "evidence": ["test"],
            }),)),
            LLMResponse("тесты проходят"),
        ]

    def tree(self):
        return {p.name: p.read_bytes() for p in self.workspace.iterdir() if p.is_file()}

    def test_planning_blocks_all_project_side_effects(self):
        original = self.tree()
        for mode in ("safe", "auto"):
            with self.subTest(mode=mode):
                fake, engine = self.engine([
                    LLMResponse(None, (ToolCall("read", "read_file", {"path": "order.py"}),)),
                    LLMResponse(None, (
                        ToolCall("write", "write_file", {"path": "order.py", "content": "changed"}),
                        ToolCall("delete", "delete_file", {"path": "order.py"}),
                        ToolCall("bash", "Bash", {"command": "pwd", "cwd": "."}),
                    )),
                    self.plan_response(),
                ], mode=mode)
                with patch.object(self.registry, "execute", wraps=self.registry.execute) as execute:
                    result = engine.run("исправь расчёт", plan=True)

                self.assertEqual(result.status, "waiting_plan")
                self.assertEqual(self.tree(), original)
                self.assertEqual([c.args[0].name for c in execute.call_args_list], ["read_file", "submit_plan"])
                self.assertIsNone(self.store.get_pending_approval(result.run_id))
                errors = fake.calls[2]["messages"][-3:]
                self.assertTrue(all(not json.loads(m["content"])["success"] for m in errors))
                self.assertEqual(
                    {t["function"]["name"] for t in fake.calls[0]["tools"]},
                    {"list_files", "read_file", "submit_plan"},
                )

    def test_executor_cannot_bypass_planning_with_approval(self):
        for name, arguments in (
            ("write_file", {"path": "order.py", "content": "changed"}),
            ("delete_file", {"path": "order.py"}),
            ("Bash", {"command": "pwd", "cwd": "."}),
        ):
            with self.subTest(name=name):
                with patch.object(self.registry, "execute") as execute:
                    result = self.executor.execute(
                        ToolCall("blocked", name, arguments), mode="auto", approved=True, phase="planning",
                    )
                self.assertFalse(result.success)
                self.assertIn("до утверждения плана", result.error)
                execute.assert_not_called()

    def test_plan_survives_restart_and_resume_waits(self):
        _, engine = self.engine([self.plan_response()])
        result = engine.run("исправь расчёт", plan=True)
        self.restart()
        fake, engine = self.engine([])

        restored = engine.resume(result.run_id)

        self.assertEqual(restored.status, "waiting_plan")
        self.assertEqual(self.store.get_plan(result.run_id), self.plan)
        self.assertEqual(restored.model_calls, 1)
        self.assertEqual(fake.calls, [])
        with self.assertRaises(StoreError):
            engine.approve(result.run_id)

    def test_safe_execution_after_plan_approval_requires_each_confirmation(self):
        tests = (self.workspace / "test_order.py").read_bytes()
        _, engine = self.engine([self.plan_response()])
        result = engine.run("исправь расчёт", plan=True)
        self.restart()
        fake, engine = self.engine(self.execution_responses())

        result = engine.approve_plan(result.run_id)

        self.assertEqual(result.status, "waiting_approval")
        self.assertEqual((self.workspace / "order.py").read_text(), self.original)
        self.assertEqual(self.store.get_pending_approval(result.run_id)["name"], "write_file")
        self.assertNotIn("submit_plan", {t["function"]["name"] for t in fake.calls[0]["tools"]})
        with self.assertRaises(StoreError):
            engine.approve_plan(result.run_id)

        result = engine.approve(result.run_id)

        self.assertEqual(result.status, "waiting_approval")
        self.assertEqual(self.store.get_pending_approval(result.run_id)["name"], "Bash")
        self.assertEqual((self.workspace / "order.py").read_text(), self.corrected)

        result = engine.approve(result.run_id)

        self.assertEqual(result.status, "completed")
        self.assertTrue(engine._tests_verified(result.messages))
        self.assertEqual(result.model_calls, 8)
        self.assertEqual((self.workspace / "test_order.py").read_bytes(), tests)
        self.assertEqual(self.store.get_run(result.run_id)["phase"], "execution")
        self.assertEqual(self.store.get_decisions(result.run_id)[0]["action"], "approve_plan")

    def test_auto_still_waits_for_plan_approval(self):
        original = self.tree()
        fake, engine = self.engine([self.plan_response(), *self.execution_responses()], mode="auto")
        result = engine.run("исправь расчёт", plan=True)
        self.assertEqual(self.tree(), original)
        self.assertEqual(len(fake.calls), 1)

        result = engine.approve_plan(result.run_id)

        self.assertEqual(result.status, "completed")
        self.assertTrue(engine._tests_verified(result.messages))
        self.assertEqual(self.store.get_plan(result.run_id), self.plan)

    def test_reject_plan_is_persisted_and_never_executes(self):
        original = self.tree()
        _, engine = self.engine([self.plan_response()])
        result = engine.run("исправь расчёт", plan=True)
        engine.reject_plan(result.run_id)
        self.restart()
        fake, engine = self.engine([])

        result = engine.resume(result.run_id)

        self.assertEqual(result.status, "plan_rejected")
        self.assertEqual(self.tree(), original)
        self.assertEqual(fake.calls, [])
        self.assertEqual(self.store.get_decisions(result.run_id)[0]["action"], "reject_plan")
        with self.assertRaises(StoreError):
            engine.approve_plan(result.run_id)

    def test_invalid_plan_is_returned_to_model(self):
        invalid = deepcopy(self.plan)
        del invalid["risks"]
        fake, engine = self.engine([
            self.plan_response("bad", invalid), self.plan_response("fixed"),
        ])

        result = engine.run("исправь расчёт", plan=True)

        self.assertEqual(result.status, "waiting_plan")
        self.assertEqual(self.store.get_plan(result.run_id), self.plan)
        error = fake.calls[1]["messages"][-1]
        self.assertEqual(error["tool_call_id"], "bad")
        self.assertFalse(json.loads(error["content"])["success"])

    def test_plan_cannot_queue_execution_in_same_batch(self):
        fake, engine = self.engine([
            LLMResponse(None, (
                ToolCall("batch_plan", "submit_plan", self.plan),
                ToolCall("hidden_write", "write_file", {"path": "order.py", "content": "changed"}),
            )),
            self.plan_response("separate_plan"),
            *self.execution_responses(),
        ], mode="auto")

        result = engine.run("исправь расчёт", plan=True)

        self.assertEqual(result.status, "waiting_plan")
        self.assertEqual((self.workspace / "order.py").read_text(), self.original)
        self.assertEqual(self.store.get_attempts(result.run_id, "hidden_write"), [])
        self.assertIn("отдельным вызовом", fake.calls[1]["messages"][-2]["content"])
        self.assertEqual(engine.approve_plan(result.run_id).status, "completed")

    def test_text_plan_cannot_complete_run(self):
        fake, engine = self.engine([LLMResponse("план: исправить и проверить")] * 3)

        result = engine.run("исправь расчёт", plan=True)

        self.assertEqual(result.status, "failed")
        self.assertIsNone(self.store.get_plan(result.run_id))
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(fake.calls[1]["response_schema"], PLAN_SCHEMA)
        self.assertIsNone(fake.calls[1]["tools"])

    def test_text_answer_falls_back_to_schema_and_waits_for_approval(self):
        original = self.tree()
        fake, engine = self.engine([
            LLMResponse("изменений не требуется, осталось проверить тесты"),
            LLMResponse(json.dumps(self.plan, ensure_ascii=False)),
        ])

        result = engine.run("проверь расчёт", plan=True)

        self.assertEqual(result.status, "waiting_plan")
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(self.tree(), original)
        self.assertEqual(self.store.get_plan(result.run_id), self.plan)
        self.assertEqual(fake.calls[1]["response_schema"], PLAN_SCHEMA)
        self.assertIsNone(fake.calls[1]["tools"])
        self.restart()
        fake, engine = self.engine(self.execution_responses())
        self.assertEqual(engine.resume(result.run_id).status, "waiting_plan")
        self.assertEqual(engine.approve_plan(result.run_id).status, "waiting_approval")
        self.assertEqual(self.tree(), original)
        self.assertIsNone(fake.calls[0]["response_schema"])
        self.assertNotIn(PLANNING_PROMPT, fake.calls[0]["messages"][0]["content"])
        self.assertFalse(any(m.get("content") == PLAN_RETRY_MESSAGE for m in fake.calls[0]["messages"]))
        saved = self.store.get_messages(result.run_id)
        self.assertIn(PLANNING_PROMPT, saved[0]["content"])
        self.assertTrue(any(m.get("content") == PLAN_RETRY_MESSAGE for m in saved))

    def test_structured_plan_is_validated_before_accepting(self):
        invalid = deepcopy(self.plan)
        invalid["steps"][0]["tools"] = []
        fake, engine = self.engine([
            LLMResponse("план готов"),
            LLMResponse('{"goal":'),
            LLMResponse(json.dumps(invalid)),
            LLMResponse(json.dumps(self.plan)),
        ])

        result = engine.run("проверь расчёт", plan=True)

        self.assertEqual(result.status, "waiting_plan")
        self.assertEqual(result.model_calls, 4)
        self.assertEqual(self.store.get_plan(result.run_id), self.plan)
        self.assertFalse(json.loads(fake.calls[3]["messages"][-1]["content"])["success"])
        self.assertTrue(all(call["response_schema"] == PLAN_SCHEMA for call in fake.calls[1:]))

    def test_structured_output_cannot_request_a_file_change(self):
        original = self.tree()
        fake, engine = self.engine([
            LLMResponse("план готов"),
            LLMResponse(json.dumps({
                "name": "write_file", "arguments": {"path": "order.py", "content": "changed"},
            })),
            LLMResponse(json.dumps(self.plan)),
        ])

        with patch.object(self.registry, "execute", wraps=self.registry.execute) as execute:
            result = engine.run("проверь расчёт", plan=True)

        self.assertEqual(result.status, "waiting_plan")
        self.assertEqual(self.tree(), original)
        self.assertEqual([call.args[0].name for call in execute.call_args_list], ["submit_plan"])
        self.assertFalse(json.loads(fake.calls[2]["messages"][-1]["content"])["success"])

    def test_plan_approval_does_not_reset_model_limit(self):
        fake, engine = self.engine([self.plan_response()], max_steps=1)
        result = engine.run("исправь расчёт", plan=True)

        result = engine.approve_plan(result.run_id)

        self.assertEqual(result.status, "limit_reached")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual((self.workspace / "order.py").read_text(), self.original)

    def test_crash_during_plan_submission_requires_explicit_retry(self):
        def crash(call, result):
            raise SimulatedCrash()

        _, engine = self.engine([self.plan_response()], after_tool_execution=crash)
        with self.assertRaises(SimulatedCrash):
            engine.run("исправь расчёт", run_id="crashed_plan", plan=True)
        self.restart()
        fake, engine = self.engine([])

        result = engine.resume("crashed_plan")

        self.assertEqual(result.status, "waiting_resolution")
        self.assertIsNone(self.store.get_plan(result.run_id))
        result = engine.resolve(result.run_id, "retry")
        self.assertEqual(result.status, "waiting_plan")
        self.assertEqual(self.store.get_plan(result.run_id), self.plan)
        self.assertEqual(len(self.store.get_attempts(result.run_id, "plan")), 2)
        self.assertEqual(fake.calls, [])

    def test_cli_plan_show_and_reject(self):
        _, engine = self.engine([self.plan_response()])
        output = io.StringIO()
        with patch("agent_my.cli.build_engine", return_value=engine):
            with redirect_stdout(output):
                code = main(["run", "исправь расчёт", "--plan", "--mode", "safe"])
        self.assertEqual(code, 3)
        self.assertIn("approve-plan", output.getvalue())
        run_id = self.store.db.execute("SELECT run_id FROM runs").fetchone()[0]
        output = io.StringIO()
        with patch("agent_my.cli.STATE_DB", self.db_path), redirect_stdout(output):
            self.assertEqual(main(["plan", run_id]), 0)
            self.assertEqual(main(["status", run_id]), 0)
        self.assertIn(self.plan["goal"], output.getvalue())
        self.assertIn("критерий готовности", output.getvalue())
        self.assertIn("планирование", output.getvalue())
        with patch("agent_my.cli.build_engine", return_value=engine), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["reject-plan", run_id]), 0)
        self.assertEqual(self.store.get_run(run_id)["status"], "plan_rejected")

    def test_existing_database_is_migrated(self):
        path = self.root / "old.sqlite3"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, status TEXT, max_steps INTEGER, model_calls INTEGER DEFAULT 0, tool_calls INTEGER DEFAULT 0, final_answer TEXT, mode TEXT DEFAULT 'auto')")
            db.execute("INSERT INTO runs (run_id, status, max_steps) VALUES ('old', 'completed', 20)")
        store = RunStore(path)
        self.addCleanup(store.close)
        self.assertEqual(store.get_run("old")["phase"], "execution")
        self.assertIsNone(store.get_plan("old"))


if __name__ == "__main__":
    unittest.main()
