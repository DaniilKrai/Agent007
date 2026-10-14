import io
import json
import tempfile
import unittest

from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_my.cli import main, print_tasks
from agent_my.engine import AgentEngine
from agent_my.llm import FakeLLM
from agent_my.models import LLMResponse, ToolCall
from agent_my.runtime.executor import Executor
from agent_my.runtime.policy import Policy
from agent_my.runtime.store import RunStore
from agent_my.tools import build_tool_registry


class SimulatedCrash(BaseException):
    pass


class TasksTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.broken = "def calculate_total(prices, discount):\n    return sum(prices) - discount\n"
        self.fixed = (
            "def calculate_total(prices, discount):\n"
            "    if not 0 <= discount <= 100:\n        raise ValueError('некорректная скидка')\n"
            "    return sum(prices) * (1 - discount / 100)\n"
        )
        (self.workspace / "order.py").write_text(self.broken)
        self.tests = (
            "import unittest\nfrom order import calculate_total\nclass TestOrder(unittest.TestCase):\n"
            "    def test_discount(self):\n        self.assertEqual(calculate_total([100, 50], 10), 135)\n"
            "    def test_invalid(self):\n        for discount in (-1, 101):\n"
            "            with self.assertRaises(ValueError):\n                calculate_total([100], discount)\n"
        )
        (self.workspace / "test_order.py").write_text(self.tests)
        policy_path = self.root / "policy.json"
        policy_path.write_text(json.dumps({
            "commands": {"allow": ["python3 -m unittest -v"], "deny": []},
            "directories": {"allow": ["."], "deny": []}, "files": {"deny": ["test_order.py"]},
        }))
        policy = Policy(policy_path, self.workspace)
        self.registry = build_tool_registry(self.workspace, policy)
        self.executor = Executor(self.registry, policy)
        self.db_path = self.root / "runs.sqlite3"
        self.store = RunStore(self.db_path)
        self.addCleanup(lambda: self.store.close())
        self.steps = [
            {"description": "прочитать исходник и тесты", "tools": ["read_file"]},
            {"description": "исправить расчёт", "tools": ["write_file"]},
            {"description": "проверить тесты", "tools": ["Bash"]},
        ]

    def engine(self, responses, mode="auto", **kwargs):
        fake = FakeLLM(responses)
        engine = AgentEngine(fake, self.registry, self.executor, store=self.store, mode=mode, **kwargs)
        return fake, engine

    @staticmethod
    def call(call_id, name, **arguments):
        return LLMResponse(None, (ToolCall(call_id, name, arguments),))

    def done(self, task, evidence, call_id=None):
        return self.call(call_id or "done_" + task, "complete_task", task_id=task, evidence=evidence)

    def plan(self, steps=None):
        return {
            "goal": "исправить расчёт", "steps": steps or self.steps,
            "risks": [], "assumptions": [], "done_when": "тесты проходят",
        }

    def start(self, steps=None, mode="auto"):
        fake, engine = self.engine([
            self.call("planning_read", "read_file", path="order.py"),
            LLMResponse(None, (ToolCall("plan", "submit_plan", self.plan(steps)),)),
        ], mode=mode)
        result = engine.run("исправь расчёт", plan=True)
        self.assertEqual(result.status, "waiting_plan")
        self.assertEqual(self.store.tasks.get_all(result.run_id), [])
        return result.run_id

    def responses(self):
        return [
            LLMResponse(None, (
                ToolCall("source", "read_file", {"path": "order.py"}),
                ToolCall("tests", "read_file", {"path": "test_order.py"}),
            )),
            self.done("task_1", ["source", "tests"]),
            self.call("write", "write_file", path="order.py", content=self.fixed),
            self.done("task_2", ["write"]),
            self.call("test", "Bash", command="python3 -m unittest -v", cwd="."),
            self.done("task_3", ["test"]),
            LLMResponse("тесты проходят"),
        ]

    def restart(self):
        self.store.close()
        self.store = RunStore(self.db_path)

    def test_missing_bash_cwd_can_be_corrected_then_approved_after_restart(self):
        (self.workspace / "order.py").write_text(self.fixed)
        run_id = self.start(steps=[self.steps[2]], mode="safe")
        fake, engine = self.engine([
            self.call("missing_cwd", "Bash", command="python3 -m unittest -v"),
            self.call("test", "Bash", command="python3 -m unittest -v", cwd="."),
        ], mode="safe")
        result = engine.approve_plan(run_id)
        self.assertEqual(result.status, "waiting_approval")
        calls = self.store.tasks.calls(run_id, "task_1")
        invalid = next(call for call in calls if call["call_id"] == "missing_cwd")
        self.assertEqual(invalid["attempt_count"], 0)
        self.assertIn("отсутствует обязательный аргумент: cwd", invalid["result"])
        self.assertEqual(self.store.get_pending_approval(run_id)["call_id"], "test")
        self.restart()
        fake, engine = self.engine([
            self.done("task_1", ["test"]), LLMResponse("тесты проходят"),
        ])
        result = engine.approve(run_id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.statuses(run_id), ["completed"])
        calls = self.store.tasks.calls(run_id, "task_1")
        test = next(call for call in calls if call["call_id"] == "test")
        self.assertEqual(test["attempt_count"], 1)
        self.assertEqual(json.loads(test["result"])["data"]["exit_code"], 0)

    def statuses(self, run_id):
        return [task["status"] for task in self.store.tasks.get_all(run_id)]

    def test_three_dependent_tasks_survive_restart_with_real_evidence_and_checklist(self):
        run_id = self.start(mode="safe")
        output = io.StringIO()
        fake, engine = self.engine(self.responses()[:3], mode="safe",
                                   on_tasks_changed=lambda key: print_tasks(self.store, key))
        with redirect_stdout(output):
            result = engine.approve_plan(run_id)
        self.assertEqual(result.status, "waiting_approval")
        self.assertEqual(self.statuses(run_id), ["completed", "running", "pending"])
        self.assertEqual((self.workspace / "order.py").read_text(), self.broken)
        self.assertIn("[x] task_1", output.getvalue())
        self.assertIn("[>] task_2", output.getvalue())
        self.assertIn("[ ] task_3", output.getvalue())
        spent = self.store.get_budget(run_id)
        self.restart()
        fake, engine = self.engine(self.responses()[3:5])
        self.assertEqual(engine.resume(run_id).status, "waiting_approval")
        self.assertEqual(self.store.get_budget(run_id), spent)
        result = engine.approve(run_id)
        self.assertEqual(result.status, "waiting_approval")
        self.assertEqual(self.statuses(run_id), ["completed", "completed", "running"])
        self.assertEqual((self.workspace / "order.py").read_text(), self.fixed)
        self.restart()
        fake, engine = self.engine(self.responses()[5:])
        result = engine.approve(run_id)
        self.assertEqual(result.status, "completed")
        self.assertTrue(engine._tests_verified(result.messages))
        self.assertEqual(self.statuses(run_id), ["completed"] * 3)
        tasks = self.store.tasks.get_all(run_id)
        self.assertEqual([t["dependencies"] for t in tasks], [[], ["task_1"], ["task_2"]])
        self.assertEqual([t["evidence"] for t in tasks], [["source", "tests"], ["write"], ["test"]])
        for call_id, task_id in (("planning_read", None), ("plan", None), ("source", "task_1"), ("write", "task_2"), ("test", "task_3")):
            self.assertEqual(self.store.get_tool_call(run_id, call_id)["task_id"], task_id)
            self.assertEqual(len(self.store.get_attempts(run_id, call_id)), 1)
        self.assertEqual((self.workspace / "test_order.py").read_text(), self.tests)
        self.restart()
        fake, engine = self.engine([])
        self.assertEqual(engine.resume(run_id).status, "completed")
        self.assertEqual(fake.calls, [])
        output = io.StringIO()
        with patch("agent_my.cli.STATE_DB", self.db_path), redirect_stdout(output):
            self.assertEqual(main(["tasks", run_id]), 0)
        self.assertIn("[x] task_3", output.getvalue())
        self.assertIn("зависимости: task_2", output.getvalue())
        self.assertIn("test: Bash, успешно, код завершения 0", output.getvalue())

    def test_model_cannot_start_later_task_or_finish_it_early(self):
        run_id = self.start()
        fake, engine = self.engine([
            self.call("early_write", "write_file", path="order.py", content=self.fixed),
            self.done("task_3", ["imaginary"], "early_done"),
            *self.responses(),
        ])
        result = engine.approve_plan(run_id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.store.get_attempts(run_id, "early_write"), [])
        self.assertEqual(self.store.get_attempts(run_id, "early_done"), [])
        self.assertIn("текущей задаче", fake.calls[1]["messages"][-2]["content"])

    def test_runtime_checks_dependencies_again_before_execution(self):
        run_id = self.start()
        self.store.decide_plan(run_id, "approve")
        with self.store.db:
            self.store.db.execute("UPDATE tasks SET status = 'running' WHERE run_id = ? AND task_id = 'task_2'", (run_id,))
        self.assertIn("зависимости", self.store.tasks.check_call(run_id, "task_2", "write_file"))
        self.assertEqual((self.workspace / "order.py").read_text(), self.broken)

    def test_text_claim_does_not_complete_task(self):
        run_id = self.start()
        fake, engine = self.engine([LLMResponse("всё выполнено")] * 3)
        result = engine.approve_plan(run_id)
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.statuses(run_id), ["failed", "pending", "pending"])
        self.assertEqual(self.store.tasks.get_all(run_id)[0]["evidence"], [])
        self.assertEqual((self.workspace / "order.py").read_text(), self.broken)

    def test_planning_and_foreign_task_evidence_are_rejected(self):
        steps = [self.steps[0], self.steps[0]]
        run_id = self.start(steps)
        fake, engine = self.engine([
            self.done("task_1", ["planning_read"], "bad_planning"),
            self.call("first", "read_file", path="order.py"),
            self.done("task_1", ["first"]),
            self.done("task_2", ["first"], "bad_foreign"),
            self.call("first", "read_file", path="order.py"),
            self.call("second", "read_file", path="order.py"),
            self.done("task_2", ["second"]), LLMResponse("готово"),
        ])
        result = engine.approve_plan(run_id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.store.get_attempts(run_id, "bad_planning"), [])
        self.assertEqual(self.store.get_attempts(run_id, "bad_foreign"), [])
        self.assertEqual(len(self.store.get_attempts(run_id, "first")), 1)
        self.assertEqual(self.store.tasks.get_all(run_id)[1]["evidence"], ["second"])

    def test_complete_task_in_batch_cannot_unlock_later_write(self):
        run_id = self.start()
        responses = self.responses()
        fake, engine = self.engine([
            responses[0],
            LLMResponse(None, (
                ToolCall("mixed_done", "complete_task", {"task_id": "task_1", "evidence": ["source"]}),
                ToolCall("mixed_write", "write_file", {"path": "order.py", "content": self.fixed}),
            )),
            *responses[1:],
        ])
        result = engine.approve_plan(run_id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.store.get_attempts(run_id, "mixed_write"), [])
        self.assertEqual(self.store.get_attempts(run_id, "mixed_done"), [])

    def test_failed_bash_cannot_prove_task(self):
        run_id = self.start([self.steps[2]])
        fake, engine = self.engine([
            self.call("bad_test", "Bash", command="python3 -m unittest -v", cwd="."),
            self.done("task_1", ["bad_test"]), *([LLMResponse("готово")] * 3),
        ])
        result = engine.approve_plan(run_id)
        self.assertIn("python3 -m unittest -v", fake.calls[0]["messages"][-1]["content"])
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.store.tasks.get_all(run_id)[0]["evidence"], [])
        self.assertEqual(self.store.get_attempts(run_id, "done_task_1"), [])

    def test_truncated_result_cannot_prove_task(self):
        (self.workspace / "order.py").write_text("x" * 70000)
        run_id = self.start([self.steps[0]])
        fake, engine = self.engine([
            self.call("large", "read_file", path="order.py"),
            self.done("task_1", ["large"]), *([LLMResponse("готово")] * 3),
        ])
        result = engine.approve_plan(run_id)
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.store.get_attempts(run_id, "done_task_1"), [])

    def test_test_evidence_before_last_write_is_not_sufficient(self):
        (self.workspace / "order.py").write_text(self.fixed)
        run_id = self.start([{"description": "исправить и проверить", "tools": ["write_file", "Bash"]}])
        fake, engine = self.engine([
            self.call("old_test", "Bash", command="python3 -m unittest -v", cwd="."),
            self.call("broken_write", "write_file", path="order.py", content=self.broken),
            self.done("task_1", ["old_test", "broken_write"], "stale_done"),
            self.call("fixed_write", "write_file", path="order.py", content=self.fixed),
            self.call("new_test", "Bash", command="python3 -m unittest -v", cwd="."),
            self.done("task_1", ["fixed_write", "new_test"]), LLMResponse("готово"),
        ])
        result = engine.approve_plan(run_id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.store.get_attempts(run_id, "stale_done"), [])
        self.assertEqual(self.store.tasks.get_all(run_id)[0]["evidence"], ["fixed_write", "new_test"])

    def test_completed_task_is_not_repeated_after_crash(self):
        run_id = self.start()

        def crash_after_completion(call, result):
            if call.name == "complete_task":
                raise SimulatedCrash()

        fake, engine = self.engine(self.responses()[:2], on_tool_result=crash_after_completion)
        with self.assertRaises(SimulatedCrash):
            engine.approve_plan(run_id)
        self.restart()
        self.assertEqual(self.statuses(run_id), ["completed", "pending", "pending"])
        fake, engine = self.engine(self.responses()[2:])
        result = engine.resume(run_id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(self.store.get_attempts(run_id, "source")), 1)
        self.assertEqual(len(self.store.get_attempts(run_id, "done_task_1")), 1)

    def test_crash_before_task_commit_keeps_dependencies_blocked(self):
        run_id = self.start()

        def crash(call, result):
            if call.name == "complete_task":
                raise SimulatedCrash()

        fake, engine = self.engine(self.responses()[:2], after_tool_execution=crash)
        with self.assertRaises(SimulatedCrash):
            engine.approve_plan(run_id)
        self.restart()
        fake, engine = self.engine(self.responses()[2:])
        self.assertEqual(engine.resume(run_id).status, "waiting_resolution")
        self.assertEqual(self.statuses(run_id), ["running", "pending", "pending"])
        self.assertEqual((self.workspace / "order.py").read_text(), self.broken)
        result = engine.resolve(run_id, "retry")
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(self.store.get_attempts(run_id, "done_task_1")), 2)

    def test_without_plan_calls_still_have_task_id_and_evidence(self):
        fake, engine = self.engine([self.call("read", "read_file", path="order.py"), LLMResponse("прочитано")])
        result = engine.run("прочитай файл")
        self.assertEqual(self.store.get_tool_call(result.run_id, "read")["task_id"], "task_1")
        self.assertEqual(self.store.tasks.get_all(result.run_id)[0]["evidence"], ["read"])

    def test_text_completion_uses_schema_only_when_evidence_exists(self):
        run_id = self.start([self.steps[0]])
        fake, engine = self.engine([
            self.call("read", "read_file", path="order.py"),
            LLMResponse("файл прочитан"),
            LLMResponse(json.dumps({"task_id": "task_1", "evidence": ["read"]})),
            LLMResponse("готово"),
        ])
        result = engine.approve_plan(run_id)
        self.assertEqual(result.status, "completed")
        self.assertIsNone(fake.calls[2]["tools"])
        schema = fake.calls[2]["response_schema"]
        self.assertEqual(schema["properties"]["task_id"]["enum"], ["task_1"])
        self.assertEqual(schema["properties"]["evidence"]["items"]["enum"], ["read"])
        self.assertIsNone(fake.calls[3]["response_schema"])
        self.assertIn("обычным текстом", fake.calls[3]["messages"][-1]["content"])
        self.assertEqual(self.store.tasks.get_all(run_id)[0]["evidence"], ["read"])

    def test_schema_cannot_complete_a_future_task_or_use_forged_evidence(self):
        run_id = self.start([self.steps[0]])
        fake, engine = self.engine([
            self.call("read", "read_file", path="order.py"), LLMResponse("готово"),
            LLMResponse(json.dumps({"task_id": "task_2", "evidence": ["read"]})),
            LLMResponse(json.dumps({"task_id": "task_1", "evidence": ["made_up"]})),
            LLMResponse(json.dumps({"task_id": "task_1", "evidence": ["read"]})),
            LLMResponse("готово"),
        ])
        result = engine.approve_plan(run_id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.store.tasks.get_all(run_id)[0]["evidence"], ["read"])
        self.assertFalse(json.loads(fake.calls[3]["messages"][-3]["content"])["success"])

    def test_failed_tests_allow_repair_in_current_task_without_repeating_dependencies(self):
        run_id = self.start()
        responses = self.responses()
        fake, engine = self.engine([
            *responses[:2],
            self.call("write", "write_file", path="order.py", content=self.broken),
            responses[3], responses[4],
            self.call("repair", "write_file", path="order.py", content=self.fixed),
            self.call("retest", "Bash", command="python3 -m unittest -v", cwd="."),
            self.done("task_3", ["retest"]), LLMResponse("готово"),
        ])
        result = engine.approve_plan(run_id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.statuses(run_id), ["completed"] * 3)
        self.assertEqual(self.store.get_tool_call(run_id, "repair")["task_id"], "task_3")
        self.assertEqual(self.store.tasks.get_all(run_id)[2]["evidence"], ["retest"])
        self.assertEqual(len(self.store.get_attempts(run_id, "write")), 1)
        self.assertTrue(engine._tests_verified(result.messages))
        self.assertEqual((self.workspace / "test_order.py").read_text(), self.tests)


if __name__ == "__main__":
    unittest.main()
