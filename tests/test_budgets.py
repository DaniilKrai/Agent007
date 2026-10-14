import io
import json
import tempfile
import unittest

from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from agent_my.cli import main, build_parser
from agent_my.engine import AgentEngine
from agent_my.llm import FakeLLM, LLMError
from agent_my.models import LLMResponse, ToolCall
from agent_my.runtime.budgets import BudgetLimits, MIN_RESULT_BYTES, MAX_RESULT_BYTES
from agent_my.runtime.executor import Executor
from agent_my.runtime.policy import Policy
from agent_my.runtime.store import RunStore
from agent_my.tools import build_tool_registry


class SimulatedCrash(BaseException):
    pass


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "order.py").write_text("value = 1\n")
        self.policy_path = self.root / "policy.json"
        self.policy_path.write_text(json.dumps({
            "commands": {"allow": ["pwd", "python3 -m unittest -v"], "deny": []},
            "directories": {"allow": ["."], "deny": []},
            "files": {"deny": ["test_order.py"]},
        }))
        self.policy = Policy(self.policy_path, self.workspace)
        self.registry = build_tool_registry(self.workspace, self.policy)
        self.executor = Executor(self.registry, self.policy)
        self.db_path = self.root / "runs.sqlite3"
        self.store = RunStore(self.db_path)
        self.addCleanup(lambda: self.store.close())

    def engine(self, responses, budgets=None, max_steps=20, mode="auto", crash=None):
        fake = FakeLLM(responses)
        return fake, AgentEngine(
            fake, self.registry, self.executor, store=self.store,
            budgets=budgets, max_steps=max_steps, mode=mode, after_tool_execution=crash,
        )

    def restart(self):
        self.store.close()
        self.store = RunStore(self.db_path)

    def read(self, call_id="read"):
        return ToolCall(call_id, "read_file", {"path": "order.py"})

    def write(self, call_id="write"):
        return ToolCall(call_id, "write_file", {"path": "order.py", "content": "value = 2\n"})

    def assert_result_accounting(self, run_id):
        messages = self.store.get_messages(run_id)
        size = sum(len(m["content"].encode("utf-8")) for m in messages if m["role"] == "tool")
        budget = self.store.get_budget(run_id)
        self.assertEqual(budget["result_bytes_used"], size)
        self.assertEqual(budget["result_bytes_reserved"], 0)
        self.assertLessEqual(size, budget["result_bytes_limit"])

    def test_failed_model_request_is_spent_before_resume(self):
        fake, engine = self.engine([], max_steps=1)
        with self.assertRaises(LLMError):
            engine.run("прочитай файл", run_id="model_failure")
        self.assertEqual(self.store.get_run("model_failure")["model_calls"], 1)
        self.restart()
        fake, engine = self.engine([LLMResponse("готово")], max_steps=100)
        result = engine.resume("model_failure")
        self.assertEqual(result.status, "limit_reached")
        self.assertEqual(fake.calls, [])
        self.assertEqual(result.model_calls, 1)

    def test_crash_after_model_debit_does_not_refund_request(self):
        self.store.create_run("model_crash", "задача", "агент", 2)
        self.assertTrue(self.store.begin_model_request("model_crash"))
        self.restart()
        fake, engine = self.engine([LLMResponse("готово")])
        result = engine.resume("model_crash")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(len(fake.calls), 1)

    def test_entire_batch_is_rejected_for_each_budget(self):
        for limits, reason in (
            (BudgetLimits(tool_calls=1), "вызовов инструментов"),
            (BudgetLimits(side_effects=0), "побочным эффектом"),
            (BudgetLimits(result_bytes=MIN_RESULT_BYTES * 2 - 1), "результатов"),
        ):
            with self.subTest(limits=limits):
                fake, engine = self.engine([
                    LLMResponse(None, (self.read(), self.write())),
                ], budgets=limits)
                with patch.object(self.registry, "execute", wraps=self.registry.execute) as execute:
                    result = engine.run("измени файл")
                execute.assert_not_called()
                self.assertEqual(result.status, "limit_reached")
                self.assertIn(reason, result.final_answer)
                self.assertEqual(result.tool_calls, 0)
                self.assertEqual((self.workspace / "order.py").read_text(), "value = 1\n")
                budget = self.store.get_budget(result.run_id)
                self.assertEqual(budget["tool_calls_used"], 0)
                self.assertEqual(budget["side_effects_used"], 0)
                self.assertEqual(budget["result_bytes_reserved"], 0)
                self.assertIsNone(self.store.next_pending(result.run_id))

    def test_side_effect_limit_counts_bash_and_write_together(self):
        fake, engine = self.engine([
            LLMResponse(None, (self.write(), ToolCall("bash", "Bash", {"command": "pwd", "cwd": "."}))),
        ], budgets=BudgetLimits(side_effects=1))
        result = engine.run("измени и проверь")
        self.assertEqual(result.status, "limit_reached")
        self.assertEqual((self.workspace / "order.py").read_text(), "value = 1\n")

    def test_exact_limits_allow_whole_batch(self):
        fake, engine = self.engine([
            LLMResponse(None, (self.read(), self.write())), LLMResponse("готово"),
        ], budgets=BudgetLimits(tool_calls=2, side_effects=1))
        result = engine.run("измени файл")
        self.assertEqual(result.status, "completed")
        budget = self.store.get_budget(result.run_id)
        self.assertEqual(budget["tool_calls_used"], 2)
        self.assertEqual(budget["side_effects_used"], 1)
        self.assert_result_accounting(result.run_id)

    def test_large_unicode_result_is_truncated_and_tool_receives_allowance(self):
        (self.workspace / "order.py").write_text('я\\"' * 3000, encoding="utf-8")
        fake, engine = self.engine([
            LLMResponse(None, (self.read(),)), LLMResponse("прочитано частично"),
        ], budgets=BudgetLimits(result_bytes=256))
        with patch.object(self.registry, "execute", wraps=self.registry.execute) as execute:
            result = engine.run("прочитай файл")
        self.assertEqual(execute.call_args.kwargs["result_limit"], 256)
        message = fake.calls[1]["messages"][-1]
        self.assertTrue(json.loads(message["content"])["truncated"])
        self.assertLessEqual(len(message["content"].encode("utf-8")), 256)
        self.assert_result_accounting(result.run_id)

    def test_batch_reserves_output_for_every_call(self):
        (self.workspace / "order.py").write_text("x" * 10000)
        fake, engine = self.engine([
            LLMResponse(None, (self.read("a"), self.read("b"))),
            LLMResponse("готово"),
        ], budgets=BudgetLimits(result_bytes=512))
        result = engine.run("прочитай дважды")
        self.assertEqual(result.tool_calls, 2)
        results = [m for m in result.messages if m["role"] == "tool"]
        self.assertTrue(all(json.loads(m["content"])["truncated"] for m in results))
        self.assert_result_accounting(result.run_id)
        self.assertEqual(self.store.get_budget(result.run_id)["result_bytes_used"], 512)

    def test_output_exhaustion_stops_next_batch(self):
        (self.workspace / "order.py").write_text("x" * 10000)
        fake, engine = self.engine([
            LLMResponse(None, (self.read(),)), LLMResponse(None, (self.write(),)),
        ], budgets=BudgetLimits(result_bytes=256))
        result = engine.run("прочитай и измени")
        self.assertEqual(result.status, "limit_reached")
        self.assertIn("результатов", result.final_answer)
        self.assertEqual((self.workspace / "order.py").read_text(), "x" * 10000)
        self.assertEqual(result.tool_calls, 1)

    def test_denial_and_resume_do_not_reset_reserved_call_budget(self):
        fake, engine = self.engine([LLMResponse(None, (self.write(),))], mode="safe",
                                   budgets=BudgetLimits(tool_calls=1, side_effects=1))
        result = engine.run("измени файл")
        before = self.store.get_budget(result.run_id)
        self.assertEqual(result.status, "waiting_approval")
        self.restart()
        fake, engine = self.engine([LLMResponse(None, (self.read(),))])
        self.assertEqual(engine.resume(result.run_id).status, "waiting_approval")
        self.assertEqual(self.store.get_budget(result.run_id), before)
        result = engine.reject(result.run_id)
        self.assertEqual(result.status, "limit_reached")
        self.assertEqual(self.store.get_budget(result.run_id)["tool_calls_used"], 1)
        self.assertEqual((self.workspace / "order.py").read_text(), "value = 1\n")
        self.assert_result_accounting(result.run_id)

    def test_cached_call_is_not_executed_but_delivery_has_a_budget(self):
        fake, engine = self.engine([
            LLMResponse(None, (self.write(),)),
            LLMResponse(None, (self.write(),)), LLMResponse("готово"),
        ])
        result = engine.run("измени файл")
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(self.store.get_budget(result.run_id)["tool_calls_used"], 2)
        self.assert_result_accounting(result.run_id)

    def test_plan_approval_preserves_tool_budget(self):
        plan = {
            "goal": "проверить", "steps": [{"description": "прочитать", "tools": ["read_file"]}],
            "risks": [], "assumptions": [], "done_when": "файл прочитан",
        }
        fake, engine = self.engine([
            LLMResponse(None, (ToolCall("plan", "submit_plan", plan),)),
        ], budgets=BudgetLimits(tool_calls=1))
        result = engine.run("проверь", plan=True)
        self.assertEqual(result.status, "waiting_plan")
        before = self.store.get_budget(result.run_id)
        self.restart()
        fake, engine = self.engine([LLMResponse(None, (self.read(),))])
        result = engine.approve_plan(result.run_id)
        self.assertEqual(result.status, "limit_reached")
        self.assertEqual(self.store.get_budget(result.run_id), before)
        self.assertEqual(result.model_calls, 2)

    @staticmethod
    def crash(call, result):
        raise SimulatedCrash()

    def test_crash_consumes_output_reservation_and_retry_costs_more(self):
        fake, engine = self.engine([LLMResponse(None, (self.read(),))], crash=self.crash)
        with self.assertRaises(SimulatedCrash):
            engine.run("прочитай файл", run_id="crashed")
        budget = self.store.get_budget("crashed")
        self.assertEqual(budget["tool_calls_used"], 1)
        self.assertEqual(budget["result_bytes_reserved"], MAX_RESULT_BYTES)
        self.restart()
        fake, engine = self.engine([LLMResponse("готово")])
        self.assertEqual(engine.resume("crashed").status, "waiting_resolution")
        budget = self.store.get_budget("crashed")
        self.assertEqual(budget["result_bytes_used"], MAX_RESULT_BYTES)
        self.assertEqual(budget["result_bytes_reserved"], 0)
        result = engine.resolve("crashed", "retry")
        self.assertEqual(result.status, "completed")
        budget = self.store.get_budget("crashed")
        self.assertEqual(budget["tool_calls_used"], 2)
        self.assertGreater(budget["result_bytes_used"], MAX_RESULT_BYTES)
        self.assertEqual(len(self.store.get_attempts("crashed", "read")), 2)

    def test_retry_cannot_bypass_exhausted_budget(self):
        for limits in (BudgetLimits(tool_calls=1), BudgetLimits(result_bytes=256)):
            with self.subTest(limits=limits):
                fake, engine = self.engine([LLMResponse(None, (self.read(),))], budgets=limits, crash=self.crash)
                run_id = "retry_" + str(limits.result_bytes)
                with self.assertRaises(SimulatedCrash):
                    engine.run("прочитай файл", run_id=run_id)
                self.restart()
                fake, engine = self.engine([])
                self.assertEqual(engine.resume(run_id).status, "waiting_resolution")
                with patch.object(self.registry, "execute") as execute:
                    result = engine.resolve(run_id, "retry")
                self.assertEqual(result.status, "limit_reached")
                execute.assert_not_called()
                self.assertEqual(self.store.get_budget(run_id)["tool_calls_used"], 1)

    def test_crash_before_handler_keeps_debit_and_started_state(self):
        fake, engine = self.engine([LLMResponse(None, (self.write(),))])
        with patch.object(self.registry, "execute", side_effect=SimulatedCrash):
            with self.assertRaises(SimulatedCrash):
                engine.run("измени файл", run_id="before_handler")
        self.assertEqual(self.store.get_budget("before_handler")["side_effects_used"], 1)
        self.assertEqual(self.store.get_tool_call("before_handler", "write")["status"], "started")
        self.assertEqual((self.workspace / "order.py").read_text(), "value = 1\n")
        self.restart()
        fake, engine = self.engine([LLMResponse("готово")])
        self.assertEqual(engine.resume("before_handler").status, "waiting_resolution")
        result = engine.resolve("before_handler", "skip")
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.store.get_budget(result.run_id)["side_effects_used"], 1)

    def test_failed_transaction_does_not_partially_admit_batch(self):
        self.store.create_run("atomic", "задача", "агент", 20)
        self.store.begin_model_request("atomic")
        original = self.store._admit_batch

        def crash_after_debit(*args):
            original(*args)
            raise SimulatedCrash()

        with patch.object(self.store, "_admit_batch", side_effect=crash_after_debit):
            with self.assertRaises(SimulatedCrash):
                self.store.record_response("atomic", LLMResponse(None, (self.read(), self.write())))
        self.restart()
        self.assertEqual(self.store.get_budget("atomic")["tool_calls_used"], 0)
        self.assertEqual(self.store.get_budget("atomic")["result_bytes_reserved"], 0)
        self.assertIsNone(self.store.next_pending("atomic"))
        self.assertEqual(self.store.get_run("atomic")["model_calls"], 1)

    def test_bash_output_is_bounded_and_cannot_verify_tests_when_truncated(self):
        (self.workspace / "test_order.py").write_text(
            "import unittest\nclass TestOutput(unittest.TestCase):\n"
            "    def test_output(self):\n        print('я' * 10000)\n", encoding="utf-8",
        )
        fake, engine = self.engine([
            LLMResponse(None, (ToolCall("bash", "Bash", {"command": "python3 -m unittest -v", "cwd": "."}),)),
            LLMResponse("вывод неполный"),
        ], budgets=BudgetLimits(result_bytes=256))
        result = engine.run("проверь тесты")
        tool_message = next(m for m in result.messages if m["role"] == "tool")
        self.assertTrue(json.loads(tool_message["content"])["truncated"])
        self.assertFalse(engine._tests_verified(result.messages))
        self.assert_result_accounting(result.run_id)

    def test_cli_limits_and_saved_status(self):
        args = build_parser().parse_args([
            "run", "задача", "--max-model-calls", "4", "--max-tool-calls", "3",
            "--max-side-effects", "0", "--max-result-bytes", "512",
        ])
        self.assertEqual((args.max_steps, args.max_tool_calls, args.max_side_effects, args.max_result_bytes), (4, 3, 0, 512))
        fake, engine = self.engine([LLMResponse("готово")], budgets=BudgetLimits(3, 0, 512))
        result = engine.run("задача")
        output = io.StringIO()
        with patch("agent_my.cli.STATE_DB", self.db_path), redirect_stdout(output):
            self.assertEqual(main(["status", result.run_id]), 0)
        self.assertIn("бюджет результатов в байтах: 0/512", output.getvalue())
        self.assertIn("остаток: 512", output.getvalue())
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "задача", "--max-result-bytes", "-1"]), 1)

    def test_old_database_preserves_history_and_pending_approval(self):
        fake, engine = self.engine([
            LLMResponse(None, (self.read(),)), LLMResponse(None, (self.write(),)),
        ], mode="safe")
        result = engine.run("измени файл")
        self.assertEqual(result.status, "waiting_approval")
        with self.store.db:
            self.store.db.execute("DROP TABLE budgets")
            self.store.db.execute("ALTER TABLE pending_calls DROP COLUMN result_limit")
            self.store.db.execute("ALTER TABLE pending_calls DROP COLUMN result_charged")
            self.store.db.execute("ALTER TABLE runs DROP COLUMN model_started")
        self.restart()
        budget = self.store.get_budget(result.run_id)
        self.assertEqual(budget["tool_calls_used"], 2)
        self.assertEqual(budget["side_effects_used"], 1)
        self.assertGreater(budget["result_bytes_used"], 0)
        self.assertEqual(budget["result_bytes_reserved"], MAX_RESULT_BYTES)
        self.restart()
        self.assertEqual(self.store.get_budget(result.run_id), budget)
        fake, engine = self.engine([LLMResponse("готово")])
        result = engine.approve(result.run_id)
        self.assertEqual(result.status, "completed")
        self.assertEqual((self.workspace / "order.py").read_text(), "value = 2\n")
        self.assert_result_accounting(result.run_id)

    def test_result_transaction_failure_keeps_reservation_and_started_call(self):
        fake, engine = self.engine([LLMResponse(None, (self.write(),))])
        append = self.store._append_message

        def crash_on_result(run_id, message):
            if message["role"] == "tool":
                raise SimulatedCrash()
            append(run_id, message)

        with patch.object(self.store, "_append_message", side_effect=crash_on_result):
            with self.assertRaises(SimulatedCrash):
                engine.run("измени файл", run_id="result_crash")
        self.assertEqual((self.workspace / "order.py").read_text(), "value = 2\n")
        self.assertEqual(self.store.get_tool_call("result_crash", "write")["status"], "started")
        budget = self.store.get_budget("result_crash")
        self.assertEqual(budget["result_bytes_used"], 0)
        self.assertEqual(budget["result_bytes_reserved"], MAX_RESULT_BYTES)
        self.restart()
        fake, engine = self.engine([])
        self.assertEqual(engine.resume("result_crash").status, "waiting_resolution")
        budget = self.store.get_budget("result_crash")
        self.assertEqual(budget["result_bytes_used"], MAX_RESULT_BYTES)
        self.assertEqual(budget["result_bytes_reserved"], 0)


if __name__ == "__main__":
    unittest.main()
