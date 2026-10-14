import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from agent_my.models import ToolCall
from agent_my.runtime.executor import Executor
from agent_my.runtime.policy import Policy
from agent_my.tools import build_tool_registry
from agent_my.tools.bash import BashTool


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.root = Path(self.temp.name)

        self.workspace = self.root / "creating_space"
        self.workspace.mkdir()

        self.private = self.workspace / "private"
        self.private.mkdir()

        (self.workspace / "order.py").write_text(
            "value = 1\n",
            encoding="utf-8",
        )

        (self.workspace / "test_order.py").write_text(
            "value = 2\n",
            encoding="utf-8",
        )

        (self.private / "secret.txt").write_text(
            "secret",
            encoding="utf-8",
        )

        self.policy_path = self.root / "policy.json"

        self.config = {
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
        }

        self.save_policy()

    def save_policy(self):
        self.policy_path.write_text(
            json.dumps(self.config),
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

    def call(self, name, **arguments):
        return self.executor.execute(
            ToolCall(
                id="call_test",
                name=name,
                arguments=arguments,
            ),
            mode="auto",
        )

    def test_executor_requires_approval_without_agent_loop(self):
        for mode, name, arguments in (
            ("safe", "write_file", {"path": "order.py", "content": "changed"}),
            ("safe", "Bash", {"command": "pwd", "cwd": "."}),
            ("safe", "delete_file", {"path": "order.py"}),
            ("auto", "delete_file", {"path": "order.py"}),
        ):
            with self.subTest(mode=mode, name=name):
                with patch.object(self.registry, "execute") as execute:
                    result = self.executor.execute(ToolCall("call", name, arguments), mode=mode)

                self.assertFalse(result.success)
                self.assertIn("требуется подтверждение", result.error)
                execute.assert_not_called()

        self.assertEqual((self.workspace / "order.py").read_text(), "value = 1\n")

    def test_executor_defaults_to_safe(self):
        result = self.executor.execute(ToolCall("call", "write_file", {
            "path": "order.py", "content": "changed",
        }))

        self.assertFalse(result.success)
        self.assertIn("подтверждение", result.error)
        self.assertEqual((self.workspace / "order.py").read_text(), "value = 1\n")

    def test_executor_rejects_unknown_mode(self):
        with patch.object(self.registry, "execute") as execute:
            result = self.executor.execute(
                ToolCall("call", "read_file", {"path": "order.py"}),
                mode="unknown", approved=True,
            )

        self.assertFalse(result.success)
        self.assertIn("неизвестный режим", result.error)
        execute.assert_not_called()

    def test_executor_checks_policy_again_before_execution(self):
        call = ToolCall("write", "write_file", {"path": "order.py", "content": "changed"})
        self.assertIsNone(self.executor.check(call, "safe", approved=True).error)
        self.config["files"]["deny"].append("order.py")
        self.policy_path.write_text(json.dumps(self.config), encoding="utf-8")

        with patch.object(self.registry, "execute") as execute:
            result = self.executor.execute(call, mode="safe", approved=True)

        self.assertFalse(result.success)
        self.assertIn("запрещено", result.error)
        execute.assert_not_called()
        self.assertEqual((self.workspace / "order.py").read_text(), "value = 1\n")

    def test_invalid_policy_blocks_execution(self):
        self.policy_path.write_text("{invalid", encoding="utf-8")

        with patch.object(self.registry, "execute") as execute:
            result = self.executor.execute(
                ToolCall("write", "write_file", {"path": "order.py", "content": "changed"}),
                mode="auto",
            )

        self.assertFalse(result.success)
        self.assertIn("некорректный json", result.error)
        execute.assert_not_called()

    def test_non_object_arguments_return_tool_error(self):
        for arguments in ('null', '[]', '"text"', 'true', '42', '{}'):
            with self.subTest(arguments=arguments):
                with patch.object(self.registry, "execute") as execute:
                    result = self.executor.execute(
                        ToolCall("bad", "write_file", arguments), mode="auto",
                    )

                self.assertFalse(result.success)
                self.assertEqual(result.tool_call_id, "bad")
                self.assertIn("должны быть объектом", result.error)
                execute.assert_not_called()

    def test_bash_registered(self):
        names = {
            tool["function"]["name"]
            for tool in self.registry.openai_tools()
        }

        self.assertIn("Bash", names)
        self.assertIn("read_file", names)

    def test_allowed_command(self):
        result = self.call(
            "Bash",
            command="pwd",
            cwd=".",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data["exit_code"], 0)

        actual_path = Path(result.data["stdout"].strip()).resolve()
        expected_path = self.workspace.resolve()

        self.assertEqual(actual_path, expected_path)

    def test_invalid_bash_arguments_explain_error_without_execution(self):
        for arguments, expected in (
            ({"command": "pwd"}, "отсутствует обязательный аргумент: cwd"),
            ({"command": "pwd", "cwd": 1}, "неверный тип аргумента cwd: ожидается строка"),
            ({"command": "pwd", "cwd": ".", "timeout": 5}, "неизвестные аргументы: timeout"),
        ):
            with self.subTest(arguments=arguments):
                with patch.object(self.registry, "execute") as execute:
                    result = self.executor.execute(ToolCall("bad", "Bash", arguments))
                self.assertFalse(result.success)
                self.assertIn(expected, result.error)
                execute.assert_not_called()

    def test_denied_command(self):
        result = self.call(
            "Bash",
            command="rm -rf .",
            cwd=".",
        )

        self.assertFalse(result.success)
        self.assertIn(
            "команда запрещена",
            result.error,
        )

    def test_unknown_command(self):
        result = self.call(
            "Bash",
            command="echo hello",
            cwd=".",
        )

        self.assertFalse(result.success)
        self.assertIn(
            "не входит в список разрешённых",
            result.error,
        )

    def test_deny_has_priority(self):
        self.config["commands"]["deny"].append("pwd")
        self.save_policy()

        result = self.call(
            "Bash",
            command="pwd",
            cwd=".",
        )

        self.assertFalse(result.success)
        self.assertIn(
            "команда запрещена",
            result.error,
        )

    def test_forbidden_bash_syntax(self):
        self.config["commands"]["allow"].extend([
            "pwd; echo hello",
            "pwd | cat",
            "python3 -c print(123)",
        ])

        self.save_policy()

        commands = [
            "pwd; echo hello",
            "pwd | cat",
            "python3 -c print(123)",
        ]

        for command in commands:
            with self.subTest(command=command):
                result = self.call(
                    "Bash",
                    command=command,
                    cwd=".",
                )

                self.assertFalse(result.success)

    def test_denied_working_directory(self):
        result = self.call(
            "Bash",
            command="pwd",
            cwd="private",
        )

        self.assertFalse(result.success)
        self.assertIn(
            "доступ к пути запрещён",
            result.error,
        )

    def test_denied_subdirectory(self):
        result = self.call(
            "read_file",
            path="private/secret.txt",
        )

        self.assertFalse(result.success)
        self.assertIn(
            "доступ к пути запрещён",
            result.error,
        )

    def test_parent_traversal(self):
        result = self.call(
            "read_file",
            path="../policy.json",
        )

        self.assertFalse(result.success)
        self.assertIn(
            "переход в родительскую директорию запрещён",
            result.error,
        )

    def test_symlink(self):
        outside = self.root / "outside.txt"
        outside.write_text(
            "outside",
            encoding="utf-8",
        )

        link = self.workspace / "link.txt"
        link.symlink_to(outside)

        result = self.call(
            "read_file",
            path="link.txt",
        )

        self.assertFalse(result.success)
        self.assertIn(
            "использование символьных ссылок запрещено",
            result.error,
        )

    def test_protected_file(self):
        original = (
            self.workspace / "test_order.py"
        ).read_text(encoding="utf-8")

        result = self.call(
            "write_file",
            path="test_order.py",
            content="changed",
        )

        self.assertFalse(result.success)

        self.assertIn(
            "изменение файла запрещено",
            result.error,
        )

        current = (
            self.workspace / "test_order.py"
        ).read_text(encoding="utf-8")

        self.assertEqual(original, current)

    def test_protected_file_can_be_read(self):
        result = self.call(
            "read_file",
            path="test_order.py",
        )

        self.assertTrue(result.success)
        self.assertEqual(
            result.data["content"],
            "value = 2\n",
        )

    def test_protected_file_cannot_be_deleted(self):
        result = self.call(
            "delete_file",
            path="test_order.py",
        )

        self.assertFalse(result.success)
        self.assertTrue(
            (self.workspace / "test_order.py").exists()
        )

    def test_invalid_arguments(self):
        result = self.call(
            "Bash",
            command=123,
            cwd=".",
        )

        self.assertFalse(result.success)
        self.assertIn(
            "некорректные аргументы",
            result.error,
        )

    def test_policy_inside_workspace(self):
        inside = self.workspace / "policy.json"

        inside.write_text(
            json.dumps(self.config),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "вне рабочей директории",
        ):
            Policy(inside, self.workspace)

    def test_output_limit(self):
        bash = BashTool(
            self.policy,
            max_output_bytes=3,
        )

        result = bash.run(
            command="pwd",
            cwd=".",
        )

        self.assertTrue(result["truncated"])

        output_size = (
            len(result["stdout"].encode("utf-8"))
            + len(result["stderr"].encode("utf-8"))
        )

        self.assertLessEqual(output_size, 3)

    def test_unittest_command(self):
        test_file = self.workspace / "test_sample.py"

        test_file.write_text(
            "import unittest\n"
            "\n"
            "class SampleTest(unittest.TestCase):\n"
            "    def test_example(self):\n"
            "        self.assertEqual(2 + 2, 4)\n",
            encoding="utf-8",
        )

        result = self.call(
            "Bash",
            command="python3 -m unittest -v",
            cwd=".",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data["exit_code"], 0)
        self.assertIn(
            "test_example",
            result.data["stderr"],
        )

    def test_timeout(self):
        test_file = self.workspace / "test_slow.py"

        test_file.write_text(
            "import time\n"
            "import unittest\n"
            "\n"
            "class SlowTest(unittest.TestCase):\n"
            "    def test_wait(self):\n"
            "        time.sleep(1)\n",
            encoding="utf-8",
        )

        bash = BashTool(
            self.policy,
            timeout_seconds=0.2,
        )

        result = bash.run(
            command="python3 -m unittest -v",
            cwd=".",
        )

        self.assertTrue(result["timed_out"])


if __name__ == "__main__":
    unittest.main()
