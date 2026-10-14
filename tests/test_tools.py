
import json
import tempfile
import unittest

from pathlib import Path

from agent_my.models import ToolCall
from agent_my.tools import build_tool_registry


class ToolRegistryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.workspace = Path(self.temp.name) / "creating_space"
        self.workspace.mkdir()

        (self.workspace / "order.py").write_text(
            "print('hello')\n",
            encoding="utf-8",
        )

        (self.workspace / "subdir").mkdir()

        self.registry = build_tool_registry(self.workspace)

    def call(self, name, **arguments):
        return self.registry.execute(
            ToolCall(
                id="call_test",
                name=name,
                arguments=arguments,
            )
        )

    def test_registered_tools(self):
        definitions = self.registry.openai_tools()

        names = {
            tool["function"]["name"]
            for tool in definitions
        }

        self.assertEqual(
            names,
            {
                "list_files",
                "read_file",
                "write_file",
                "delete_file",
            },
        )

    def test_list_files(self):
        result = self.call("list_files", path=".")

        self.assertTrue(result.success)

        names = {
            item["name"]
            for item in result.data["items"]
        }

        self.assertEqual(names, {"order.py", "subdir"})

    def test_read_file(self):
        result = self.call("read_file", path="order.py")

        self.assertTrue(result.success)
        self.assertEqual(
            result.data["content"],
            "print('hello')\n",
        )

    def test_write_existing_file(self):
        result = self.call(
            "write_file",
            path="order.py",
            content="print('updated')\n",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data["operation"], "updated")

        content = (self.workspace / "order.py").read_text(
            encoding="utf-8",
        )

        self.assertEqual(content, "print('updated')\n")

    def test_create_file(self):
        result = self.call(
            "write_file",
            path="new.py",
            content="x = 42\n",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data["operation"], "created")
        self.assertTrue((self.workspace / "new.py").exists())

    def test_delete_file(self):
        result = self.call("delete_file", path="order.py")

        self.assertTrue(result.success)
        self.assertFalse((self.workspace / "order.py").exists())

    def test_unknown_tool(self):
        result = self.call("unknown_tool", path="order.py")

        self.assertFalse(result.success)
        self.assertIn("неизвестный инструмент", result.error)

    def test_missing_argument(self):
        result = self.call("read_file")

        self.assertFalse(result.success)
        self.assertIn(
            "отсутствует обязательный аргумент",
            result.error,
        )

    def test_wrong_argument_type(self):
        result = self.call("read_file", path=123)

        self.assertFalse(result.success)
        self.assertIn(
            "неверный тип аргумента",
            result.error,
        )

    def test_extra_argument(self):
        result = self.call(
            "read_file",
            path="order.py",
            unexpected=True,
        )

        self.assertFalse(result.success)
        self.assertIn("неизвестные аргументы", result.error)

    def test_missing_file(self):
        result = self.call("read_file", path="missing.py")

        self.assertFalse(result.success)
        self.assertIn("файл не найден", result.error)

    def test_parent_traversal(self):
        result = self.call(
            "read_file",
            path="../outside.txt",
        )

        self.assertFalse(result.success)
        self.assertIn(
            "переход в родительскую директорию запрещён",
            result.error,
        )

    def test_absolute_path(self):
        result = self.call(
            "write_file",
            path=str(Path(self.temp.name) / "outside.txt"),
            content="danger",
        )

        self.assertFalse(result.success)
        self.assertIn("абсолютные пути запрещены", result.error)
        self.assertFalse(
            (Path(self.temp.name) / "outside.txt").exists()
        )

    def test_symlink_to_outside(self):
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")

        link = self.workspace / "link.txt"
        link.symlink_to(outside)

        for name in ("read_file", "delete_file"):
            with self.subTest(tool=name):
                result = self.call(name, path="link.txt")

                self.assertFalse(result.success)
                self.assertIn(
                    "использование символьных ссылок запрещено",
                    result.error,
                )

        result = self.call(
            "write_file",
            path="link.txt",
            content="modified",
        )

        self.assertFalse(result.success)
        self.assertIn(
            "использование символьных ссылок запрещено",
            result.error,
        )

        self.assertEqual(
            outside.read_text(encoding="utf-8"),
            "secret",
        )

    def test_symlink_directory(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()

        (outside / "secret.txt").write_text(
            "secret",
            encoding="utf-8",
        )

        (self.workspace / "linkdir").symlink_to(
            outside,
            target_is_directory=True,
        )

        result = self.call(
            "read_file",
            path="linkdir/secret.txt",
        )

        self.assertFalse(result.success)
        self.assertIn(
            "использование символьных ссылок запрещено",
            result.error,
        )

    def test_tool_message(self):
        result = self.call("read_file", path="order.py")
        message = result.to_message()

        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "call_test")

        content = json.loads(message["content"])

        self.assertTrue(content["success"])
        self.assertEqual(
            content["data"]["content"],
            "print('hello')\n",
        )


if __name__ == "__main__":
    unittest.main()
