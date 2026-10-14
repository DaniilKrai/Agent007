
import io
import unittest

from contextlib import redirect_stdout
from unittest.mock import patch

from agent_my.cli import main
from agent_my.models import LLMResponse, ToolCall

class CliTest(unittest.TestCase):
    def test_help(self):
        output = io.StringIO()

        with redirect_stdout(output):
            result = main([])

        self.assertEqual(result, 0)
        self.assertIn("coding-agent", output.getvalue())

    def test_version(self):
        output = io.StringIO()

        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as error:
                main(["--version"])

        self.assertEqual(error.exception.code, 0)
        self.assertIn("0.1.0", output.getvalue())

    def test_model_text(self):
        output = io.StringIO()

        with patch(
            "agent_my.cli.OpenAICompatibleClient"
        ) as client_class:
            client_class.return_value.complete.return_value = (
                LLMResponse(content="Привет.")
            )

            with redirect_stdout(output):
                result = main(["model-test"])

        self.assertEqual(result, 0)
        self.assertIn("ответ модели: привет", output.getvalue())

    def test_model_tool_call(self):
        output = io.StringIO()

        response = LLMResponse(
            content=None,
            tool_calls=(
                ToolCall(
                    id="call_1",
                    name="list_files",
                    arguments={"path": "."},
                ),
            ),
        )

        with patch(
            "agent_my.cli.OpenAICompatibleClient"
        ) as client_class:
            client_class.return_value.complete.return_value = response

            with patch(
                "agent_my.tools.registry.ToolRegistry.execute"
            ) as execute:
                with redirect_stdout(output):
                    result = main(["model-test", "--tools"])

                execute.assert_not_called()

        self.assertEqual(result, 0)
        self.assertIn(
            "модель запросила инструмент: list_files",
            output.getvalue(),
        )

if __name__ == "__main__":
    unittest.main()