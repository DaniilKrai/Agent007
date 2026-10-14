import io
import unittest

from contextlib import redirect_stdout
from agent_my.cli import main

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