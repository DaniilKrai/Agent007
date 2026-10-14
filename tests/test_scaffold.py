import unittest

from agent_my.config import PROJECT_ROOT, WORKSPACE_ROOT


class ScaffoldTest(unittest.TestCase):
    def test_workspace_exists(self):
        self.assertTrue(WORKSPACE_ROOT.is_dir())

    def test_control_files_exist(self):
        self.assertTrue((WORKSPACE_ROOT / "order.py").is_file())
        self.assertTrue((WORKSPACE_ROOT / "test_order.py").is_file())

    def test_workspace_inside_project(self):
        self.assertEqual(WORKSPACE_ROOT.parent, PROJECT_ROOT)