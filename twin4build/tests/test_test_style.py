"""Every test is a ``unittest.TestCase``; pytest is only the runner.

Tests written as plain pytest functions or classes never ran while CI ran
``python -m unittest discover``, and a test that imports pytest does not run
under unittest.  No module under ``twin4build/tests`` may import pytest,
define a test function at module level, or define a ``Test*`` class without a
base class (pytest collects it, unittest does not).
"""

# Standard library imports
import ast
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parents[1]


def _imports_pytest(node):
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.level == 0:
        names = [node.module or ""]
    else:
        return False
    return any(name == "pytest" or name.startswith("pytest.") for name in names)


def _violations(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    found = [
        (node.lineno, "imports pytest") for node in ast.walk(tree) if _imports_pytest(node)
    ]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                found.append((node.lineno, f"module-level test function {node.name}"))
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            bases = [ast.unparse(base) for base in node.bases]
            if bases in ([], ["object"]):
                found.append((node.lineno, f"class {node.name} is not a unittest.TestCase"))
    return sorted(found)


class TestTestStyle(unittest.TestCase):
    def test_tests_are_unittest_testcases_without_pytest(self):
        violations = [
            f"{path.relative_to(ROOT).as_posix()}:{line}: {problem}"
            for path in sorted(TESTS.rglob("*.py"))
            if "__pycache__" not in path.parts
            for line, problem in _violations(path)
        ]
        if violations:
            self.fail(
                "Write tests as unittest.TestCase methods and use unittest's tools "
                "(unittest.mock, subTest, skipIf, assertRaises) instead of pytest's:\n"
                + "\n".join(violations)
            )


if __name__ == "__main__":
    unittest.main()
