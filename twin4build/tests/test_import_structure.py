"""Repository-wide import placement contract."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (ROOT / "twin4build", ROOT / "benchmarks", ROOT / "docs")
SKIP_PARTS = {
    "__pycache__",
    "build",
    "dist",
    "generated_files",
    "twin4build-2.0.0",
}


class _NestedImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope_depth = 0
        self.nested_imports: list[tuple[int, str]] = []

    def _visit_scope(self, node: ast.AST) -> None:
        self.scope_depth += 1
        self.generic_visit(node)
        self.scope_depth -= 1

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope
    visit_Lambda = _visit_scope

    def visit_Import(self, node: ast.Import) -> None:
        if self.scope_depth:
            self.nested_imports.append((node.lineno, ast.unparse(node)))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.scope_depth:
            self.nested_imports.append((node.lineno, ast.unparse(node)))


def _python_files() -> list[Path]:
    return sorted(
        path
        for source_root in SOURCE_ROOTS
        if source_root.exists()
        for path in source_root.rglob("*.py")
        if not any(part in SKIP_PARTS for part in path.parts)
    )


def test_all_imports_are_at_module_scope() -> None:
    violations = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        visitor = _NestedImportVisitor()
        visitor.visit(tree)
        violations.extend(
            f"{path.relative_to(ROOT)}:{line}: {statement}"
            for line, statement in visitor.nested_imports
        )

    assert not violations, "Imports must be at module scope:\n" + "\n".join(violations)
