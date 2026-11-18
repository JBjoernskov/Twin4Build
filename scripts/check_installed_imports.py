"""Import every module of the installed twin4build, in an environment that
holds only the package and the dependencies it declares.

The test jobs install ``.[dev]``, whose tools bring packages the library
itself imports.  A dependency missing from ``pyproject.toml`` then passes
every test and fails for a user on the first import: 2.0.0rc1 shipped without
PyYAML, which brickschema's brickify imports through the semantic model.

Run it from outside the checkout, so that ``import twin4build`` is the
installed package::

    cd /tmp && python /path/to/scripts/check_installed_imports.py
"""

import importlib
import pkgutil
import sys
import traceback

#: Top-level modules that only an extra installs, and which extra.
EXTRAS = {
    "psycopg2": "database",
    "sqlalchemy": "database",
    "triton": "gpu",
    "nbformat": "notebooks",
    "jupyter": "notebooks",
}
#: Not library code: the test suite and the example scripts.
SKIPPED = ("twin4build.tests", "twin4build.examples")


def _modules(package):
    """Every module below ``package``, without entering the skipped ones."""

    for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if info.name.startswith(SKIPPED):
            continue
        yield info.name
        if info.ispkg:
            try:
                subpackage = importlib.import_module(info.name)
            except Exception:  # reported when the name itself is imported
                continue
            yield from _modules(subpackage)


def main():
    import twin4build

    print(f"twin4build from {twin4build.__file__}")
    failures, needs_extra, count = [], [], 0
    for name in _modules(twin4build):
        count += 1
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            missing = (error.name or "").split(".")[0]
            if missing in EXTRAS:
                needs_extra.append(f"{name}: {missing} ([{EXTRAS[missing]}])")
            else:
                failures.append((name, traceback.format_exc()))
        except Exception:
            failures.append((name, traceback.format_exc()))
    print(f"{count} modules imported")
    for line in needs_extra:
        print(f"needs an extra, not installed: {line}")
    for name, text in failures:
        print(f"\nFAILED {name}\n{text}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
