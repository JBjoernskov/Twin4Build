#!/usr/bin/env python3
"""
Code Validation Script for Twin4Build

This script runs all code quality checks that ensure your code meets
Twin4Build's standards before committing or submitting pull requests.

Current checks:
- Code formatting (Black)
- Import sorting (isort)
- Code style (flake8)
- File format issues (trailing whitespace, newlines)
- Tests (optional with --test flag)

Usage:
    python scripts/validate_code.py [--fix] [--test]

Options:
    --fix     Automatically fix issues where possible (formatting, imports)
    --test    Run the test suite as part of validation
"""

# Standard library imports
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_tool_available(tool_name):
    """Check if a tool is available in the PATH."""
    return shutil.which(tool_name) is not None


def run_command(command, description, check=True):
    """Run a command and handle errors with clear output."""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    print()

    try:
        result = subprocess.run(command, shell=True, check=check, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}")
        return False


def check_environment():
    """Verify we're in the correct environment and directory."""
    # Check if we're in project root
    if not Path("pyproject.toml").exists():
        print("❌ Error: Please run this script from the Twin4Build project root")
        print("   Current directory:", Path.cwd())
        return False

    print("✅ Running from project root directory")

    # Check which tools are available
    missing_tools = []
    tools = {
        "black": "Code formatting",
        "isort": "Import sorting",
        "flake8": "Code style checking",
    }

    for tool, description in tools.items():
        if not check_tool_available(tool):
            missing_tools.append(f"{tool} ({description})")

    if missing_tools:
        print("\n⚠️  Missing development tools:")
        for tool in missing_tools:
            print(f"   • {tool}")
        print("\n💡 Install missing tools with:")
        print("   pip install -e .[dev]")
        print("   # or manually: pip install black isort flake8")
        return False

    print("✅ All required development tools are available")
    return True


def run_black(fix_mode=False):
    """Run Black code formatter."""
    if not check_tool_available("black"):
        print("❌ Black not available - skipping")
        return False

    if fix_mode:
        cmd = "black ."
        desc = "Formatting code with Black"
    else:
        cmd = "black --check --diff ."
        desc = "Checking code formatting with Black"

    return run_command(cmd, desc, check=False)


def run_flake8():
    """Run flake8 linting."""
    if not check_tool_available("flake8"):
        print("❌ flake8 not available - skipping")
        return False

    # Use pyproject.toml configuration - no command line overrides
    cmd = "flake8 ."
    desc = "Checking code syntax and logic with flake8"
    return run_command(cmd, desc, check=False)


def run_isort(fix_mode=False):
    """Run isort import sorting."""
    if not check_tool_available("isort"):
        print("❌ isort not available - skipping")
        return False

    # Note: Critical files are skipped via pyproject.toml skip_glob configuration

    if fix_mode:
        cmd = "isort ."
        desc = "Sorting imports with isort"
    else:
        cmd = "isort . --check-only --diff"
        desc = "Checking import sorting with isort"

    return run_command(cmd, desc, check=False)


def check_file_issues():
    """Check for common file issues."""
    print(f"\n{'='*60}")
    print("🔍 Checking for file issues")
    print(f"{'='*60}")

    issues_found = False

    # Check for trailing whitespace
    print("Checking for trailing whitespace...")
    result = subprocess.run(
        "grep -r '[ \t]$' twin4build/ scripts/ || true",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print("❌ Found trailing whitespace:")
        print(result.stdout)
        issues_found = True
    else:
        print("✅ No trailing whitespace found")

    # Check for files missing final newline (simplified check)
    print("\nChecking for files without final newline...")
    py_files = list(Path("twin4build").rglob("*.py")) + list(
        Path("scripts").rglob("*.py")
    )
    missing_newline = []

    for file_path in py_files:
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                if content and not content.endswith(b"\n"):
                    missing_newline.append(file_path)
        except Exception:
            continue

    if missing_newline:
        print("❌ Files missing final newline:")
        for f in missing_newline:
            print(f"  {f}")
        issues_found = True
    else:
        print("✅ All files end with newline")

    if not issues_found:
        print("✅ File issues check - PASSED")
        return True
    else:
        print("❌ File issues check - FAILED")
        return False


def run_tests():
    """Run the test suite."""
    cmd = "python -m unittest discover twin4build/tests/ -v"
    desc = "Running test suite"
    return run_command(cmd, desc, check=False)


def print_summary(results, fix_mode):
    """Print a summary of all validation results."""
    print(f"\n{'='*60}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*60}")

    passed = sum(results.values())
    total = len(results)

    for check, passed_check in results.items():
        status = "✅ PASSED" if passed_check else "❌ FAILED"
        print(f"{check:<30} {status}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All validation checks passed!")
        print("✅ Your code is ready for commit/pull request")
        return True
    else:
        print(f"\n⚠️  {total - passed} validation checks failed")
        if not fix_mode:
            print("\n💡 To automatically fix formatting and import issues:")
            print("   python scripts/validate_code.py --fix")
        print("\n💡 Manual fixes may be needed for:")
        print("   • Code style issues (flake8)")
        print("   • File format issues")
        print("   • Test failures (if tests enabled)")
        return False


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Twin4Build code quality")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix formatting and import issues",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the test suite as part of validation"
    )
    args = parser.parse_args()

    print("Twin4Build Code Validation")
    print("=" * 40)

    if not check_environment():
        sys.exit(1)

    print(f"\nMode: {'Fix issues automatically' if args.fix else 'Check only'}")
    if args.test:
        print("🧪 Test suite will be included in validation")
    if args.fix:
        print("⚠️  This will modify your files to fix formatting and import issues")
        response = input("Continue? (y/N): ").strip().lower()
        if response not in ("y", "yes"):
            print("Validation cancelled")
            return

    # Run all validation checks
    results = {
        "Code formatting (Black)": run_black(args.fix),
        "Import sorting (isort)": run_isort(args.fix),
        "Code style (flake8)": run_flake8(),
        "File issues": check_file_issues(),
    }

    # Add tests if requested
    if args.test:
        results["Tests"] = run_tests()

    # Print summary and exit with appropriate code
    success = print_summary(results, args.fix)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
