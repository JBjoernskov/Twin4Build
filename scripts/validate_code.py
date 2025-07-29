#!/usr/bin/env python3
"""
Code Validation Script for Twin4Build

This script runs all code quality checks that ensure your code meets
Twin4Build's standards before committing or submitting pull requests.

Usage:
    python scripts/validate_code.py [--fix]

Options:
    --fix    Automatically fix issues where possible (formatting, imports)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command and handle errors with clear output."""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    print()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            text=True
        )
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
    return True


def run_black(fix_mode=False):
    """Run Black code formatter."""
    if fix_mode:
        cmd = "black ."
        desc = "Formatting code with Black"
    else:
        cmd = "black --check --diff ."
        desc = "Checking code formatting with Black"
    
    return run_command(cmd, desc, check=False)


def run_flake8():
    """Run flake8 linting."""
    cmd = "flake8 twin4build/ scripts/ --max-line-length=88 --extend-ignore=E203,W503"
    desc = "Checking code style with flake8"
    return run_command(cmd, desc, check=False)


def run_isort(fix_mode=False):
    """Run isort import sorting."""
    if fix_mode:
        cmd = "isort . --profile=black --line-length=88"
        desc = "Sorting imports with isort"
    else:
        cmd = "isort . --profile=black --line-length=88 --check-only --diff"
        desc = "Checking import sorting with isort"
    
    return run_command(cmd, desc, check=False)


def run_mypy():
    """Run mypy type checking."""
    cmd = "mypy twin4build/ --ignore-missing-imports"
    desc = "Type checking with mypy"
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
        text=True
    )
    if result.stdout.strip():
        print("❌ Found trailing whitespace:")
        print(result.stdout)
        issues_found = True
    else:
        print("✅ No trailing whitespace found")
    
    # Check for files missing final newline (simplified check)
    print("\nChecking for files without final newline...")
    py_files = list(Path("twin4build").rglob("*.py")) + list(Path("scripts").rglob("*.py"))
    missing_newline = []
    
    for file_path in py_files:
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                if content and not content.endswith(b'\n'):
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
        print("   • Type annotation issues (mypy)")
        print("   • Test failures")
        print("   • File format issues")
        return False


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Twin4Build code quality")
    parser.add_argument(
        "--fix", 
        action="store_true", 
        help="Automatically fix formatting and import issues"
    )
    args = parser.parse_args()
    
    print("Twin4Build Code Validation")
    print("=" * 40)
    
    if not check_environment():
        sys.exit(1)
    
    print(f"\nMode: {'Fix issues automatically' if args.fix else 'Check only'}")
    if args.fix:
        print("⚠️  This will modify your files to fix formatting and import issues")
        response = input("Continue? (y/N): ").strip().lower()
        if response not in ('y', 'yes'):
            print("Validation cancelled")
            return
    
    # Run all validation checks
    results = {
        "Code formatting (Black)": run_black(args.fix),
        "Import sorting (isort)": run_isort(args.fix),
        "Code style (flake8)": run_flake8(),
        "Type checking (mypy)": run_mypy(),
        "File issues": check_file_issues(),
        "Tests": run_tests(),
    }
    
    # Print summary and exit with appropriate code
    success = print_summary(results, args.fix)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 