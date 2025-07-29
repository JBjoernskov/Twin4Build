#!/usr/bin/env python3
"""
Development Environment Setup Script for Twin4Build

This script automates the setup of a development environment for Twin4Build contributors.
It handles conda environment creation, dependency installation, and pre-commit setup.
"""

# Standard library imports
import os
import platform
import subprocess
import sys
from pathlib import Path


def run_command(command, check=True, capture_output=False):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(
            command, shell=True, check=check, capture_output=capture_output, text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e}")
        if capture_output and e.stdout:
            print(f"stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"stderr: {e.stderr}")
        return None


def check_and_change_directory():
    """Ensure we're in the project root directory."""
    current_dir = Path.cwd()

    # Check if we're in the scripts directory
    if current_dir.name == "scripts":
        # Move up one level to project root
        project_root = current_dir.parent
        os.chdir(project_root)
        print(f"📁 Changed directory to project root: {project_root}")

    # Verify we're in the correct directory by checking for pyproject.toml
    if not Path("pyproject.toml").exists():
        print("❌ Error: Could not find pyproject.toml")
        print("   Please run this script from the Twin4Build project root directory")
        print("   or from the scripts/ subdirectory")
        return False

    print(f"✅ Running from project directory: {Path.cwd()}")
    return True


def check_python_version():
    """Check if Python version meets requirements."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_conda_available():
    """Check if conda is available."""
    result = run_command("conda --version", capture_output=True, check=False)
    if result is None or result.returncode != 0:
        print("❌ Conda not found. Please install conda or miniconda first.")
        print("   Download from: https://docs.conda.io/en/latest/miniconda.html")
        return False
    print(f"✅ {result.stdout.strip()}")
    return True


def create_conda_environment():
    """Create a conda environment."""
    env_name = "t4bdev"

    # Check if environment already exists
    print(f"Checking if conda environment '{env_name}' exists...")
    result = run_command("conda env list", capture_output=True, check=False)
    if result and result.returncode == 0:
        # Look for the environment name in the output
        if (
            f" {env_name} " in result.stdout
            or result.stdout.endswith(f" {env_name}")
            or result.stdout.startswith(f"{env_name} ")
        ):
            print(f"✅ Conda environment '{env_name}' already exists")
            return True

    print(f"Creating conda environment '{env_name}'...")
    result = run_command(f"conda create -n {env_name} python=3.9 -y")
    if result is None:
        return False

    print(f"✅ Conda environment '{env_name}' created")
    return True


def get_conda_commands():
    """Get conda-specific commands."""
    env_name = "t4bdev"
    if platform.system() == "Windows":
        activate_cmd = f"conda activate {env_name}"
        pip_cmd = f"conda run -n {env_name} pip"
        python_cmd = f"conda run -n {env_name} python"
    else:
        activate_cmd = f"conda activate {env_name}"
        pip_cmd = f"conda run -n {env_name} pip"
        python_cmd = f"conda run -n {env_name} python"

    return activate_cmd, pip_cmd, python_cmd


def install_dependencies():
    """Install project dependencies using conda environment."""
    print("Installing project dependencies in conda environment...")

    _, pip_cmd, _ = get_conda_commands()

    # Install the package in development mode (skip pip upgrade to avoid issues)
    print("Installing Twin4Build in development mode...")
    result = run_command(f"{pip_cmd} install -e .")
    if result is None:
        return False

    # Install development dependencies
    print("Installing development dependencies...")
    result = run_command(f"{pip_cmd} install -e .[dev]")
    if result is None:
        return False

    print("✅ Dependencies installed in conda environment")
    return True


def run_tests():
    """Run the test suite to verify setup using conda environment."""
    print("\n" + "=" * 50)
    print("🧪 Running test suite to verify installation...")
    print("=" * 50)
    print("ℹ️  This may take a few minutes depending on your system")
    print("ℹ️  Testing will verify that Twin4Build components work correctly")
    print("ℹ️  You'll see detailed test output below...")
    print("")

    _, _, python_cmd = get_conda_commands()

    print("⏳ Starting test discovery and execution...")
    result = run_command(f"{python_cmd} -m unittest discover twin4build/tests/ -v")

    if result is None:
        print("\n❌ Tests failed!")
        print("⚠️  This might indicate:")
        print("   • Missing dependencies")
        print("   • Environment configuration issues")
        print("   • Test-specific requirements not met")
        print("⚠️  Your development environment may still be functional")
        print("   Try running tests manually after activation:")
        print("   conda activate t4bdev")
        print("   python -m unittest discover twin4build/tests/ -v")
        return False

    print("\n✅ All tests passed!")
    print("🎉 Your Twin4Build development environment is working correctly")
    return True


def print_next_steps():
    """Print next steps for the developer."""
    activate_cmd, _, _ = get_conda_commands()

    print("\n" + "=" * 60)
    print("✅ Development environment setup complete!")
    print("\n🎯 Next steps:")
    print("   1. Activate the environment: conda activate t4bdev")
    print("   2. Run tests: python -m unittest discover twin4build/tests/")
    print("   3. Check code quality: python scripts/validate_code.py")
    print("   4. Read the developer guide: docs/source/manual/developer_reference.rst")
    print("\n🛠️  Available tools:")
    print("   • Black (code formatting)")
    print("   • isort (import sorting)")
    print("   • flake8 (style checking)")
    print("   • Coverage (test coverage)")
    print("   • Sphinx (documentation)")
    print(
        "\n💡 Use 'python scripts/validate_code.py --fix' to auto-fix formatting issues"
    )


def main():
    """Main setup function."""
    print("Twin4Build Development Environment Setup")
    print("=" * 40)

    # Check and change to correct directory
    if not check_and_change_directory():
        sys.exit(1)

    print(
        "\n🐍 This script will create a conda environment for Twin4Build development."
    )
    print("📦 Environment name: 't4bdev'")
    print("🔒 This will NOT modify your base conda environment.")

    # Ask for confirmation
    response = input("\nDo you want to continue? (y/N): ").strip().lower()
    if response not in ("y", "yes"):
        print("Setup cancelled.")
        print("\n💡 To set up manually:")
        print("   conda create -n t4bdev python=3.9")
        print("   conda activate t4bdev")
        print("   pip install -e .")
        print("   pip install -e .[dev]")
        return

    # Check conda availability
    if not check_conda_available():
        sys.exit(1)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create conda environment
    if not create_conda_environment():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Run tests
    run_tests()

    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
