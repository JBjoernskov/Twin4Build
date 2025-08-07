#!/usr/bin/env python3
"""
Development Environment Setup Script for Twin4Build

This script automates the setup of a development environment for Twin4Build contributors.
It handles conda environment creation, dependency installation, and pre-commit setup.
"""

# Standard library imports
import argparse
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
    """Check if Python version meets requirements.

    Note: This function checks the current Python interpreter version,
    not the version that will be installed in the conda environment.
    It's kept for potential future use but is not called in main() since
    conda will install the specified Python version regardless of the
    current interpreter version.
    """
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


def create_conda_environment(python_version="3.12", env_name="t4bdev"):
    """Create a conda environment with specified Python version."""

    # Check if environment already exists
    print(f"Checking if conda environment '{env_name}' exists...")
    result = run_command("conda env list", capture_output=True, check=False)
    if result and result.returncode == 0:
        # Parse each line to extract environment names
        for line in result.stdout.strip().split("\n"):
            # Skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue

            # Split the line and get the first column (environment name)
            parts = line.split()
            if parts and parts[0] == env_name:
                print(f"❌ Conda environment '{env_name}' already exists!")
                print(
                    f"   Please choose a different environment name or remove the existing one:"
                )
                print(f"   conda env remove -n {env_name}")
                print(f"   Then run this script again.")
                return False

    print(f"Creating conda environment '{env_name}' with Python {python_version}...")
    result = run_command(f"conda create -n {env_name} python={python_version} -y")
    if result is None:
        return False

    print(f"✅ Conda environment '{env_name}' created with Python {python_version}")
    return True


def get_conda_commands(env_name="t4bdev"):
    """Get conda-specific commands."""
    if platform.system() == "Windows":
        activate_cmd = f"conda activate {env_name}"
        pip_cmd = f"conda run -n {env_name} pip"
        python_cmd = f"conda run -n {env_name} python"
    else:
        activate_cmd = f"conda activate {env_name}"
        pip_cmd = f"conda run -n {env_name} pip"
        python_cmd = f"conda run -n {env_name} python"

    return activate_cmd, pip_cmd, python_cmd


def install_dependencies(env_name="t4bdev"):
    """Install project dependencies using conda environment."""

    _, pip_cmd, _ = get_conda_commands(env_name)

    # Install the package in development mode with development dependencies
    print("Installing Twin4Build in development mode with dev dependencies...")
    result = run_command(f"{pip_cmd} install -e .[dev]")
    if result is None:
        return False

    print("✅ Dependencies installed in conda environment")
    return True


def run_tests(env_name="t4bdev"):
    """Run the test suite to verify setup using conda environment."""
    print("\n" + "=" * 50)
    print("🧪 Running test suite to verify installation...")
    print("=" * 50)
    print("ℹ️  This may take a few minutes depending on your system")
    print("ℹ️  Testing will verify that Twin4Build components work correctly")
    print("ℹ️  You'll see detailed test output below...")
    print("")

    _, _, python_cmd = get_conda_commands(env_name)

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
        print(f"   conda activate {env_name}")
        print("   python -m unittest discover twin4build/tests/ -v")
        return False

    print("\n✅ All tests passed!")
    print("🎉 Your Twin4Build development environment is working correctly")
    return True


def print_next_steps(env_name="t4bdev"):
    """Print next steps for the developer."""
    activate_cmd, _, _ = get_conda_commands(env_name)

    print("\n" + "=" * 60)
    print("✅ Development environment setup complete!")
    print("\n🎯 Next steps:")
    print(f"   1. Activate the environment: conda activate {env_name}")
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Set up Twin4Build development environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_dev.py                   # Use Python 3.12 (default)
  python setup_dev.py --python 3.9      # Use Python 3.9
  python setup_dev.py --python 3.10     # Use Python 3.10
  python setup_dev.py --python 3.11     # Use Python 3.11
  python setup_dev.py --python 3.12     # Use Python 3.12
  python setup_dev.py --env myproject   # Use custom environment name
  python setup_dev.py --python 3.11 --env myproject  # Custom Python version and env name
        """,
    )
    parser.add_argument(
        "--python",
        default="3.12",
        help="Python version to use in conda environment (default: 3.12)",
    )
    parser.add_argument(
        "--env",
        default="t4bdev",
        help="Name of conda environment to create (default: t4bdev)",
    )
    args = parser.parse_args()

    print("Twin4Build Development Environment Setup")
    print("=" * 40)

    # Check and change to correct directory
    if not check_and_change_directory():
        sys.exit(1)

    print(
        f"\n🐍 This script will create a conda environment for Twin4Build development."
    )
    print(f"📦 Environment name: '{args.env}'")
    print(f"🐍 Python version: {args.python}")
    print("🔒 This will NOT modify your base conda environment.")

    # Ask for confirmation
    response = input("\nDo you want to continue? (y/N): ").strip().lower()
    if response not in ("y", "yes"):
        print("Setup cancelled.")
        print("\n💡 To set up manually:")
        print(f"   conda create -n {args.env} python={args.python}")
        print(f"   conda activate {args.env}")
        print("   pip install -e .")
        print("   pip install -e .[dev]")
        return

    # Check conda availability
    if not check_conda_available():
        sys.exit(1)

    # Note: Removed check_python_version() call since it checks the current interpreter
    # version, not the conda environment version that will be created

    # Create conda environment with specified Python version
    if not create_conda_environment(args.python, args.env):
        sys.exit(1)

    # Install dependencies
    if not install_dependencies(args.env):
        sys.exit(1)

    # Run tests
    run_tests(args.env)

    # Print next steps
    print_next_steps(args.env)


if __name__ == "__main__":
    main()
