# Contributing to Twin4Build

Thank you for your interest in contributing to Twin4Build! This document provides a quick overview of how to get started.

## Quick Start

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create** a feature branch: `git checkout -b feature/your-feature-name`
4. **Make** your changes following our [code style guidelines](docs/source/manual/developer_reference.rst#code-style-and-conventions)
5. **Validate** your code: `python scripts/validate_code.py`
6. **Test** your changes: `python -m unittest discover twin4build/tests/`
7. **Commit** with a descriptive message
8. **Push** to your fork and create a Pull Request

## Development Setup

### Using Conda (Recommended)
```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/Twin4Build.git
cd Twin4Build

# Create conda environment
conda create -n t4bdev python=3.9
conda activate t4bdev

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### Alternative Environment Managers
You can also use **venv**, **virtualenv**, **poetry**, or **pipenv** - just ensure you have an isolated Python 3.9+ environment.

## What to Contribute

We welcome contributions in many forms:

- **Bug fixes**: Report bugs via GitHub Issues
- **New features**: Discuss major features via Issues first
- **Documentation**: Improve docs, add examples, fix typos
- **Tests**: Add test coverage for existing or new functionality
- **Examples**: Create new example notebooks or scripts

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Add tests for new functionality using unittest
- Keep commits focused and well-described

## Code Quality Validation

Before committing, **always run the validation script**:

```bash
# Check code quality (recommended before every commit)
python scripts/validate_code.py

# Auto-fix formatting and import issues
python scripts/validate_code.py --fix
```

This script checks:
- Code formatting (Black)
- Import sorting (isort)  
- Code style (flake8)
- Type checking (mypy)
- File format issues
- Test suite

## Testing

We use Python's built-in `unittest` framework. To run tests:

```bash
# Run all tests
python -m unittest discover twin4build/tests/

# Run specific test file
python -m unittest twin4build.tests.test_examples

# Run with coverage
coverage run -m unittest discover twin4build/tests/
coverage report
```

## Getting Help

- **Documentation**: [Developer Reference](docs/source/manual/developer_reference.rst)
- **Issues**: [GitHub Issues](https://github.com/JBjoernskov/Twin4Build/issues)
- **Examples**: Check the [examples directory](twin4build/examples/)

## License

By contributing to Twin4Build, you agree that your contributions will be licensed under the MIT License.

For more detailed information, see the [Developer Reference](docs/source/manual/developer_reference.rst). 