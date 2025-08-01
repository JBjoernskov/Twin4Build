# Contributing to Twin4Build

Thank you for your interest in contributing to Twin4Build! 

## Quick Start

1. **Fork** the repository and **clone** your fork locally
2. **Setup environment**: `conda create -n t4bdev python=3.9 && conda activate t4bdev`
3. **Install in dev mode**: `pip install -e .[dev]`
4. **Create feature branch**: `git checkout -b feature/your-feature-name`
5. **Make changes** following our coding standards
6. **Validate code**: `python scripts/validate_code.py --fix`
7. **Run tests**: `python -m unittest discover twin4build/tests/`
8. **Push and create Pull Request**

## Before You Start

- Check existing [Issues](https://github.com/JBjoernskov/Twin4Build/issues) for known bugs or planned features
- For major features, **discuss via Issues first**
- Review our [Developer Reference](docs/source/manual/developer_reference.rst) for detailed guidelines

## What We Need Help With

- **Bug fixes** - Report and fix issues
- **New features** - Building systems, components, algorithms  
- **Documentation** - Examples, tutorials, API improvements
- **Testing** - Add coverage for existing or new functionality

## Code Quality

Before submitting:
- Run `python scripts/validate_code.py --fix` to auto-format and check code
- Ensure tests pass: `python -m unittest discover twin4build/tests/`
- Follow PEP 8 and add type hints + docstrings

## Complete Developer Guide

For comprehensive information on:
- Architecture and package structure
- Detailed setup instructions for different environments
- Code style guidelines and examples  
- Testing strategies and documentation building
- Advanced topics like creating custom components

**👉 See the [Developer Reference](docs/source/manual/developer_reference.rst)**

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/JBjoernskov/Twin4Build/issues)
- **Examples**: [examples directory](twin4build/examples/)
- **Documentation**: [Developer Reference](docs/source/manual/developer_reference.rst)

By contributing, you agree that your contributions will be licensed under the MIT License. 