# Twin4Build Development Scripts

This directory contains scripts to help with Twin4Build development.

## Scripts

### `setup_dev.py`
**Purpose**: Automates development environment setup

**Usage**:
```bash
python scripts/setup_dev.py
```

**What it does**:
- Creates conda environment named `t4bdev`
- Installs Twin4Build in development mode
- Installs development dependencies
- Runs tests to verify setup

### `validate_code.py`
**Purpose**: Validates code quality before committing

**Usage**:
```bash
# Check code quality (basic validation)
python scripts/validate_code.py

# Fix formatting and import issues automatically
python scripts/validate_code.py --fix

# Include test suite in validation
python scripts/validate_code.py --test

# Combine options: fix issues and run tests
python scripts/validate_code.py --fix --test
```

**What it checks**:
- ✅ Code formatting (Black)
- ✅ Import sorting (isort)
- ✅ Code style (flake8)
- ✅ File format issues
- ⚠️ Test suite - optional

**Sample output**:
```
============================================================
🔍 Checking code formatting with Black
============================================================
Running: black --check --diff .

✅ Checking code formatting with Black - PASSED

============================================================
📊 VALIDATION SUMMARY
============================================================
Code formatting (Black)      ✅ PASSED
Import sorting (isort)       ✅ PASSED
Code style (flake8)          ✅ PASSED
File issues                  ✅ PASSED

Total: 4/4 checks passed

🎉 All validation checks passed!
✅ Your code is ready for commit/pull request
```

## Development Workflow

1. **Setup**: Run `python scripts/setup_dev.py` once
2. **Develop**: Write your code
3. **Validate**: Run `python scripts/validate_code.py` before committing
4. **Commit**: `git commit -m "Your message"`

## Tips

- **Auto-fix**: Use `--fix` flag to automatically resolve formatting issues
- **Failed validation**: The script will show exactly what needs to be fixed
- **Manual tools**: You can run individual tools (black, flake8, etc.) manually if needed 