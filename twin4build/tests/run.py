#!/usr/bin/env python
"""
Script to run unittest discovery with coverage measurement.
Discovers tests from the new refactored folder structure that mirrors twin4build.

Test folder structure:
tests/
├── estimator/           - Tests for twin4build.estimator
├── examples/            - Tests for twin4build.examples
├── model/               - Tests for twin4build.model
│   ├── semantic_model/  - Tests for twin4build.model.semantic_model
│   └── simulation_model/- Tests for twin4build.model.simulation_model
├── optimizer/           - Tests for twin4build.optimizer
├── simulator/           - Tests for twin4build.simulator
├── systems/             - Tests for twin4build.systems
│   ├── air_to_air_heat_recovery/
│   ├── building_space/
│   ├── coil/
│   ├── controller/
│   ├── damper/
│   ├── fan/
│   ├── junction/
│   ├── outdoor_environment/
│   ├── schedule/
│   ├── sensor/
│   ├── space_heater/
│   ├── utils/
│   └── valve/
├── translator/          - Tests for twin4build.translator
└── utils/               - Tests for twin4build.utils
"""

# Standard library imports
import os
import sys
import unittest
import webbrowser

try:
    # Third party imports
    import coverage
except ImportError:
    print("Error: coverage module not installed.")
    print("Install it with: pip install coverage")
    sys.exit(1)

# Set test flag
import twin4build
twin4build._IS_TESTING = True


def main():
    # Initialize coverage
    cov = coverage.Coverage()

    # Start measuring coverage
    cov.start()

    try:
        # Get the directory containing the tests
        test_dir = os.path.dirname(os.path.abspath(__file__))

        # Create a test loader and discover tests
        loader = unittest.TestLoader()

        # Discover all tests in the current directory and subdirectories
        # This will find all test_*.py files in the new folder structure
        tests = loader.discover(start_dir=test_dir, pattern="test_*.py")

        # Print discovered test count
        print(f"Discovered {tests.countTestCases()} tests")

        # Create a test runner
        runner = unittest.TextTestRunner(verbosity=2)

        # Run the tests
        result = runner.run(tests)

        cov.stop()

        if True:
            # Stop measuring coverage

            # Save coverage data
            cov.save()

            print("\n" + "=" * 70)
            print("Generating coverage reports...")
            print("=" * 70)

            # Generate text report to console
            print("\nCoverage Summary:")
            print("-" * 70)
            cov.report()

            # Generate HTML report
            print("\n" + "=" * 70)
            print("Generating HTML coverage report...")

            # html_report() returns the coverage percentage, not the directory
            coverage_percent = cov.html_report()

            # The HTML report is generated in the 'htmlcov' directory by default
            html_dir = "htmlcov"
            index_path = os.path.join(html_dir, "index.html")
            abs_index_path = os.path.abspath(index_path)

            print(f"HTML report generated in: {os.path.abspath(html_dir)}")
            print(f"Coverage: {coverage_percent:.1f}%")
            print(f"Opening report in browser...")
            print("=" * 70)

            # Open the HTML report in the default browser
            webbrowser.open(f"file://{abs_index_path}")

        # Exit with appropriate code based on test results
        if result.wasSuccessful():
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        # Stop coverage even if there's an error
        cov.stop()
        cov.save()
        print(f"\nError occurred: {e}", file=sys.stderr)
        sys.exit(1)


def _match_pattern(test_module: str, pattern: str) -> bool:
    """Check if a test module matches a pattern.
    
    Args:
        test_module: Full module path (e.g., 'optimizer.test_optimizer')
        pattern: Pattern to match (e.g., 'optimizer', 'test_optimizer')
        
    Returns:
        True if the pattern matches the module path
    """
    return (
        f".{pattern}." in test_module or 
        test_module.startswith(f"{pattern}.") or
        test_module.endswith(f".{pattern}") or
        test_module == pattern
    )


def _filter_tests_including(suite, include_patterns: list[str]) -> unittest.TestSuite:
    """Recursively filter tests, including only those matching specified patterns.
    
    Args:
        suite: Test suite to filter
        include_patterns: List of patterns to include (e.g., ['optimizer', 'simulator'])
        
    Returns:
        Filtered test suite containing only matching tests
    """
    filtered = unittest.TestSuite()
    
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            # Recursively filter nested suites
            filtered.addTests(_filter_tests_including(test, include_patterns))
        else:
            # Check if test module path matches any included pattern
            test_module = test.__class__.__module__
            should_include = any(
                _match_pattern(test_module, pattern)
                for pattern in include_patterns
            )
            if should_include:
                filtered.addTest(test)
    
    return filtered


def _filter_tests_excluding(suite, exclude_patterns: list[str]) -> unittest.TestSuite:
    """Recursively filter tests, excluding those matching specified patterns.
    
    Args:
        suite: Test suite to filter
        exclude_patterns: List of patterns to exclude (e.g., ['examples', 'optimizer'])
        
    Returns:
        Filtered test suite
    """
    filtered = unittest.TestSuite()
    
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            # Recursively filter nested suites
            filtered.addTests(_filter_tests_excluding(test, exclude_patterns))
        else:
            # Check if test module path matches any excluded pattern
            test_module = test.__class__.__module__
            should_exclude = any(
                _match_pattern(test_module, pattern)
                for pattern in exclude_patterns
            )
            if not should_exclude:
                filtered.addTest(test)
    
    return filtered


def run_tests_including(include_patterns: list[str]):
    """Run only tests matching specified patterns.

    Args:
        include_patterns: List of patterns to include (e.g., ['optimizer', 'simulator'])
    """
    cov = coverage.Coverage()
    cov.start()

    try:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        loader = unittest.TestLoader()
        
        # Discover all tests
        all_tests = loader.discover(start_dir=test_dir, pattern="test_*.py")
        total_before = all_tests.countTestCases()
        
        # Filter to only included patterns
        filtered_tests = _filter_tests_including(all_tests, include_patterns)
        total_after = filtered_tests.countTestCases()
        
        print(f"Discovered {total_before} tests, running {total_after} (matching {include_patterns})")

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(filtered_tests)

        cov.stop()
        cov.save()
        cov.report()

        if result.wasSuccessful():
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        cov.stop()
        cov.save()
        print(f"\nError occurred: {e}", file=sys.stderr)
        sys.exit(1)


def run_tests_excluding(exclude_patterns: list[str]):
    """Run all tests except those matching specified patterns.

    Args:
        exclude_patterns: List of patterns to exclude (e.g., ['examples', 'optimizer'])
    """
    cov = coverage.Coverage()
    cov.start()

    try:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        loader = unittest.TestLoader()
        
        # Discover all tests
        all_tests = loader.discover(start_dir=test_dir, pattern="test_*.py")
        total_before = all_tests.countTestCases()
        
        # Filter out excluded patterns
        filtered_tests = _filter_tests_excluding(all_tests, exclude_patterns)
        total_after = filtered_tests.countTestCases()
        
        print(f"Discovered {total_before} tests, running {total_after} (excluded {total_before - total_after} matching {exclude_patterns})")

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(filtered_tests)

        cov.stop()
        cov.save()
        cov.report()

        if result.wasSuccessful():
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        cov.stop()
        cov.save()
        print(f"\nError occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # To run all tests:
    # main()

    # Run all tests EXCEPT examples:
    # run_tests_excluding(["examples"])

    # Run only tests matching patterns:
    run_tests_including(["optimizer"])
