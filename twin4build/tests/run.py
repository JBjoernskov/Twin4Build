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


def run_specific_tests(patterns: list[str]):
    """Run tests matching specific patterns.

    Args:
        patterns: List of patterns to match (e.g., ['test_model.py', 'test_simulator.py'])
    """
    cov = coverage.Coverage()
    cov.start()

    try:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        tests = unittest.TestSuite()

        for pattern in patterns:
            suite = unittest.TestLoader().discover(start_dir=test_dir, pattern=pattern)
            for test_group in suite:
                tests.addTests(test_group)
            print(f"Discovered {suite.countTestCases()} tests from {pattern}")

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(tests)

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
    main()

    # Run specific tests that were previously failing:
    # run_specific_tests(
    #     [
    #         "test_building_space_torch_system.py",  # BuildingSpaceTorchSystem tests
    #         "test_utility_systems.py",  # MaxSystem, OnOffSystem, PassInputToOutput, PiecewiseLinearSystem
    #         "test_utils.py",  # sample_from_df test
    #         "test_plot.py",  # plot_component tests
    #     ]
    # )
